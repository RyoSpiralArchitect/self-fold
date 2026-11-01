"""Score paired descriptions with a fixed question, crossed label order and slot controls."""
import argparse
import json
from pathlib import Path
import time

import torch

from selffold import prompts
from selffold.cli import ROOT, make_host
from selffold.records import RunStore, State, code_manifest, digest, file_sha, load_state
from selffold.resources import inventory
from selffold.replay_probe import run_replay_probe
from selffold.rng import isolated_rng


DATA = ROOT / "data/slot_contrasts.dev.json"
ARCHIVE = ROOT / "provenance/mistral_eos_20261101/trials"
MODELS = {
    "mistral7b": ("c170c708c41dac9275d15a8fff4eca08d52bab71",
                  "5d141a13ea9f5fce9c142bc26bdef64ea905fd9d27c64aff472f6875dd4296e9"),
    "gemma2b": ("299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
                "f3503907cc4d2ace62abccf30393a05919827c1c8969b2f2755f15627b9756ad"),
}


def score(host, store, group, route, state, *, pole=None, phrasing=None, slot_token_ids=None):
    fingerprint = state.sha() if state is not None else None
    plus, minus = group["options"]
    orders = []
    with isolated_rng():
        for plus_label, first, second in (("A", plus, minus), ("B", minus, plus)):
            question = f"{group['question']}\nA = {first}. B = {second}. Use the supplied description."
            if slot_token_ids is None:
                scores = host.score_choices(state.clone() if state is not None else None, question, ["A", "B"])
            else:
                scores = host.score_choices(None, question, ["A", "B"], slot_token_ids=slot_token_ids)
            delta = scores["A"]["logprob"] - scores["B"]["logprob"]
            orders.append({"question": question, "plus_label": plus_label, "scores": scores,
                           "plus_margin": delta if plus_label == "A" else -delta})
    if state is not None and state.sha() != fingerprint:
        raise RuntimeError("A scored state changed during readout")
    store.append("scores.jsonl", {"group": group["id"], "pole": pole, "phrasing": phrasing,
                                 "route": route, "state_sha256": fingerprint, "orders": orders,
                                 "plus_margin": sum(row["plus_margin"] for row in orders) / 2})


def magnitude_control(state):
    # Reuse one direction per slot across every donor of the same width.
    cpu_slots = state.slots.float().cpu()
    directions = torch.randn(cpu_slots.shape, generator=torch.Generator().manual_seed(4711))
    directions = torch.nn.functional.normalize(directions, dim=1)
    norms = cpu_slots.norm(dim=1, keepdim=True)
    values = (directions * norms).to(state.slots)
    error = (values.float().cpu().norm(dim=1, keepdim=True) - norms).abs() / norms.clamp_min(1e-12)
    return State(slots=values), error.max().item()


def archived(host, store, group, zero):
    score(host, store, group, "uninjected", None)
    score(host, store, group, "zero", zero)
    origins = {}
    for pole in ("plus", "minus"):
        folder = ARCHIVE / f"caution_{pole}_p0_self-eos"
        identity = json.loads((folder / "host_identity.json").read_text())
        if host.identity != identity:
            raise ValueError("The saved state belongs to a different host")
        origins[pole] = {"directory": str(folder.relative_to(ROOT)),
                         "source_commit": json.loads((folder / "source.json").read_text())["git_commit"]}
        for route, filename in (("text", "T_000"), ("E", "L_000"), ("L1", "L_001"),
                                ("L2", "L_002"), ("B1", "B_001"), ("B2", "B_002")):
            state = load_state(folder / "states" / filename, device=host.device)
            score(host, store, group, route, state, pole=pole, phrasing=0)
    store.write("archive_sources.json", origins)


def verbalize(host, store, case_id, route, state):
    with isolated_rng():
        output = host.generate(prompts.read_prompt()) if state is None else host.decode(state.clone())
    store.append("generations.jsonl", {"case": case_id, "route": route, **output.receipt()})
    return output


def contrasts(host, store, groups, zero):
    verbalize(host, store, None, "uninjected", None)
    verbalize(host, store, None, "zero", zero)
    for group in groups:
        score(host, store, group, "uninjected", None)
        score(host, store, group, "zero", zero)
        for pole in ("plus", "minus"):
            for phrasing, text in enumerate(group[pole]):
                case_id = f"{group['id']}-{pole}-p{phrasing}"
                encoded = host.encode(text)
                updated = host.update_latent(encoded.clone(), "plain")
                verbalize(host, store, case_id, "E", encoded)
                decoded = verbalize(host, store, case_id, "G", updated)
                norm_sham, norm_error = magnitude_control(updated)
                states = {"text": State(text=text), "E": encoded, "G": updated,
                          "G_reverse": State(slots=updated.slots.flip(0)), "G_magnitude": norm_sham}
                if decoded.text:
                    states["B"] = host.encode(decoded.text)
                store.append("cases.jsonl", {"case": case_id, "text": text,
                                            "roundtrip_available": bool(decoded.text),
                                            "max_relative_slot_norm_error": norm_error})
                for route, state in states.items():
                    if route in {"E", "G", "B"}:
                        store.save_state(f"states/{case_id}/{route}", state)
                    score(host, store, group, route, state, pole=pole, phrasing=phrasing)


def lexical_alignment(host, store, groups):
    # The donor forward ends at the last slot, before any question or answer alternatives.
    template, positions = host.render(prompts.task_prompt(""))
    slots = positions[prompts.INPUT_MARKER]
    prefix = template[:slots[-1] + 1]
    assert slots == list(range(len(prefix) - host.slots, len(prefix)))

    def capture(slot_tokens):
        donor_prefix = prefix[:slots[0]] + slot_tokens
        _, captures = host.forward(donor_prefix, capture_positions=slots)
        return State(slots=captures["pre_cut"][0])

    neutral = capture([host.placeholder_id] * host.slots)
    store.write("prefix.json", {"native_prefix_ids": prefix, "slot_positions": slots,
                                "neutral_state_sha256": neutral.sha()})
    for group in groups:
        for first, second in (group["options"], group["options"][::-1]):
            q = f"{group['question']}\nA = {first}. B = {second}. Use the supplied description."
            ids, pos = host.render(prompts.task_prompt(q))
            if pos[prompts.INPUT_MARKER] != slots or ids[:len(prefix)] != prefix:
                raise ValueError("Donor and recipient prefix positions differ")
        score(host, store, group, "uninjected", None)
        score(host, store, group, "neutral_replay", neutral)
        for pole, words in zip(("plus", "minus"), group["options"], strict=True):
            tokens = host.tokenizer.encode(words, add_special_tokens=False)
            if not 1 <= len(tokens) <= host.slots:
                raise ValueError(f"Option does not fit the fixed slots: {group['id']} {pole}")
            padded = tokens + [host.placeholder_id] * (host.slots - len(tokens))
            encoded = host.encode(words)
            aligned = capture(padded)
            case = f"{group['id']}-{pole}"
            store.append("cases.jsonl", {"case": case, "words": words, "token_ids": tokens, "padded_ids": padded})
            store.save_state(f"states/{case}/E", encoded)
            store.save_state(f"states/{case}/aligned", aligned)
            score(host, store, group, "native_words", None, pole=pole, phrasing=0, slot_token_ids=padded)
            for route, donor in (("E_words", encoded), ("aligned", aligned),
                                 ("aligned_reverse", State(slots=aligned.slots.flip(0)))):
                score(host, store, group, route, donor, pole=pole, phrasing=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["archive", "contrasts", "alignment", "replay"])
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(backend="hf", cpu_threads=1, device="cuda:0", dtype="bfloat16",
                        slots=8, cut=None, writer=None, max_context=1024, max_new_tokens=None,
                        until_eos=True, wall_seconds=1200, transport=None, reviewed_local_code=None)
    args = parser.parse_args()
    if args.study == "archive" and args.model != "mistral7b":
        parser.error("Archived donors are from Mistral")
    args.profile = ROOT / "configs/models" / f"{args.model}.yaml"
    store = RunStore(args.out)
    start = time.monotonic()
    source = code_manifest(ROOT)
    store.write("source.json", source)
    store.write("environment.before.json", inventory(ROOT))
    revision, expected_digest = MODELS[args.model]
    store.write("request.json", {"study": args.study, "model": args.model, "snapshot_revision": revision,
                                 "checkpoint_sha256": expected_digest, "fixture_sha256": file_sha(DATA),
                                 "split": "development", "operator": "plain", "sham_seed": 4711})
    try:
        if source["git_status"] != "" or not source["git_commit"]:
            raise ValueError("Run the frozen study from a clean source commit")
        host, identity = make_host(args)
        store.write("checkpoint_identity.json", identity)
        store.write("host_identity.json", host.identity)
        if identity["checkpoint_sha256"] != expected_digest:
            raise ValueError("Checkpoint fingerprint differs from the fixed experiment")
        parity = host.native_parity()
        store.write("native_parity.json", parity)
        if not parity["exact"]:
            raise ValueError("Native parity failed")
        data = json.loads(DATA.read_text())
        zero = State(slots=torch.zeros((host.slots, host.width), device=host.device, dtype=host.dtype))
        if args.study == "archive":
            archived(host, store, data["archive"], zero)
        elif args.study == "alignment":
            lexical_alignment(host, store, data["groups"])
        elif args.study == "replay":
            run_replay_probe(host, store, data["groups"])
        else:
            contrasts(host, store, data["groups"], zero)
        store.write("environment.after.json", inventory(ROOT))
        receipt = {"status": "COMPLETED_DEVELOPMENT_PROBE", "compute": host.meter.snapshot(),
                   "host_sha256": digest(host.identity), "elapsed_seconds": time.monotonic() - start}
        store.write("terminal.json", receipt)
        print(json.dumps(receipt))
    except Exception as exc:
        store.write("terminal.json", {"status": "FAILED", "error_type": type(exc).__name__,
                                      "message": str(exc), "elapsed_seconds": time.monotonic() - start})
        raise


if __name__ == "__main__":
    main()
