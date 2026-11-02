"""Run the fixed two-arm Gemma development experiment in PLAN.md."""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from fit_codec import parameter_sha, CHECKPOINT_SHA
from selffold import prompts
from selffold.calibration import codec_examples
from selffold.carrier_probe import capture_description, neutral_replay
from selffold.cli import ROOT, make_host
from selffold.codec_learning import EncoderBridge, codec_objectives, load_bridge, save_bridge
from selffold.policy_binding import prepare_pairs, pair_schedule, raw_readout, world_question
from selffold.records import RunStore, State, code_manifest, file_sha, digest
from selffold.resources import inventory

DATA = ROOT / "data/codec_preparation_20261102"
PLAN = ROOT / "provenance/policy_binding_20261103/PLAN.md"
ARMS = {"copy": 0., "opposition": 1.}


def fit(host, cases, slots, schedule, arm, store, source):
    bridge = EncoderBridge(host.width, 16).to(host.device)
    start_sha = parameter_sha(bridge)
    count = sum(p.numel() for p in bridge.parameters())
    if count != 73728:
        raise ValueError("Unexpected trainable parameter count")
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=3e-4, weight_decay=0.)
    start = host.meter.snapshot()
    for step, batch in enumerate(schedule, 1):
        optimizer.zero_grad(set_to_none=True)
        copy_losses, binary_losses = [], []
        for case in batch:
            row = cases[case]
            nll, margin = codec_objectives(host, bridge, slots[case], row["ids"], contrast=(row["index"], row["foil"]))
            binary = F.softplus(-margin)
            ((nll + ARMS[arm] * binary) / len(batch)).backward()
            copy_losses.append(nll.item())
            binary_losses.append(binary.item())
        if any(p.grad is not None for p in host.model.parameters()):
            raise RuntimeError("Backbone received a gradient")
        norm = torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1., error_if_nonfinite=True).item()
        if norm <= 0:
            raise ValueError("No bridge gradient")
        optimizer.step()
        store.append("training.jsonl", {"arm": arm, "step": step, "cases": batch, "nll": copy_losses,
                                        "opposed_token_ce": binary_losses, "gradient_norm": norm})
    end = host.meter.snapshot()
    work = {k: end[k] - start[k] for k in ("forwards", "processed_tokens")}
    config = {"completed_steps": len(schedule), "schedule_sha256": digest(schedule), "contrast_weight": ARMS[arm],
              "plan_sha256": file_sha(PLAN), "arm": arm, "initial_parameters_sha256": start_sha}
    path = store.path / (arm + ".npz")
    sha = save_bridge(path, bridge, host.identity, dataset_sha256=file_sha(DATA / "manifest.json"),
                      source_sha256=source["source_sha256"], training=config)
    restored, _, check = load_bridge(path, host.identity, device=host.device)
    if check != sha or parameter_sha(bridge) != parameter_sha(restored):
        raise ValueError("Artifact roundtrip failed")
    store.append("fits.jsonl", {**config, **work, "parameters": count, "sha256": sha,
                                "final_parameters_sha256": parameter_sha(restored)})
    return restored, (start_sha, work)


def score_prefix(host, module, dev, slots, arm, store):
    prompt, positions = host.render(prompts.read_prompt())
    for case, row in dev.items():
        i, ids, foil = row["index"], row["ids"], row["foil"]
        nll, margin = codec_objectives(host, module, slots[case], ids, contrast=(i, foil))
        state = State(slots=module(slots[case]))
        masked = prompt + ids[:i] + [host.placeholder_id] * (len(ids)-i)
        logits, _ = host.forward(masked, state=state, input_positions=positions[prompts.INPUT_MARKER])
        at = logits[0, len(prompt)+i-1].float()
        replay = (at[ids[i]] - at[foil]).item()
        row_out = {"case": case, "group": row["group"], "arm": arm, "word": row["word"], "index": i,
                   "target_token": ids[i], "foil_token": foil, "tokens_including_eot": len(ids),
                   "prefix_sha256": digest(prompt + ids[:i]), "state_sha256": state.sha(), "source_nll": nll.item(),
                   "margin": margin.item(), "masked_suffix_replay_margin": replay}
        store.append("prefix.jsonl", row_out)
        if abs(margin.item() - replay) > 1e-5:
            raise ValueError("Full-source and common-prefix replay differ")


def functional(host, bridges, dev, groups, slots, train, store):
    train_norm = torch.stack([slots[k].float().norm(dim=1).mean() for k in train]).mean()
    random = F.normalize(torch.randn((host.slots, host.width), generator=torch.Generator().manual_seed(9292)), dim=1)
    random = (random.to(host.device) * train_norm).to(host.dtype)
    first = slots["conditional-00-plus-p0"]
    controls = {"zero": torch.zeros_like(first), "random": random, "unrelated": first}
    cache, checked = {}, set()
    for arm, bridge in bridges.items():
        fixed = {name: State(slots=bridge(s)) for name, s in controls.items()}
        for case, row in dev.items():
            group, pole, wording = groups[row["group"]], row["pole"], row["wording"]
            states = {"matched": State(slots=bridge(slots[case])), "literal": State(text=row["source"]),
                      "empty": None, **fixed}
            for route, state in states.items():
                fingerprint = state.sha() if state is not None else "empty"
                for world_index in range(len(group["poles"][pole]["truth_ledger"])):
                    for order in (0, 1):
                        question, gold = world_question(group, pole, world_index, order)
                        if question not in checked:
                            store.append("neutral.jsonl", {"question": question, **neutral_replay(host, question)})
                            checked.add(question)
                        key = (fingerprint, question)
                        if key not in cache:
                            scores = host.score_choices(state, question, ["A", "B"])
                            cache[key] = {k: v["logprob"] for k, v in scores.items()}
                        scores = cache[key]
                        store.append("functional.jsonl", {"case": case, "group": group["id"], "pole": pole,
                            "wording": wording, "world": world_index, "order": order, "arm": arm, "route": route,
                            "state_sha256": fingerprint, "A": scores["A"], "B": scores["B"], "gold": gold})
                if state is not None and fingerprint != state.sha():
                    raise ValueError("Functional readout mutated a source")
    return len(cache)


def run(args, store):
    source = code_manifest(ROOT)
    store.write("source.json", source)
    if source["git_status"] != "" or not source["git_commit"]:
        raise ValueError("Clean source required")
    manifest = json.loads((DATA / "manifest.json").read_text())
    if manifest["qualification"]["status"] != "sealed_before_fitting":
        raise ValueError("Missing qualification commitment")
    store.write("request.json", {"arms": ARMS, "data_sha256": file_sha(DATA / "manifest.json"),
              "plan_sha256": file_sha(PLAN), "qualification_commitment": manifest["qualification"]["sha256"],
              "steps_per_arm": 64, "effective_batch": 8, "selection": "fixed_final_step"})
    store.write("environment.json", inventory(ROOT))
    host, checkpoint = make_host(args)
    store.write("host.json", host.identity)
    store.write("checkpoint.json", checkpoint)
    parity = host.native_parity()
    store.write("native_parity.json", parity)
    end_id = host.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if checkpoint["checkpoint_sha256"] != CHECKPOINT_SHA or not parity["exact"] or end_id != 107 or end_id not in host.eos_ids:
        raise ValueError("Host identity, native parity, or EOT mismatch")
    before = parameter_sha(host.model)
    train, train_pairs = prepare_pairs(codec_examples(DATA, "train"), host.tokenizer, end_id)
    dev, dev_pairs = prepare_pairs(codec_examples(DATA, "dev"), host.tokenizer, end_id)
    if (len(train), len(train_pairs), len(dev), len(dev_pairs)) != (192, 96, 96, 48) or train.keys() & dev.keys():
        raise ValueError("Invalid paired split")
    slots = {}
    for case, row in (train | dev).items():
        state, layout = capture_description(host, row["source"], "workspace")
        slots[case] = state.slots
        store.append("donors.jsonl", {"case": case, "source_sha256": digest(row["source"]),
                     "state_sha256": state.sha(), "positions": layout["capture_positions"]})
    rows, flipped = raw_readout(train, dev, slots)
    store.write("sham_sites.json", flipped)
    for row in rows:
        store.append("raw_readout.jsonl", row)
    schedule = pair_schedule(train_pairs)
    bridges, budgets = {}, {}
    for arm in ARMS:
        bridges[arm], budgets[arm] = fit(host, train, slots, schedule, arm, store, source)
    if budgets["copy"] != budgets["opposition"] or budgets["copy"][1]["forwards"] != 512:
        raise ValueError("Arms differ in initialization or training compute")
    # Both terminal artifacts exist and are fixed before any dev task is scored.
    groups = {g["id"]: g for g in map(json.loads, (DATA / "dev.groups.jsonl").read_text().splitlines())}
    with torch.no_grad():
        for arm, module in bridges.items():
            score_prefix(host, module, dev, slots, arm, store)
        unique_scores = functional(host, bridges, dev, groups, slots, train, store)
    after = parameter_sha(host.model)
    if before != after:
        raise ValueError("Backbone parameters changed")
    return {"status": "COMPLETED_PAIRED_DEVELOPMENT_COMPARISON", "compute": host.meter.snapshot(),
            "unique_functional_state_questions": unique_scores, "backbone_sha256_before": before,
            "backbone_sha256_after": after, "G_teacher_used_by_loss": False, "qualification_content_loaded": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(backend="hf", cpu_threads=1, device="cuda:0", dtype="bfloat16", slots=8, cut=None,
                   writer=None, max_context=1024, max_new_tokens=None, until_eos=True, transport=None,
                   encoder_bridge=None, reviewed_local_code=None, wall_seconds=1800, profile=ROOT / "configs/models/gemma2b.yaml")
    args = p.parse_args()
    store, start = RunStore(args.out), time.monotonic()
    try:
        terminal = run(args, store)
    except Exception as e:
        store.write("terminal.json", {"status": "INVALID_STOPPED", "error": str(e), "type": type(e).__name__,
                                      "elapsed_seconds": time.monotonic()-start})
        raise
    store.write("terminal.json", {**terminal, "elapsed_seconds": time.monotonic()-start})


if __name__ == "__main__":
    main()
