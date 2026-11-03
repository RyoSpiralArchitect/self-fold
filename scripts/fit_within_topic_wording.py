"""Matched one-style and four-style maps in the eight Section 6.12 fitting topics."""

import argparse
import json
from pathlib import Path
import time

import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_continuity_recipient import batch_inputs, model_scores, training_loss
from fit_source_lexical import (
    DATA as PREVIOUS_DATA,
    LEXICAL_SHA,
    PRIOR,
    capture_lexical,
    replay_previous,
    source_examples,
    rows,
)
from probe_slot_readout import update
from selffold.cli import ROOT, make_host
from selffold.records import RunStore, State, code_manifest, digest, file_sha
from selffold.recipient_learning import RecipientAdapter, batched_recipient_logits
from selffold.source_mapping import load_lexical_interface, load_source_mapper
from selffold.wording_transfer import (
    fit_policy_directions,
    save_policy_mixture,
    load_policy_mixture,
)

DATA = ROOT / "data/within_topic_wording_20261103.json"
STUDY = ROOT / "provenance/within_topic_20261103"
OLD = ROOT / "provenance/source_lexical_20261103/run"
ARMS = ("single", "many")
# Each block has two stems in both displayed orders for a single E state.
PANELS = {
    "fit": ("fit", (0, 1)),
    "questions": ("fit", (2, 3)),
    "source": ("fresh", (0, 1)),
    "joint": ("fresh", (2, 3)),
    "syntax": ("syntax", (2, 3)),
    "lexicon": ("lexicon", (2, 3)),
}


def make_bank(data, arm):
    if arm not in ARMS:
        raise ValueError("Unknown fitting arm")
    bank = {part: [] for part in ("fit", "fresh", "syntax", "lexicon")}
    for index, group in enumerate(data["groups"]):
        fit = [
            (
                data["fit_styles"][(a + b + index) % 4]
                if arm == "many"
                else data["fit_styles"][0],
                a,
                b,
            )
            for a in data["fit_lexeme_indices"]
            for b in data["fit_lexeme_indices"]
        ]
        fresh = [
            (style, a, a)
            for style in data["fresh_styles"]
            for a in data["fresh_lexeme_indices"]
        ]
        syntax = [(style, 0, 0) for style in data["fresh_styles"]]
        lexicon = [(data["fit_styles"][0], a, a) for a in data["fresh_lexeme_indices"]]
        for part, combinations in (
            ("fit", fit),
            ("fresh", fresh),
            ("syntax", syntax),
            ("lexicon", lexicon),
        ):
            for style, a, b in combinations:
                form = data["styles"][style]
                for polarity, pole in enumerate(("plus", "minus")):
                    options = (group["plus"][a], group["minus"][b])
                    clauses = (
                        form["chosen"].format(options[polarity]),
                        form["rejected"].format(options[1 - polarity]),
                    )
                    for order in (0, 1):
                        stem = f"{group['id']}/{part}/{a}-{b}/{style}"
                        bank[part].append(
                            {
                                "key": f"{stem}/{pole}/{order}",
                                "unit": f"{a}-{b}/{style}",
                                "opposed_key": f"{stem}/{'plus' if polarity else 'minus'}/{order}",
                                "group": group["id"],
                                "class_id": 2 * index + polarity,
                                "pole": pole,
                                "style": style,
                                "a": a,
                                "b": b,
                                "source_order": order,
                                "source": " ".join(
                                    clauses if order == 0 else clauses[::-1]
                                ),
                            }
                        )
    return bank


def tasks_for(case, group, wordings):
    result = []
    for wording in wordings:
        for order, options in enumerate((group["options"], group["options"][::-1])):
            question = f"{group['questions'][wording]}\nA = {options[0]}. B = {options[1]}. Use the supplied description."
            gold = "A" if (case["pole"] == "plus") == (order == 0) else "B"
            result.append(
                {
                    "key": case["key"],
                    "source": case["source"],
                    "question": question,
                    "wording": wording,
                    "label_order": order,
                    "gold": gold,
                }
            )
    return result


def old_geometry(host, cached, store):
    import numpy as np

    with np.load(OLD / "ridge.npz", allow_pickle=False) as archive:
        anchors = torch.from_numpy(archive["anchors"]).double()
        mean = torch.from_numpy(archive["source_mean"]).double()
        scale = torch.from_numpy(archive["source_scale"]).double()
    eigen, basis = torch.linalg.eigh(anchors @ anchors.T)
    keep = eigen > eigen.max() * 1e-8
    axes = (basis[:, keep].T @ anchors) / eigen[keep].sqrt()[:, None]
    references = {r["key"]: r for r in rows(OLD / "sources.jsonl")}
    vectors = {}

    def record(key, style, kind, x):
        energy = x.square().sum()
        if energy <= 0:
            raise ValueError("Zero vector in old span diagnostic")
        fraction = (axes @ x).square().sum() / energy
        store.append(
            "geometry.jsonl",
            {
                "key": key,
                "style": style,
                "kind": kind,
                "span_energy_fraction": fraction.item(),
            },
        )

    for case in source_examples(json.loads(PREVIOUS_DATA.read_text())):
        if case["split"] not in ("seen_sources", "new_style"):
            continue
        text = case["source"]
        if text not in cached:
            cached[text] = host.encode(text)
        state = cached[text]
        if state.sha() != references[case["key"]]["state_sha256"]:
            raise ValueError("Archived source state replay failed")
        x = (state.slots.cpu().double().flatten() - mean) / scale
        vectors[case["group"], case["pole"], case["phrasing"]] = x
        record(case["key"], case["style"], "state", x)
    for (group, pole, phrasing), x in vectors.items():
        if pole == "plus":
            record(
                f"{group}/p{phrasing}",
                phrasing // 2,
                "opposed_difference",
                x - vectors[group, "minus", phrasing],
            )
    store.write(
        "geometry.json",
        {
            "rank": int(keep.sum()),
            "old_ridge_sha256": file_sha(OLD / "ridge.npz"),
            "source_replays": len(vectors),
            "relative_gram_cutoff": 1e-8,
        },
    )


def checked_save(
    host, lexical, adapter, model, first, tasks, store, arm, endpoint, meta
):
    source = first.slots

    def output(module):
        projected = module(source)
        supplied = {tasks[0]["key"]: lexical(projected)}
        logits = batched_recipient_logits(
            host, adapter, *batch_inputs(host, tasks, supplied, "matched")
        )
        return projected, logits

    with torch.no_grad():
        expected = output(model)
    path = store.path / f"{arm}_{endpoint}.npz"
    bound = meta | {
        "arm": arm,
        "endpoint": endpoint,
        "updates": 64 if endpoint == "final" else 0,
    }
    sha = save_policy_mixture(path, model, bound)
    restored, _, actual = load_policy_mixture(
        path, host.identity, LEXICAL_SHA, host.device
    )
    with torch.no_grad():
        result = output(restored)
    if actual != sha or any(
        not torch.equal(a, b) for a, b in zip(expected, result, strict=True)
    ):
        raise ValueError("Mixture serialization mismatch")
    store.append(
        "fits.jsonl",
        bound
        | {
            "artifact": path.name,
            "sha256": sha,
            "rows_exact": True,
            "logits_exact": True,
        },
    )
    return restored


def train_arm(
    host,
    lexical,
    adapter,
    bank,
    groups,
    states,
    teachers,
    schedule,
    store,
    arm,
    binding,
    first_attempt=None,
):
    fit = bank["fit"]
    x = torch.stack([states[c["source"]].slots for c in fit])
    labels = [c["class_id"] for c in fit]
    model, stats = fit_policy_directions(x, labels, teachers)
    model = model.to(host.device)
    fixed = {k: v.clone() for k, v in model.named_buffers()}
    meta = binding | {
        "source_bank_sha256": digest(
            [(c["key"], states[c["source"]].sha()) for c in fit]
        ),
        "fit_stats": stats,
    }
    first = fit[0]
    sample = tasks_for(first, groups[first["group"]], (0, 1))
    initial = checked_save(
        host,
        lexical,
        adapter,
        model,
        states[first["source"]],
        sample,
        store,
        arm,
        "initial",
        meta,
    )
    if arm == "single" and first_attempt is not None:
        import numpy as np

        if (
            file_sha(first_attempt / "single_initial.npz")
            != "f8c8d183c939ea99164457c4aef7b90b39d8e44a1702555be85edb97a3c2a0e4"
        ):
            raise ValueError("Unexpected initial-attempt artifact")
        replay_files = (
            "request.json",
            "sources.jsonl",
            "teachers.jsonl",
            "schedule.json",
            "lexical_replay.jsonl",
            "geometry.jsonl",
        )
        if any(
            file_sha(first_attempt / name) != file_sha(store.path / name)
            for name in replay_files
        ):
            raise ValueError("Initial-attempt capture or design replay failed")
        with (
            np.load(first_attempt / "single_initial.npz", allow_pickle=False) as old,
            np.load(store.path / "single_initial.npz", allow_pickle=False) as new,
        ):
            fields = sorted(set(old.files) - {"metadata_utf8"})
            if set(old.files) != set(new.files) or any(
                not np.array_equal(old[k], new[k]) or old[k].strides != new[k].strides
                for k in fields
            ):
                raise ValueError("Initial analytic weights or storage order differ")
        store.write(
            "repair_replay.json",
            {
                "files_exact": list(replay_files),
                "tensor_values_and_strides_exact": fields,
                "updates_before_replay": 0,
                "first_artifact_sha256": file_sha(first_attempt / "single_initial.npz"),
            },
        )
    cases = {(c["group"], c["a"], c["b"], c["source_order"], c["pole"]): c for c in fit}
    label_ids = [
        host.tokenizer.encode(t, add_special_tokens=False)[0] for t in ("A", "B")
    ]
    optim = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0)
    start = host.meter.snapshot()
    for step, units in enumerate(schedule, 1):
        chosen = [cases[(*unit, pole)] for unit in units for pole in ("plus", "minus")]
        optim.zero_grad(set_to_none=True)
        losses = []
        for case in chosen:
            tasks = tasks_for(case, groups[case["group"]], (0, 1))
            supplied = {
                case["key"]: lexical(
                    model(states[case["source"]].slots.detach().clone())
                )
            }
            logits = batched_recipient_logits(
                host, adapter, *batch_inputs(host, tasks, supplied, "matched")
            )
            target = torch.tensor(
                [label_ids[t["gold"] == "B"] for t in tasks], device=host.device
            )
            loss = training_loss(logits, target, label_ids, "native_binary")
            (loss / len(chosen)).backward()
            losses.append(loss.item())
        mapped = model(torch.stack([states[c["source"]].slots for c in chosen])).float()
        target = teachers[[c["class_id"] for c in chosen]].float()
        rep = (mapped - target).square().mean() / stats["teacher_variance"]
        matched_std = (mapped.var(0, correction=0) + 1e-8).sqrt()
        target_std = (target.var(0, correction=0) + 1e-8).sqrt()
        var = (
            torch.relu(target_std - matched_std).square().mean()
            / stats["teacher_variance"]
        )
        (0.1 * rep + 0.01 * var).backward()
        if any(
            p.grad is not None for p in (*lexical.parameters(), *adapter.parameters())
        ):
            raise ValueError("Frozen interface received gradients")
        store.append(
            "training.jsonl",
            {
                "arm": arm,
                "step": step,
                "source_keys": [c["key"] for c in chosen],
                "native_losses": losses,
                "representation_loss": rep.item(),
                "variance_loss": var.item(),
                "gradient_norm": update(model, optim, host),
            },
        )
    end = host.meter.snapshot()
    if any(not torch.equal(fixed[k], v) for k, v in model.named_buffers()):
        raise ValueError("Fixed discrimination or teacher tensors changed")
    final = checked_save(
        host,
        lexical,
        adapter,
        model,
        states[first["source"]],
        sample,
        store,
        arm,
        "final",
        meta,
    )
    store.append(
        "training_compute.jsonl",
        {
            "arm": arm,
            "forwards": end["forwards"] - start["forwards"],
            "processed_tokens": end["processed_tokens"] - start["processed_tokens"],
        },
    )
    return initial, final


def score_panel(
    host,
    lexical,
    adapter,
    model,
    bank,
    groups,
    states,
    teachers,
    store,
    arm,
    endpoint,
    panel,
    routes,
):
    part, wordings = PANELS[panel]
    cases = bank[part]
    by_key = {c["key"]: c for c in cases}
    directions = torch.nn.functional.normalize(
        torch.randn(
            host.slots, host.width, generator=torch.Generator().manual_seed(4711)
        ),
        dim=-1,
    ).to(host.device)
    cache = {}
    for case in cases:
        tasks = tasks_for(case, groups[case["group"]], wordings)
        source = states[case["source"]].slots
        if model is not None and arm in ARMS:
            weights = model.weights(source)[0]
            store.append(
                "assignments.jsonl",
                {
                    "arm": arm,
                    "endpoint": endpoint,
                    "panel": panel,
                    "key": case["key"],
                    "gold_class": case["class_id"],
                    "predicted_class": int(weights.argmax()),
                    "true_weight": weights[case["class_id"]].item(),
                },
            )
        for route in routes:
            if route == "literal":
                supplied = None
            elif route == "raw_E":
                supplied = source
            elif route == "teacher":
                supplied = lexical(teachers[case["class_id"]])
            elif route == "rendered_teacher":
                supplied = lexical(model.renderer(teachers[case["class_id"]]))
            else:
                value = source
                if route == "opposed":
                    value = states[by_key[case["opposed_key"]]["source"]].slots
                elif route == "mean":
                    value = model.mean.reshape_as(source).to(source.dtype)
                elif route == "zero":
                    value = torch.zeros_like(source)
                elif route == "norm":
                    value = (directions * source.float().norm(dim=-1, keepdim=True)).to(
                        source.dtype
                    )
                elif route == "reverse":
                    value = source.flip(0)
                supplied = lexical(model(value))
            state_sha = (
                State(slots=supplied).sha()
                if supplied is not None
                else digest(case["source"])
            )
            # Explicit donor exchanges are forwarded. Only identical null and oracle inputs share a score.
            shared = route in ("mean", "zero", "teacher", "rendered_teacher")
            cache_key = (route, state_sha, tuple(t["question"] for t in tasks))
            if shared and cache_key in cache:
                scores = cache[cache_key]
            else:
                scores = model_scores(
                    host,
                    adapter,
                    tasks,
                    {case["key"]: supplied},
                    "literal_base" if route == "literal" else "matched",
                )
                if shared:
                    cache[cache_key] = scores
            for task, score in zip(tasks, scores, strict=True):
                store.append(
                    "decisions.jsonl",
                    {
                        "arm": arm,
                        "endpoint": endpoint,
                        "panel": panel,
                        "route": route,
                        **{
                            k: case[k]
                            for k in (
                                "key",
                                "unit",
                                "group",
                                "pole",
                                "source_order",
                                "style",
                            )
                        },
                        **{k: task[k] for k in ("wording", "label_order", "gold")},
                        "source_sha256": digest(case["source"]),
                        "question_sha256": digest(task["question"]),
                        "injected_sha256": state_sha,
                        **score,
                    },
                )


def run(args, store):
    seal = code_manifest(ROOT)
    if seal["git_status"] or not seal["git_commit"]:
        raise ValueError("Clean sealed source required")
    store.write("source.json", seal)
    data = json.loads(DATA.read_text())
    request = {
        "plan_sha256": file_sha(STUDY / "PLAN.md"),
        "fixture_sha256": file_sha(DATA),
        "scorer_sha256": file_sha(STUDY / "summarize.py"),
        "lexical_sha256": LEXICAL_SHA,
        "prior_final_sha256": file_sha(OLD / "final.npz"),
        "arms": list(ARMS),
        "updates_per_arm": 64,
        "seed": 47,
        "topics": [g["id"] for g in data["groups"]],
    }
    store.write("request.json", request)
    host, checkpoint = make_host(args)
    store.write("host.json", host.identity)
    store.write("native_parity.json", parity := host.native_parity())
    if (
        checkpoint["checkpoint_sha256"] != CHECKPOINT_SHA
        or host.identity != json.loads((OLD / "host.json").read_text())
        or not parity["exact"]
    ):
        raise ValueError("Host binding or native parity failed")
    lexical, _, lex_sha = load_lexical_interface(
        PRIOR / "lexical_bridge.npz", host.identity, device=host.device
    )
    if lex_sha != LEXICAL_SHA:
        raise ValueError("Lexical checkpoint differs")
    before, lex_before = parameter_sha(host.model), parameter_sha(lexical)
    banks = {arm: make_bank(data, arm) for arm in ARMS}
    groups = {g["id"]: g for g in data["groups"]}
    states = {}
    for arm, bank in banks.items():
        for part, cases in bank.items():
            for case in cases:
                text = case["source"]
                if text not in states:
                    states[text] = host.encode(text)
                store.append(
                    "sources.jsonl",
                    {
                        "arm": arm,
                        "part": part,
                        "key": case["key"],
                        "group": case["group"],
                        "class_id": case["class_id"],
                        "source_sha256": digest(text),
                        "state_sha256": states[text].sha(),
                    },
                )
    old_geometry(host, states, store)
    references = {(r["group"], r["pole"]): r for r in rows(OLD / "teachers.jsonl")}
    teacher_states = []
    for group in groups.values():
        for pole, words in zip(("plus", "minus"), group["options"], strict=True):
            state, layout = capture_lexical(host, words)
            if state.sha() != references[group["id"], pole]["state_sha256"]:
                raise ValueError("Canonical teacher changed")
            teacher_states.append(state.slots)
            store.append(
                "teachers.jsonl",
                {
                    "group": group["id"],
                    "pole": pole,
                    "state_sha256": state.sha(),
                    **layout,
                },
            )
    teachers = torch.stack(teacher_states)
    units = [
        (group, a, b, order)
        for group in groups
        for a in range(4)
        for b in range(4)
        for order in range(2)
    ]
    permutation = torch.randperm(
        len(units), generator=torch.Generator().manual_seed(47)
    ).tolist()
    schedule = [
        [units[i] for i in permutation[start : start + 4]] for start in range(0, 256, 4)
    ]
    store.write("schedule.json", schedule)
    adapter = RecipientAdapter(host, rank=16, last_layers=4, seed=47).requires_grad_(
        False
    )
    try:
        replay_previous(host, lexical, adapter, store)
        fitted = {}
        meta = {
            "base_host_sha256": digest(host.identity),
            "lexical_sha256": lex_sha,
            "source_sha256": seal["source_sha256"],
            "fixture_sha256": request["fixture_sha256"],
            "plan_sha256": request["plan_sha256"],
            "schedule_sha256": digest(schedule),
        }
        for arm in ARMS:
            fitted[arm] = train_arm(
                host,
                lexical,
                adapter,
                banks[arm],
                groups,
                states,
                teachers,
                schedule,
                store,
                arm,
                meta,
                args.first_attempt,
            )
        # Both fits are complete and serialized before any fresh readout is scored.
        previous, _, old_sha = load_source_mapper(
            OLD / "final.npz",
            host_identity=host.identity,
            lexical_sha256=lex_sha,
            device=host.device,
        )
        if old_sha != request["prior_final_sha256"]:
            raise ValueError("Section 6.12 control binding mismatch")
        with torch.no_grad():
            for panel in ("source", "joint", "syntax", "lexicon"):
                score_panel(
                    host,
                    lexical,
                    adapter,
                    None,
                    banks["many"],
                    groups,
                    states,
                    teachers,
                    store,
                    "reference",
                    "fixed",
                    panel,
                    ("literal", "raw_E", "teacher"),
                )
            score_panel(
                host,
                lexical,
                adapter,
                previous,
                banks["many"],
                groups,
                states,
                teachers,
                store,
                "archived",
                "fixed",
                "joint",
                ("matched", "opposed"),
            )
            for arm, (initial, final) in fitted.items():

                def score(module, name, panel, routes):
                    score_panel(
                        host,
                        lexical,
                        adapter,
                        module,
                        banks[arm],
                        groups,
                        states,
                        teachers,
                        store,
                        arm,
                        name,
                        panel,
                        routes,
                    )

                for panel in ("fit", "joint"):
                    score(initial, "initial", panel, ("matched",))
                for panel in PANELS:
                    if panel in ("joint", "syntax"):
                        routes = (
                            "matched",
                            "opposed",
                            "mean",
                            "zero",
                            "norm",
                            "reverse",
                            "rendered_teacher",
                        )
                    elif panel == "fit":
                        routes = ("matched", "opposed")
                    else:
                        routes = ("matched",)
                    score(final, "final", panel, routes)
    finally:
        adapter.close()
    after, lex_after = parameter_sha(host.model), parameter_sha(lexical)
    if before != after or lex_before != lex_after:
        raise ValueError("Frozen weights changed")
    return {
        "status": "COMPLETED_WITHIN_TOPIC_WORDING",
        "backbone_sha256_before": before,
        "backbone_sha256_after": after,
        "lexical_sha256_before": lex_before,
        "lexical_sha256_after": lex_after,
        "source_captures": len(states),
        "updates": 128,
        "compute": host.meter.snapshot(),
        "G_trained": False,
        "qualification_content_loaded": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--first-attempt",
        type=Path,
        help="Verify the archived stopped attempt before the first optimizer update",
    )
    parser.set_defaults(
        backend="hf",
        cpu_threads=1,
        device="cuda:0",
        dtype="bfloat16",
        slots=8,
        cut=None,
        writer=None,
        max_context=1024,
        max_new_tokens=None,
        until_eos=True,
        transport=None,
        encoder_bridge=None,
        reviewed_local_code=None,
        wall_seconds=2400,
        profile=ROOT / "configs/models/gemma2b.yaml",
    )
    args = parser.parse_args()
    store, start = RunStore(args.out), time.monotonic()
    try:
        result = run(args, store)
    except Exception as exc:
        store.write(
            "terminal.json",
            {
                "status": "FAILED_NO_RETRY",
                "type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        raise
    store.write("terminal.json", result | {"elapsed_seconds": time.monotonic() - start})
    print(json.dumps(result))


if __name__ == "__main__":
    main()
