"""Run two prospective development screens on the known eight-topic policy bank."""

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_continuity_recipient import batch_inputs, model_scores, training_loss
from fit_source_lexical import (
    LEXICAL_SHA,
    PRIOR,
    capture_lexical,
    replay_previous,
    rows,
)
from probe_slot_readout import update
from selffold.cli import ROOT, make_host
from selffold.polarity_readout import fit_axes, predicted_class, read_axes, source_cases
from selffold.recipient_learning import RecipientAdapter, batched_recipient_logits
from selffold.records import RunStore, State, code_manifest, digest, file_sha
from selffold.source_mapping import load_lexical_interface
from selffold.wording_transfer import load_policy_mixture

STUDY = ROOT / "provenance/teacher_polarity_20261103"
DATA = ROOT / "data/teacher_polarity_20261103.json"
OLD = ROOT / "provenance/within_topic_20261103/run"
OLD_MAP = "1a6ae43d04da0bf69cd109f4cc448fac0f0ac2e149fb34f32b23778bc47150eb"
spec = importlib.util.spec_from_file_location(
    "teacher_polarity_scores", STUDY / "summarize.py"
)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def make_tasks(groups, topic, pole, split, wording):
    group = groups[topic]
    stem = group["questions" if split == "fit" else "fresh_questions"][wording]
    key = str(2 * topic + pole)
    tasks = []
    for order, options in enumerate((group["options"], group["options"][::-1])):
        tasks.append(
            {
                "key": key,
                "source": "",
                "topic": topic,
                "group": group["id"],
                "pole": pole,
                "split": split,
                "wording": wording,
                "label_order": order,
                "gold": "A" if pole == order else "B",
                "question": f"{stem}\nA = {options[0]}. B = {options[1]}. Use the supplied description.",
            }
        )
    return tasks


def calibrate(host, adapter, lexical, archived, teachers, groups, store):
    renderer = deepcopy(archived.renderer).requires_grad_(True)
    optim = torch.optim.AdamW(renderer.parameters(), lr=0.0003, weight_decay=0.0)
    ids = [host.tokenizer.encode(t, add_special_tokens=False)[0] for t in ("A", "B")]
    generator = torch.Generator().manual_seed(47)
    schedule = []
    for _ in range(8):
        order = torch.randperm(32, generator=generator).tolist()
        schedule.extend(
            [[divmod(k, 4) for k in order[i : i + 4]] for i in range(0, 32, 4)]
        )
    store.write("schedule.json", schedule)

    def batch(module, t, q):
        tasks = [r for p in range(2) for r in make_tasks(groups, t, p, "fit", q)]
        supplied = {
            str(2 * t + p): lexical(module(teachers[2 * t + p])) for p in (0, 1)
        }
        logits = batched_recipient_logits(
            host, adapter, *batch_inputs(host, tasks, supplied, "matched")
        )
        targets = torch.tensor(
            [ids[r["gold"] == "B"] for r in tasks], device=host.device
        )
        return logits, targets

    for step, blocks in enumerate(schedule, 1):
        optim.zero_grad(set_to_none=True)
        losses = []
        for t, q in blocks:
            logits, target = batch(renderer, t, q)
            loss = training_loss(logits, target, ids, "native_full")
            (loss / len(blocks)).backward()
            losses.append(loss.item())
        store.append(
            "training.jsonl",
            {
                "step": step,
                "blocks": blocks,
                "losses": losses,
                "gradient_norm": update(renderer, optim, host),
            },
        )
    renderer.eval().requires_grad_(False)
    path = store.path / "renderer.npz"
    np.savez_compressed(
        path, **{k: v.detach().cpu().numpy() for k, v in renderer.state_dict().items()}
    )
    saved = deepcopy(archived.renderer)
    with np.load(path, allow_pickle=False) as f:
        saved.load_state_dict(
            {k: torch.from_numpy(f[k].copy(order="K")) for k in f.files}
        )
    with torch.no_grad():
        if not torch.equal(renderer(teachers), saved(teachers)) or not torch.equal(
            batch(saved, 0, 0)[0], batch(renderer, 0, 0)[0]
        ):
            raise ValueError("Calibrated renderer replay failed")
    store.write(
        "renderer.json",
        {
            "initial_parameter_sha256": parameter_sha(archived.renderer),
            "final_parameter_sha256": parameter_sha(saved),
            "artifact_sha256": file_sha(path),
            "updates": len(schedule),
            "parameters": sum(p.numel() for p in saved.parameters()),
            "rows_exact_all_teachers": True,
            "fitting_native_logits_exact": True,
        },
    )
    return saved


class NativePairs:
    """Cache only exactly identical injected tensors and displayed question pairs."""

    def __init__(self, host, adapter, groups, store):
        self.host, self.adapter, self.groups, self.store = host, adapter, groups, store
        self.cache = {}

    def score(self, topic, pole, split, wording, state):
        tasks = make_tasks(self.groups, topic, pole, split, wording)
        sha = State(slots=state).sha()
        question_hashes = [digest(t["question"]) for t in tasks]
        key = digest([sha, question_hashes])
        if key not in self.cache:
            output = model_scores(
                self.host, self.adapter, tasks, {tasks[0]["key"]: state}, "matched"
            )
            self.cache[key] = output
            self.store.append(
                "native_pairs.jsonl",
                {
                    "pair_sha256": key,
                    "injected_sha256": sha,
                    "question_sha256": question_hashes,
                    "outputs": output,
                },
            )
        return [
            {k: v for k, v in t.items() if k not in ("key", "source", "question")}
            | {"pair_sha256": key, "injected_sha256": sha, "question_sha256": h}
            | s
            for t, h, s in zip(tasks, question_hashes, self.cache[key], strict=True)
        ]


def run(args, store):
    seal = code_manifest(ROOT)
    if not seal["git_commit"] or seal["git_status"] != "":
        raise ValueError("Clean source commit required")
    fixture = json.loads(DATA.read_text())
    prior = ROOT / fixture["prior_fixture"]
    if file_sha(prior) != fixture["prior_sha256"]:
        raise ValueError("Previous action vocabulary changed")
    old = json.loads(prior.read_text())
    groups = old["groups"]
    for g in groups:
        g["fresh_questions"] = (
            fixture["generic_questions"] + fixture["specific_questions"][g["id"]]
        )
    store.write("source.json", seal)
    store.write(
        "request.json",
        {
            "plan_sha256": file_sha(STUDY / "PLAN.md"),
            "reducer_sha256": file_sha(STUDY / "summarize.py"),
            "fixture_sha256": file_sha(DATA),
            "prior_fixture_sha256": file_sha(prior),
            "prior_map_sha256": OLD_MAP,
            "lexical_sha256": LEXICAL_SHA,
            "topics": [g["id"] for g in groups],
            "seed": 47,
            "updates": 64,
        },
    )
    host, checkpoint = make_host(args)
    host.max_forwards, host.max_processed_tokens = 5000, 3_000_000
    store.write("host.json", host.identity)
    store.write("native_parity.json", parity := host.native_parity())
    if (
        checkpoint["checkpoint_sha256"] != CHECKPOINT_SHA
        or not parity["exact"]
        or host.identity != json.loads((OLD / "host.json").read_text())
    ):
        raise ValueError("Native host binding failed")
    lexical, _, lex_sha = load_lexical_interface(
        PRIOR / "lexical_bridge.npz", host.identity, device=host.device
    )
    archived, _, map_sha = load_policy_mixture(
        OLD / "many_final.npz", host.identity, LEXICAL_SHA, host.device
    )
    if (lex_sha, map_sha) != (LEXICAL_SHA, OLD_MAP):
        raise ValueError("Archived interface binding failed")
    before = [parameter_sha(m) for m in (host.model, lexical, archived)]
    expected = {
        (r["group"], r["pole"]): r["state_sha256"] for r in rows(OLD / "teachers.jsonl")
    }
    teachers = []
    for group in groups:
        for pole, words in zip(("plus", "minus"), group["options"], strict=True):
            state, _ = capture_lexical(host, words)
            if state.sha() != expected[group["id"], pole] or not torch.equal(
                state.slots.float(), archived.teachers[len(teachers)]
            ):
                raise ValueError("Canonical teacher replay failed")
            teachers.append(state.slots)
            store.append(
                "captures.jsonl",
                {"group": group["id"], "pole": pole, "teacher_sha256": state.sha()},
            )
    teachers = torch.stack(teachers)
    adapter = RecipientAdapter(host, rank=16, last_layers=4, seed=47).requires_grad_(
        False
    )
    try:
        replay_previous(host, lexical, adapter, store)
        renderer = calibrate(host, adapter, lexical, archived, teachers, groups, store)
        with torch.no_grad():
            pairs = NativePairs(host, adapter, groups, store)
            # Keep scalar renderer forwards identical for fit, fresh and conditional use.
            states_by_route = {
                "canonical": [lexical(z) for z in teachers],
                "archived": [lexical(archived.renderer(z)) for z in teachers],
                "calibrated": [lexical(renderer(z)) for z in teachers],
            }
            null = lexical(renderer(teachers.float().mean(0).to(host.dtype)))
            for split in ("fit", "fresh"):
                for topic in range(len(groups)):
                    for wording in range(4):
                        for p in (0, 1):
                            index = 2 * topic + p
                            for route in (
                                "canonical",
                                "archived",
                                "calibrated",
                                "mean",
                            ):
                                state = (
                                    null
                                    if route == "mean"
                                    else states_by_route[route][index]
                                )
                                for row in pairs.score(topic, p, split, wording, state):
                                    store.append(
                                        "teachers.jsonl", row | {"route": route}
                                    )
            _, teacher_pass = scorer.teacher_stats(rows(store.path / "teachers.jsonl"))
            store.write(
                "teacher_gate.json",
                {"passed": teacher_pass, "source_fit_started": False},
            )
            forms = {s: old["styles"][s] for s in old["fit_styles"]}
            fit = list(source_cases(groups, forms, fixture["familiar_pairs"]))
            fresh = list(
                source_cases(groups, fixture["fresh_frames"], fixture["familiar_pairs"])
            )
            checks = {
                r["source_sha256"]: r["state_sha256"]
                for r in rows(OLD / "sources.jsonl")
            }

            def capture(cases):
                vectors, hashes = [], []
                for c in cases:
                    state = host.encode(c["source"])
                    sha = state.sha()
                    prior_sha = checks.get(digest(c["source"]))
                    if prior_sha is not None and sha != prior_sha:
                        raise ValueError("Exposed E capture did not replay")
                    vectors.append(state.slots.cpu().float().numpy())
                    hashes.append(sha)
                return np.stack(vectors), hashes

            x, x_hashes = capture(fit)
            axes = fit_axes(x, fit, len(groups))
            path = store.path / "axes.npz"
            np.savez_compressed(path, **axes)
            with np.load(path, allow_pickle=False) as f:
                restored = {k: f[k].copy(order="K") for k in f.files}
            if any(read_axes(v, axes) != read_axes(v, restored) for v in x):
                raise ValueError("Polarity score replay failed")
            store.write(
                "axes.json",
                {
                    "artifact_sha256": file_sha(path),
                    "score_replay": len(x),
                    "fit_state_sha256": digest(x_hashes),
                    "shape": {k: list(v.shape) for k, v in axes.items()},
                },
            )
            test, test_hashes = capture(fresh)
            for split, cases, values, hashes in (
                ("fit", fit, x, x_hashes),
                ("fresh", fresh, test, test_hashes),
            ):
                for c, v, sha in zip(cases, values, hashes, strict=True):
                    scores = read_axes(v, restored)
                    flat = archived.weights(
                        torch.from_numpy(v).to(host.device, host.dtype)
                    )[0]
                    ids = {
                        m: predicted_class(scores, m)
                        for m in ("factor", "direct", "mention")
                    }
                    store.append(
                        "sources.jsonl",
                        {k: v for k, v in c.items() if k != "source"}
                        | ids
                        | {
                            "split": split,
                            "source_sha256": digest(c["source"]),
                            "state_sha256": sha,
                            "prior_replay": digest(c["source"]) in checks,
                            "scores": scores,
                            "oracle_factor": predicted_class(
                                scores, oracle_topic=c["topic"]
                            ),
                            "flat": int(flat.argmax())
                            if int((flat == flat.max()).sum()) == 1
                            else None,
                        },
                    )
            source_rows = rows(store.path / "sources.jsonl")
            _, source_pass = scorer.source_stats(source_rows)
            store.write(
                "source_gate.json", {"passed": source_pass, "joint_started": False}
            )
            if teacher_pass and source_pass:
                selected = {
                    (r["topic"], r["frame"], r["pair"], r["pole"], r["order"]): r[
                        "factor"
                    ]
                    for r in source_rows
                    if r["split"] == "fresh"
                }
                null_classes = {
                    name: predicted_class(read_axes(v, restored))
                    for name, v in (
                        ("mean_E", restored["mean"]),
                        ("zero_E", np.zeros_like(restored["mean"])),
                    )
                }
                for c in fresh:
                    t, f, w, p, o = (
                        c[k] for k in ("topic", "frame", "pair", "pole", "order")
                    )
                    routes = {
                        "matched": selected[t, f, w, p, o],
                        "opposed_role_order": selected[t, f, w, 1 - p, o],
                        "opposed_action_order": selected[t, f, w, 1 - p, 1 - o],
                        "oracle": 2 * t + p,
                    } | null_classes
                    for route, cls in routes.items():
                        state = (
                            null if cls is None else states_by_route["calibrated"][cls]
                        )
                        for q in range(4):
                            for rec in pairs.score(t, p, "fresh", q, state):
                                store.append(
                                    "joint.jsonl",
                                    rec
                                    | {
                                        "key": c["key"],
                                        "frame": f,
                                        "pair": w,
                                        "order": o,
                                        "route": route,
                                        "selected_class": cls,
                                    },
                                )
    finally:
        adapter.close()
    after = [parameter_sha(m) for m in (host.model, lexical, archived)]
    if before != after:
        raise ValueError("Frozen weights changed")
    return {
        "status": "COMPLETED_TEACHER_POLARITY",
        "frozen_parameter_sha256_before": before,
        "frozen_parameter_sha256_after": after,
        "teacher_gate": teacher_pass,
        "source_gate": source_pass,
        "joint_executed": teacher_pass and source_pass,
        "compute": host.meter.snapshot(),
        "G_trained": False,
        "qualification_content_loaded": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
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
        wall_seconds=1200,
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
                "status": "TECHNICAL_STOP",
                "error": str(exc),
                "type": type(exc).__name__,
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        raise
    store.write("terminal.json", result | {"elapsed_seconds": time.monotonic() - start})
    print(json.dumps(result))
