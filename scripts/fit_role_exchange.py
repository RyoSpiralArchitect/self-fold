"""Fit a paired singleton readout on role exchanges and test wholly held-out action phrases."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_continuity_recipient import model_scores
from fit_source_lexical import rows
from fit_teacher_and_polarity import make_tasks
from probe_matched_cues import ACTIONS, AXES_SHA, DATA as LAST_FIXTURE, PREV
from selffold import prompts
from selffold.cli import ROOT, make_host
from selffold.matched_cues import question_groups
from selffold.polarity_readout import read_axes, predicted_class
from selffold.recipient_learning import RecipientAdapter
from selffold.records import RunStore, code_manifest, digest, file_sha
from selffold.role_exchange import (
    crossed_records,
    role_pairs,
    fit_exchange,
    score_one,
    load_weights,
)

STUDY = ROOT / "provenance/role_exchange_20261103"
DATA = ROOT / "data/role_exchange_20261103.json"
LAST = ROOT / "provenance/matched_cues_20261103/run"


def validate_vocab(groups, fresh):
    if set(fresh) != {g["id"] for g in groups}:
        raise ValueError("Fresh topics differ from the registered bank")
    old_words = {w for g in groups for p in ("plus", "minus") for w in g[p]}
    new_words = [
        w.casefold().strip()
        for g in fresh.values()
        for p in ("plus", "minus")
        for w in g[p]
    ]
    if (
        len(set(new_words)) != len(new_words)
        or len(new_words) != 64
        or not set(new_words).isdisjoint(w.casefold().strip() for w in old_words)
        or any(len(g[p]) != 4 for g in fresh.values() for p in ("plus", "minus"))
    ):
        raise ValueError("Action-expression split is incomplete or overlaps fitting")


def capture(host, case):
    ids, pos = host.render(prompts.encode_prompt(case["source"]))
    state = host.encode(case["source"])
    layout = {
        "token_bag_sha256": digest(sorted(ids)),
        "num_tokens": len(ids),
        "slots": pos[prompts.OUTPUT_MARKER],
    }
    record = {k: v for k, v in case.items() if k != "source"} | {
        "source_sha256": digest(case["source"]),
        "state_sha256": state.sha(),
    }
    return state.slots.cpu().float().numpy(), record, layout


def record_pairs(cases, keys, ids, layouts, store, split, signs=None):
    for i, (k, (left, right)) in enumerate(zip(keys, ids, strict=True)):
        if layouts[left] != layouts[right]:
            raise ValueError(
                "Role exchange changed token multiset or workspace positions"
            )
        row = {
            "split": split,
            "topic": k[0],
            "family": k[1],
            "position": k[2],
            "pair": k[3],
            "first_action": k[4],
            "accept_first": cases[left]["key"],
            "reject_first": cases[right]["key"],
        } | layouts[left]
        if signs is not None:
            row["shuffled_orientation"] = int(signs[i])
        store.append("pairs.jsonl", row)


def checked_artifacts(maps, states, res, store):
    restored = {}
    for name, weights in maps.items():
        path = store.path / f"{name}.npz"
        np.savez_compressed(path, **weights)
        reopened = load_weights(path)
        if any(score_one(s, weights) != score_one(s, reopened) for s in states):
            raise ValueError(
                "Serialized singleton readout failed exact fitting-bank replay"
            )
        store.append(
            "fits.jsonl",
            {
                "arm": name,
                "contrast_weight": 4 if name == "weighted" else 1,
                "artifact": path.name,
                "sha256": file_sha(path),
                "fit_singleton_replays": len(states),
                "fit_capture_sha256": digest(
                    [(r["source_sha256"], r["state_sha256"]) for r in res]
                ),
                "shapes": {k: list(v.shape) for k, v in weights.items()},
            },
        )
        restored[name] = reopened
    return restored


def old_call(x, axes):
    score = read_axes(x, axes)
    return score["topic"], predicted_class(score), predicted_class(score, "direct")


def write_prediction(x, record, split, maps, store):
    store.append(
        "scores.jsonl",
        record | {"split": split} | {name: score_one(x, m) for name, m in maps.items()},
    )


@torch.no_grad()
def run(args, store):
    seal = code_manifest(ROOT)
    if not seal["git_commit"] or seal["git_status"] != "":
        raise ValueError("Clean sealed source required")
    groups = json.loads(ACTIONS.read_text())["groups"]
    settings = json.loads(LAST_FIXTURE.read_text())
    fresh = json.loads(DATA.read_text())["fresh_actions"]
    validate_vocab(groups, fresh)
    if (
        file_sha(ACTIONS) != settings["action_fixture_sha256"]
        or file_sha(PREV / "axes.npz") != AXES_SHA
    ):
        raise ValueError("Old action bank or router changed")
    store.write("source.json", seal)
    store.write(
        "request.json",
        {
            "plan_sha256": file_sha(STUDY / "PLAN.md"),
            "reducer_sha256": file_sha(STUDY / "summarize.py"),
            "fresh_fixture_sha256": file_sha(DATA),
            "fitting_fixture_sha256": file_sha(ACTIONS),
            "card_fixture_sha256": file_sha(LAST_FIXTURE),
            "router_sha256": AXES_SHA,
            "prior_sources_sha256": file_sha(LAST / "sources.jsonl"),
            "null_seed": 47,
            "contrast_weights": [4, 1],
            "optimizer_steps": 0,
            "ridge_relative": 0.01,
        },
    )
    host, checkpoint = make_host(args)
    host.max_forwards, host.max_processed_tokens = 12000, 4_000_000
    store.write("host.json", host.identity)
    store.write("native_parity.json", parity := host.native_parity())
    if (
        not parity["exact"]
        or checkpoint["checkpoint_sha256"] != CHECKPOINT_SHA
        or host.identity != json.loads((LAST / "host.json").read_text())
    ):
        raise ValueError("Pinned model or native parity differs")
    before = parameter_sha(host.model)
    router = load_weights(PREV / "axes.npz")
    expected = {r["source_sha256"]: r for r in rows(LAST / "sources.jsonl")}
    cases = list(crossed_records(groups, settings))
    if len(cases) != 4608:
        raise ValueError("Unexpected fitting size")
    keys, ids = role_pairs(cases)
    states = np.empty((len(cases), host.slots, host.width), dtype=np.float32)
    fit_records, layouts, seen = [], [], set()
    for i, c in enumerate(cases):
        x, r, layout = capture(host, c)
        states[i] = x
        topic, old, direct = old_call(x, router)
        ref = expected.get(r["source_sha256"])
        if ref is not None:
            if (r["state_sha256"], topic, old, direct) != (
                ref["state_sha256"],
                ref["scores"]["topic"],
                ref["factor"],
                ref["direct"],
            ):
                raise ValueError("Archived E or reader failed replay")
            seen.add(r["source_sha256"])
        fit_records.append(r | {"old": old})
        layouts.append(layout)
    if len(seen) != 512 or seen != set(expected):
        raise ValueError("Incomplete Section 6.15 replay")
    if any(layouts[a] != layouts[b] for a, b in ids):
        raise ValueError("Invalid fitting exchange geometry")
    store.write(
        "replay.json",
        {
            "prior_sources_and_state_hashes": len(seen),
            "prior_reader_decisions_exact": len(seen),
            "fresh_states_encoded": 0,
            "fitting_states": len(cases),
        },
    )
    maps, signs = fit_exchange(states, keys, ids, router)
    record_pairs(cases, keys, ids, layouts, store, "fit", signs)
    maps = checked_artifacts(maps, states, fit_records, store)
    for x, r in zip(states, fit_records, strict=True):
        write_prediction(x, r, "fit", maps, store)
    del states
    store.write(
        "fit_closed.json",
        {
            "fresh_sources_encoded": 0,
            "fitting_opposed_pairs": len(ids),
            "artifact_sha256": {n: file_sha(store.path / f"{n}.npz") for n in maps},
        },
    )
    tests = list(crossed_records(groups, settings, fresh))
    if len(tests) != 2048:
        raise ValueError("Unexpected fresh size")
    display = question_groups(groups, settings)
    new_layouts = []
    adapter = RecipientAdapter(host, rank=16, last_layers=4, seed=47).requires_grad_(
        False
    )
    try:
        for c in tests:
            x, r, layout = capture(host, c)
            new_layouts.append(layout)
            _, old, _ = old_call(x, router)
            write_prediction(x, r | {"old": old}, "fresh", maps, store)
            tasks = [
                t | {"source": c["source"]}
                for t in make_tasks(display, c["topic"], c["pole"], "fresh", 0)
            ]
            outcomes = model_scores(
                host, adapter, tasks, {tasks[0]["key"]: None}, "literal_base"
            )
            for t, o in zip(tasks, outcomes, strict=True):
                store.append(
                    "literal.jsonl",
                    {
                        "key": r["key"],
                        "frame": r["frame"],
                        "source_sha256": r["source_sha256"],
                        "question_sha256": digest(t["question"]),
                        "label_order": t["label_order"],
                        "gold": t["gold"],
                    }
                    | o,
                )
    finally:
        adapter.close()
    new_keys, new_ids = role_pairs(tests)
    record_pairs(tests, new_keys, new_ids, new_layouts, store, "fresh")
    after = parameter_sha(host.model)
    if before != after or file_sha(PREV / "axes.npz") != AXES_SHA:
        raise ValueError("Frozen encoding or routing changed")
    return {
        "status": "COMPLETED_ROLE_EXCHANGE_READOUT",
        "backbone_before": before,
        "backbone_after": after,
        "compute": host.meter.snapshot(),
        "fit_sources": len(cases),
        "fresh_sources": len(tests),
        "optimizer_steps": 0,
        "G_trained": False,
        "qualification_content_loaded": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(
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
    args = p.parse_args()
    store, start = RunStore(args.out), time.monotonic()
    try:
        result = run(args, store)
    except Exception as e:
        store.write(
            "terminal.json",
            {
                "status": "TECHNICAL_STOP",
                "error": str(e),
                "type": type(e).__name__,
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        raise
    store.write("terminal.json", result | {"elapsed_seconds": time.monotonic() - start})
    print(json.dumps(result))
