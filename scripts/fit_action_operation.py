"""Fit action identity while freezing verdict order, and test native question/conditional interfaces."""

import argparse
from pathlib import Path
import json, time
import numpy as np
import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_role_exchange import capture, validate_vocab
from fit_source_lexical import rows
from probe_matched_cues import bind_readers
from teacher_operation_panel import evaluate as evaluate_questions, question_groups
from conditional_teacher_audit import evaluate as evaluate_conditional
from selffold.action_swap import action_pairs, fit_action_heads, score_action
from selffold.cli import ROOT, make_host
from selffold.role_exchange import crossed_records, load_weights, score_one
from selffold.records import RunStore, code_manifest, digest, file_sha
from selffold.recipient_learning import RecipientAdapter

STUDY = ROOT / "provenance/action_operation_20261103"
DATA = ROOT / "data/action_swap_20261103.json"
ACTIONS = ROOT / "data/within_topic_wording_20261103.json"
CARDS = ROOT / "data/matched_cues_20261103.json"
FROZEN = ROOT / "provenance/role_exchange_20261103/run/weighted.npz"
SHA = "b30e42bb764c8eb82a181e08640741c820993a5f3aecde6acfd931f3c7cc2f8d"


def pairs_record(cases, layouts, store, split):
    keys, indices = action_pairs(cases)
    for key, (a, b) in zip(keys, indices, strict=True):
        if layouts[a] != layouts[b]:
            raise ValueError("Action swap changed token geometry")
        store.append(
            "action_pairs.jsonl",
            {
                "split": split,
                "stratum": list(key),
                "plus_first": cases[a]["key"],
                "minus_first": cases[b]["key"],
            }
            | layouts[a],
        )


def predict(x, record, split, maps, frozen, store):
    store.append(
        "source_scores.jsonl",
        record
        | {"split": split, "frozen": score_one(x, frozen)}
        | {name: score_action(x, m) for name, m in maps.items()},
    )


@torch.no_grad()
def run(args, store):
    seal = code_manifest(ROOT)
    if not seal["git_commit"] or seal["git_status"] != "":
        raise ValueError("Clean sealed source required")
    store.write("source.json", seal)
    settings = json.loads(CARDS.read_text())
    groups = json.loads(ACTIONS.read_text())["groups"]
    fresh = json.loads(DATA.read_text())["fresh_actions"]
    validate_vocab(groups, fresh)
    exposed = json.loads((ROOT / "data/role_exchange_20261103.json").read_text())[
        "fresh_actions"
    ]
    if not {
        s for g in fresh.values() for p in ("plus", "minus") for s in g[p]
    }.isdisjoint(s for g in exposed.values() for p in ("plus", "minus") for s in g[p]):
        raise ValueError("Fresh wording reused earlier evaluation")
    if file_sha(FROZEN) != SHA:
        raise ValueError("Frozen role and router mismatch")
    request = {
        "plan_sha256": file_sha(STUDY / "PLAN.md"),
        "reducer_sha256": file_sha(STUDY / "summarize.py"),
        "fresh_fixture_sha256": file_sha(DATA),
        "action_fixture_sha256": file_sha(ACTIONS),
        "card_fixture_sha256": file_sha(CARDS),
        "question_fixture_sha256": file_sha(
            ROOT / "data/question_operation_20261103.json"
        ),
        "conditional_fixture_sha256": file_sha(
            ROOT / "data/conditional_audit_20261103.json"
        ),
        "frozen_sha256": SHA,
        "backbone_updates": 0,
        "candidate_order": ["swap", "invariant"],
        "contrast_weight": 4,
        "covariance_coefficient": 0.5,
    }
    store.write("request.json", request)
    host, checkpoint = make_host(args)
    host.max_forwards = 15000
    host.max_processed_tokens = 6_000_000
    store.write("host.json", host.identity)
    store.write("native_parity.json", parity := host.native_parity())
    if (
        checkpoint["checkpoint_sha256"] != CHECKPOINT_SHA
        or not parity["exact"]
        or host.identity != json.loads((FROZEN.parent / "host.json").read_text())
    ):
        raise ValueError("Model binding failed")
    frozen = load_weights(FROZEN)
    backbone_before = parameter_sha(host.model)
    old = {
        r["source_sha256"]: r
        for r in rows(FROZEN.parent / "scores.jsonl")
        if r["split"] == "fit"
    }
    cases = list(crossed_records(groups, settings))
    n = len(cases)
    states = np.empty((n, host.slots, host.width), dtype=np.float32)
    records = []
    layouts = []
    if n != 4608:
        raise ValueError("Fitting denominator changed")
    for i, c in enumerate(cases):
        x, r, g = capture(host, c)
        states[i] = x
        records.append(r)
        layouts.append(g)
        ref = old[r["source_sha256"]]
        if (
            r["state_sha256"] != ref["state_sha256"]
            or score_one(x, frozen) != ref["weighted"]
        ):
            raise ValueError("Frozen fitting states or readout scores differ")
    pairs_record(cases, layouts, store, "fit")
    store.write(
        "replay.json",
        {"fit_states": n, "frozen_scores_exact": n, "fresh_sources_encoded": 0},
    )
    maps = fit_action_heads(states, cases)
    for name, m in maps.items():
        path = store.path / (name + ".npz")
        np.savez_compressed(path, **m)
        loaded = load_weights(path)
        if any(score_action(x, m) != score_action(x, loaded) for x in states):
            raise ValueError("Action artifact replay failed")
        maps[name] = loaded
        store.append(
            "fits.jsonl",
            {
                "arm": name,
                "artifact": path.name,
                "sha256": file_sha(path),
                "fit_singleton_replays": n,
                "fit_capture_sha256": digest(
                    [(r["source_sha256"], r["state_sha256"]) for r in records]
                ),
                "shapes": {k: list(v.shape) for k, v in m.items()},
            },
        )
    for x, r in zip(states, records, strict=True):
        predict(x, r, "fit", maps, frozen, store)
    del states
    store.write(
        "fit_closed.json",
        {
            "fresh_sources_encoded": 0,
            "artifact_sha256": {k: file_sha(store.path / (k + ".npz")) for k in maps},
        },
    )
    tests = list(crossed_records(groups, settings, fresh))
    layouts = []
    if len(tests) != 2048:
        raise ValueError("Fresh denominator changed")
    for c in tests:
        x, r, g = capture(host, c)
        layouts.append(g)
        predict(x, r, "fresh", maps, frozen, store)
    pairs_record(tests, layouts, store, "fresh")
    lexical, renderer, _, teachers, null = bind_readers(host, groups)
    fixed_before = [parameter_sha(m) for m in (lexical, renderer)]
    adapter = RecipientAdapter(host, rank=16, last_layers=4, seed=47).requires_grad_(
        False
    )
    try:
        query = json.loads((ROOT / "data/question_operation_20261103.json").read_text())
        evaluate_questions(
            host, adapter, teachers, null, question_groups(groups, query), store
        )
        audit = evaluate_conditional(host, adapter, groups, store)
    finally:
        adapter.close()
    after = parameter_sha(host.model)
    fixed_after = [parameter_sha(m) for m in (lexical, renderer)]
    if (
        after != backbone_before
        or fixed_before != fixed_after
        or file_sha(FROZEN) != SHA
    ):
        raise ValueError("Frozen interface changed")
    return {
        "status": "COMPLETED_ACTION_OPERATION",
        "backbone_before": backbone_before,
        "backbone_after": after,
        "fixed_before": fixed_before,
        "fixed_after": fixed_after,
        "fit_sources": n,
        "fresh_sources": len(tests),
        "audit": audit,
        "compute": host.meter.snapshot(),
        "backbone_updates": 0,
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
        wall_seconds=1800,
        profile=ROOT / "configs/models/gemma2b.yaml",
    )
    a = p.parse_args()
    store, start = RunStore(a.out), time.monotonic()
    try:
        r = run(a, store)
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
    store.write("terminal.json", r | {"elapsed_seconds": time.monotonic() - start})
    print(json.dumps(r))
