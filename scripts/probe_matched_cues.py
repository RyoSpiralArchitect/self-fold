"""Change one question heading or swap two preference fields with both interfaces frozen."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_continuity_recipient import model_scores
from fit_source_lexical import LEXICAL_SHA, PRIOR, capture_lexical, rows
from fit_teacher_and_polarity import NativePairs, make_tasks
from selffold import prompts
from selffold.cli import ROOT, make_host
from selffold.codec_learning import EncoderBridge
from selffold.matched_cues import (
    HEADERS,
    header_geometry,
    preference_records,
    question_groups,
    source_geometry,
)
from selffold.polarity_readout import predicted_class, read_axes, source_cases
from selffold.recipient_learning import RecipientAdapter
from selffold.records import RunStore, code_manifest, digest, file_sha
from selffold.source_mapping import load_lexical_interface

STUDY = ROOT / "provenance/matched_cues_20261103"
DATA = ROOT / "data/matched_cues_20261103.json"
ACTIONS = ROOT / "data/within_topic_wording_20261103.json"
PREV = ROOT / "provenance/teacher_polarity_20261103/run"
RENDERER_SHA = "8d87b849730159149be4584ff2962aca848ff55817d48cfa47edb7f1b310f985"
AXES_SHA = "37981ffed73bf31c343995a5a6947251f59953f0d10cd6743be2d96651ebbde9"


def bind_readers(host, groups):
    lexical, _, sha = load_lexical_interface(
        PRIOR / "lexical_bridge.npz", host.identity, device=host.device
    )
    if (sha, file_sha(PREV / "renderer.npz"), file_sha(PREV / "axes.npz")) != (
        LEXICAL_SHA,
        RENDERER_SHA,
        AXES_SHA,
    ):
        raise ValueError("Frozen interface differs from the published endpoint")
    renderer = EncoderBridge(host.width, 4).to(host.device).eval().requires_grad_(False)
    with np.load(PREV / "renderer.npz", allow_pickle=False) as f:
        renderer.load_state_dict(
            {k: torch.from_numpy(f[k].copy(order="K")) for k in f.files}
        )
    with np.load(PREV / "axes.npz", allow_pickle=False) as f:
        axes = {k: f[k].copy(order="K") for k in f.files}
    refs = {
        (r["group"], r["pole"]): r["teacher_sha256"]
        for r in rows(PREV / "captures.jsonl")
    }
    states = []
    for g in groups:
        for pole, words in zip(("plus", "minus"), g["options"], strict=True):
            z, _ = capture_lexical(host, words)
            if z.sha() != refs[g["id"], pole]:
                raise ValueError("Teacher capture replay failed")
            states.append(z.slots)
    null = torch.stack(states).float().mean(0).to(host.dtype)
    return (
        lexical,
        renderer,
        axes,
        [lexical(renderer(z)) for z in states],
        lexical(renderer(null)),
    )


def replay(host, adapter, teachers, axes, old_groups):
    f = json.loads((ROOT / "data/teacher_polarity_20261103.json").read_text())
    groups = [
        g
        | {"fresh_questions": f["generic_questions"] + f["specific_questions"][g["id"]]}
        for g in old_groups
    ]
    old_teacher = {
        (r["topic"], r["pole"], r["wording"], r["label_order"]): r
        for r in rows(PREV / "teachers.jsonl")
        if (r["split"], r["route"]) == ("fresh", "calibrated")
    }
    checked = 0
    for t in range(len(groups)):
        for p in (0, 1):
            for q in range(4):
                tasks = make_tasks(groups, t, p, "fresh", q)
                results = model_scores(
                    host,
                    adapter,
                    tasks,
                    {tasks[0]["key"]: teachers[2 * t + p]},
                    "matched",
                )
                for o, s in enumerate(results):
                    ref = old_teacher[t, p, q, o]
                    if (
                        any(s[k] != ref[k] for k in s)
                        or digest(tasks[o]["question"]) != ref["question_sha256"]
                    ):
                        raise ValueError("Previous native pair failed exact replay")
                    checked += 1
    previous = {
        r["key"]: r for r in rows(PREV / "sources.jsonl") if r["split"] == "fresh"
    }
    seen = set()
    for c in source_cases(groups, f["fresh_frames"], f["familiar_pairs"]):
        z = host.encode(c["source"])
        scores = read_axes(z.slots.cpu().float().numpy(), axes)
        ref = previous[c["key"]]
        if (
            z.sha() != ref["state_sha256"]
            or scores["topic"] != ref["scores"]["topic"]
            or any(
                predicted_class(scores, m) != ref[m]
                for m in ("factor", "direct", "mention")
            )
        ):
            raise ValueError("Previous source state or prediction failed replay")
        seen.add(ref["source_sha256"])
    if (checked, len(seen)) != (128, 256):
        raise ValueError("Incomplete frozen replay")
    return {
        "canonical_teachers": len(teachers),
        "native_scores_exact": checked,
        "source_hashes_and_decisions_exact": len(seen),
    }


def verify_layouts(host, groups, cases, store):
    for t in range(len(groups)):
        for q in range(4):
            for o in (0, 1):
                tasks = [
                    make_tasks(groups, t, 0, "fresh", q * 4 + i)[o] for i in range(4)
                ]
                rendered = [
                    host.render(prompts.task_prompt(task["question"])) for task in tasks
                ]
                if any(pos != rendered[0][1] for ids, pos in rendered):
                    raise ValueError("Question edit moved the teacher input slots")
                checked = header_geometry([ids for ids, pos in rendered])
                store.append(
                    "question_layout.jsonl",
                    checked
                    | {
                        "topic": t,
                        "stem": q,
                        "label_order": o,
                        "layout_sha256": [digest(ids) for ids, pos in rendered],
                        "question_sha256": [digest(task["question"]) for task in tasks],
                    },
                )
    by_key = {
        (c["topic"], c["family"], c["pair"], c["pole"], c["order"], c["position"]): c
        for c in cases
    }
    if len(by_key) != len(cases) or len(cases) != 512:
        raise ValueError("Incomplete source design")
    for (t, f, w, p, o, pos), c in by_key.items():
        if pos == "before":
            other = by_key[t, f, w, p, o, "after"]
            layout = source_geometry(host, c["source"], other["source"])
            store.append(
                "source_layout.jsonl",
                layout
                | {
                    "before": c["key"],
                    "after": other["key"],
                    "before_sha256": digest(c["source"]),
                    "after_sha256": digest(other["source"]),
                },
            )


def evaluate(host, adapter, teachers, null, axes, groups, cases, store):
    pair_scores = NativePairs(host, adapter, groups, store)
    for t in range(len(groups)):
        for stem in range(4):
            for i, header in enumerate(HEADERS):
                for p in (0, 1):
                    for route, z in (
                        ("calibrated", teachers[2 * t + p]),
                        ("mean", null),
                    ):
                        for row in pair_scores.score(t, p, "fresh", 4 * stem + i, z):
                            store.append(
                                "teachers.jsonl",
                                row | {"header": header, "stem": stem, "route": route},
                            )
    for c in cases:
        z = host.encode(c["source"])
        s = read_axes(z.slots.cpu().float().numpy(), axes)
        info = {k: v for k, v in c.items() if k != "source"}
        info["source_sha256"] = digest(c["source"])
        store.append(
            "sources.jsonl",
            info
            | {
                "state_sha256": z.sha(),
                "scores": s,
                "oracle_factor": predicted_class(s, oracle_topic=c["topic"]),
            }
            | {m: predicted_class(s, m) for m in ("factor", "direct", "mention")},
        )
        tasks = [
            t | {"source": c["source"]}
            for t in make_tasks(groups, c["topic"], c["pole"], "fresh", 0)
        ]
        values = model_scores(
            host, adapter, tasks, {tasks[0]["key"]: None}, "literal_base"
        )
        for t, out in zip(tasks, values, strict=True):
            store.append(
                "literal.jsonl",
                info
                | {
                    "label_order": t["label_order"],
                    "gold": t["gold"],
                    "question_sha256": digest(t["question"]),
                }
                | out,
            )


@torch.no_grad()
def run(args, store):
    seal = code_manifest(ROOT)
    if not seal["git_commit"] or seal["git_status"] != "":
        raise ValueError("A clean sealed commit is required")
    fixture = json.loads(DATA.read_text())
    if file_sha(ACTIONS) != fixture["action_fixture_sha256"]:
        raise ValueError("Action vocabulary changed")
    old_groups = json.loads(ACTIONS.read_text())["groups"]
    groups = question_groups(old_groups, fixture)
    cases = list(preference_records(old_groups, fixture))
    store.write("source.json", seal)
    store.write(
        "request.json",
        {
            "plan_sha256": file_sha(STUDY / "PLAN.md"),
            "reducer_sha256": file_sha(STUDY / "summarize.py"),
            "fixture_sha256": file_sha(DATA),
            "prior_sources_sha256": file_sha(PREV / "sources.jsonl"),
            "prior_teacher_scores_sha256": file_sha(PREV / "teachers.jsonl"),
            "renderer_sha256": RENDERER_SHA,
            "axes_sha256": AXES_SHA,
            "lexical_sha256": LEXICAL_SHA,
            "updates": 0,
        },
    )
    host, checkpoint = make_host(args)
    host.max_forwards, host.max_processed_tokens = 4000, 4_000_000
    store.write("host.json", host.identity)
    store.write("native_parity.json", parity := host.native_parity())
    if (
        not parity["exact"]
        or checkpoint["checkpoint_sha256"] != CHECKPOINT_SHA
        or host.identity != json.loads((PREV / "host.json").read_text())
    ):
        raise ValueError("Model or native parity mismatch")
    lexical, renderer, axes, teachers, null = bind_readers(host, old_groups)
    modules = (host.model, lexical, renderer)
    before = [parameter_sha(m) for m in modules]
    adapter = RecipientAdapter(host, rank=16, last_layers=4, seed=47).requires_grad_(
        False
    )
    try:
        store.write("replay.json", replay(host, adapter, teachers, axes, old_groups))
        verify_layouts(host, groups, cases, store)
        evaluate(host, adapter, teachers, null, axes, groups, cases, store)
    finally:
        adapter.close()
    after = [parameter_sha(m) for m in modules]
    if before != after or file_sha(PREV / "axes.npz") != AXES_SHA:
        raise ValueError("Frozen endpoint changed")
    return {
        "status": "COMPLETED_MATCHED_CUES",
        "frozen_parameter_sha256_before": before,
        "frozen_parameter_sha256_after": after,
        "compute": host.meter.snapshot(),
        "updates": 0,
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
