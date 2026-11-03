"""Reconstruct both v1 panels in a fresh native process without training."""

import gzip
import json
from pathlib import Path

import numpy as np
import torch
from fit_codec import CHECKPOINT_SHA
from fit_conditional_curriculum import parser
from fit_continuity_recipient import model_scores
from fit_preprint_reader import CONFIG, FOCUS, questions
from fit_stage_grammar import ACTIONS, BASE, DESIGN, PREVIOUS, read
from fit_stage_transitions import bank, records
from fit_three_role_capacity import score_tables
from role_memory_native import PARENT, READER, READER_SHA
from selffold.cli import ROOT, make_host
from selffold.recipient_learning import load_recipient
from selffold.records import RunStore, digest, file_sha
from selffold.stage_grammar import fixture
from selffold.state_only import ClauseReader
from selffold.three_role_capacity import ASSIGNMENTS, evaluate


@torch.no_grad()
def run(args):
    store = RunStore(args.out)
    host, ident = make_host(args)
    if (
        ident["checkpoint_sha256"] != CHECKPOINT_SHA
        or not host.native_parity()["exact"]
    ):
        raise ValueError("Different native interface")
    capacity, raw = args.capacity, args.raw
    cfg, base, old, design, actions = map(
        read, (CONFIG, BASE, PREVIOUS, DESIGN, ACTIONS)
    )
    menus = [g["options"] for g in actions["groups"]]
    compiler = ClauseReader(2 * host.width, 16).requires_grad_(False)
    with np.load(PARENT / "compiler.npz", allow_pickle=False) as f:
        compiler.load_state_dict({k: torch.from_numpy(f[k]) for k in f.files})
    native, _, sha = load_recipient(READER, host)
    if sha != READER_SHA:
        raise ValueError("Unexpected structured reader")
    clause = read(ROOT / "data/state_only_20261104.json")["styles"]["when"][1]
    n = 0
    try:
        for split, topics, names in (
            ("fit", [0, 2, 4, 6], [cfg["capacity_fit_flags"]]),
            ("test", [1, 3, 5, 7], cfg["capacity_test_flags"]),
        ):
            rewards = score_tables(
                host, compiler, native, menus, clause, topics, names, store, split
            )
            filename = f"{split}_native.npz"
            with (
                np.load(capacity / filename, allow_pickle=False) as a,
                np.load(store.path / filename, allow_pickle=False) as b,
            ):
                if set(a.files) != set(b.files) or any(
                    not np.array_equal(a[k], b[k]) for k in a.files
                ):
                    raise ValueError(f"Different {split} rows or native scores")
                n += a["top"].size
            if read(capacity / f"{split}_inputs.json") != read(
                store.path / f"{split}_inputs.json"
            ):
                raise ValueError("Input binding differed")
        supplied = evaluate(np.arange(6) % 4, ASSIGNMENTS[:4], rewards)
    finally:
        native.close()
    f = fixture(base, old, design, "names")
    f["styles"]["dispatch"] = cfg["raw_style"]
    f["fresh"].update(flag_pairs=cfg["raw_flags"], styles=["dispatch"])
    cases = records(actions["groups"], f, "fresh")
    values = bank(host, cases, store, "test")
    expected_caps = [
        r
        for r in map(json.loads, (raw / "captures.jsonl").read_text().splitlines())
        if r["split"] == "test"
    ]
    if (
        list(map(json.loads, (store.path / "captures.jsonl").read_text().splitlines()))
        != expected_caps
    ):
        raise ValueError("Different encoded teachers")
    with gzip.open(raw / "decisions.jsonl.gz", "rt") as stream:
        saved = {
            (r["arm"], r["key"], r["route"], r["world"], r["order"]): r
            for r in map(json.loads, stream)
        }
    count = len(saved)
    for arm, path in (("parent", FOCUS), ("trained", raw / "reader.npz")):
        reader, _, sha = load_recipient(path, host)
        if (
            sha
            != read(raw / "fit_closed.json")[
                "weights" if arm == "trained" else "parent"
            ]
        ):
            raise ValueError("Different raw reader")
        try:
            for k, c in cases.items():
                jobs = questions(c, actions["groups"], f)
                z = values[k].to(host.device)
                other = values[(*k[:3], 1 - k[3], *k[4:])].to(host.device)
                for route, state in (
                    ("matched", z),
                    ("opposed", other),
                    ("zero", torch.zeros_like(z)),
                    ("literal", None),
                ):
                    scores = model_scores(
                        host,
                        reader,
                        jobs,
                        {c["key"]: state},
                        "literal_base" if state is None else "matched",
                    )
                    for t, out in zip(jobs, scores, strict=True):
                        r = saved.pop(
                            (arm, c["key"], route, t["world"], t["label_order"])
                        )
                        if (
                            any(out[a] != r[a] for a in out)
                            or digest(t["question"]) != r["question"]
                            or t["gold"] != r["gold"]
                        ):
                            raise ValueError("Different native decision")
        finally:
            reader.close()
    if saved or count != 10240 or n != 216:
        raise ValueError("Incomplete replay")
    store.write(
        "receipt.json",
        dict(
            raw_decisions=count,
            capacity_decisions=n,
            raw_state_hashes=len(values),
            cpu_code_comparison=supplied,
            backbone_checkpoint=CHECKPOINT_SHA,
            raw_scores=file_sha(raw / "decisions.jsonl.gz"),
            compute=host.meter.snapshot(),
        ),
    )


if __name__ == "__main__":
    p = parser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--capacity", type=Path, required=True)
    run(p.parse_args())
