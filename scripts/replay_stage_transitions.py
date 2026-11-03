"""Replay saved raw states and decoded rule choices with the frozen native reader."""

import gzip
import json
from pathlib import Path

import numpy as np
import torch
from fit_conditional_curriculum import parser
from fit_continuity_recipient import model_scores
from fit_stage_grammar import ACTIONS, BASE, DESIGN, PREVIOUS, read
from selffold.cli import make_host
from selffold.conditional_curriculum import make_cases, tasks_for
from selffold.records import RunStore, State, digest, file_sha
from selffold.recipient_learning import load_recipient
from selffold.stage_grammar import fixture


@torch.no_grad()
def replay(args):
    store = RunStore(args.out)
    base, prev, d, action = map(read, (BASE, PREVIOUS, DESIGN, ACTIONS))
    groups = action["groups"]
    f = fixture(base, prev, d, "fresh")
    cases = [c for c in make_cases(groups, f, "fresh") if c["stage"] == "source"]
    # One description for every topic, alternating name pair, pole and mention order across topics.
    sample = {
        c["key"]: c
        for c in cases
        if c["flag_pair"] == c["topic"] % 2
        and c["pole"] == (c["topic"] // 2) % 2
        and c["source_order"] == (c["topic"] // 4) % 2
    }
    all_lookup = {c["key"]: i for i, c in enumerate(cases)}
    names = (
        "source",
        "case_teacher",
        "rule_teacher",
        "g1",
        "both",
        "opposed",
        "mean",
        "zero",
    )
    saved = {}
    with gzip.open(args.maps / "decisions.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row[2] in sample and row[1] in names:
                saved[tuple(row[:5])] = row
    idx = read(args.maps / "state_index.json")
    pred = read(args.maps / "targets.json")["predicted"]
    with np.load(args.maps / "fresh_states.npz", allow_pickle=False) as fp:
        values = torch.from_numpy(fp["states"]).view(torch.bfloat16)
    rulebook = {
        2 * c["topic"] + c["pole"]: c["source"]
        for c in make_cases(groups, base, "fit")
        if c["stage"] == "rule"
        and c["style"] == base["fit"]["styles"][0]
        and c["flag_pair"] == 0
    }
    host, _ = make_host(args)
    reader, _, sha = load_recipient(args.literal / "focus.npz", host)
    if sha != read(args.maps / "terminal.json")["reader_sha256"]:
        raise ValueError("Wrong reader")
    try:
        for mode in ("raw", "decoded"):
            for route in names:
                for key, c in sample.items():
                    pos = all_lookup[key]
                    if mode == "raw":
                        address = idx["routes"][route][pos]
                        z = values[address].to(host.device)
                        if State(slots=z).sha() != idx["sha256"][address]:
                            raise ValueError("Raw tensor differs")
                        prompt = c
                        kind = "matched"
                    else:
                        z = None
                        kind = "literal_base"
                        prompt = c | dict(
                            flags=base["fit"]["flag_pairs"][0],
                            source=rulebook[pred[route][pos]],
                        )
                    jobs = [
                        j
                        for w in range(4)
                        for j in tasks_for(prompt, groups[c["topic"]], f, w)
                    ]
                    outputs = model_scores(host, reader, jobs, {key: z}, kind)
                    for task, score in zip(jobs, outputs, strict=True):
                        row = saved.pop(
                            (mode, route, key, task["world"], task["label_order"])
                        )
                        if row[6:10] != [
                            score["greedy_token"],
                            score["A"],
                            score["B"],
                            digest(task["question"]),
                        ]:
                            raise ValueError("Native replay differs")
    finally:
        reader.close()
    if saved or len(sample) != 8:
        raise ValueError("Missing replay keys")
    store.write(
        "receipt.json",
        dict(
            checkpoint=host.identity["checkpoint_sha256"],
            reader=sha,
            source_count=len(sample),
            routes=names,
            modes=["raw", "decoded"],
            exact_native_decisions=len(sample) * len(names) * 2 * 8,
            raw_digest=file_sha(args.maps / "decisions.jsonl.gz"),
            compute=host.meter.snapshot(),
        ),
    )


if __name__ == "__main__":
    p = parser()
    p.add_argument("--literal", type=Path, required=True)
    p.add_argument("--maps", type=Path, required=True)
    replay(p.parse_args())
