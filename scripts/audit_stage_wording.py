"""Cross the wording of the source, reported bit and decision question at fixed facts."""

from collections import defaultdict
import gzip
import json
import time
from itertools import product
from pathlib import Path
import torch
from fit_codec import CHECKPOINT_SHA
from fit_conditional_curriculum import parser
from fit_stage_grammar import (
    ACTIONS,
    BASE,
    DESIGN,
    PARENT,
    PREVIOUS,
    STUDY,
    choose_logits,
    native_inputs,
    read,
)
from selffold.cli import ROOT, make_host
from selffold.conditional_curriculum import make_cases, tasks_for
from selffold.records import RunStore, digest, file_sha
from selffold.recipient_learning import load_recipient
from selffold.stage_grammar import fixture, panel_jobs
from selffold.two_flags import branch_probe


def query(host, reader, tasks, labels, legacy):
    inputs = native_inputs(host, tasks)
    logits = choose_logits(host, reader, inputs, len(inputs[0][0]), legacy)
    return [
        [w, *p]
        for w, p in zip(
            logits.argmax(-1).tolist(),
            logits.log_softmax(-1)[:, list(labels.values())].tolist(),
            strict=True,
        )
    ]


def summarize(records):
    lookup = {(r[0], r[1], r[2]): r for r in records}
    out = {}
    for stage in ("source", "cases", "rule"):
        units = defaultdict(list)
        perbit = defaultdict(list)
        weighted = []
        bp = []
        op = []
        for key, bit, order, gold, letter, *_ in records:
            split, group, flag, style, st, pole, source_order = key.split("/")
            if st != stage:
                continue
            ok = letter == gold
            units[group, flag].append(ok)
            perbit[bit].append(ok)
            weighted.extend([ok] * (1 if bit else 3))
            other = "/".join(
                (split, group, flag, style, st, str(1 - int(pole)), source_order)
            )
            opposite = "B" if gold == "A" else "A"
            bp.append(ok and lookup[key, 1 - bit, order][4] == opposite)
            op.append(ok and lookup[other, bit, order][4] == opposite)

        def count(x):
            return [sum(x), len(x)]

        out[stage] = dict(
            balanced=count(perbit[0] + perbit[1]),
            false_bit=count(perbit[0]),
            true_bit=count(perbit[1]),
            four_world=count(weighted),
            strict=count([all(v) for v in units.values()]),
            bit_pair=count(bp),
            policy_pair=count(op),
        )
    return out


@torch.no_grad()
def run(args, store):
    base, prev, d, actions = [read(p) for p in (BASE, PREVIOUS, DESIGN, ACTIONS)]
    groups = {g["id"]: g for g in actions["groups"]}
    host, ident = make_host(args)
    if ident["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Wrong checkpoint")
    labels = read(args.literal / "labels.json")
    decoder = {i: t for t, i in labels.items()}
    focus, _, sha = load_recipient(args.literal / "focus.npz", host)
    if sha != "bf4d476cf50825a92ce41d2e8430336b7a0998d31143abe03ba3a38b11e168d3":
        raise ValueError("Wrong focus endpoint")
    report = {}
    try:
        with gzip.open(store.path / "cross.jsonl.gz", "wt") as stream:
            for grammar, bitword, question in product((0, 1), repeat=3):
                f = fixture(base, prev, d, "fresh")
                settings = f["fresh"]
                if not grammar:
                    settings["styles"] = base["fit"]["styles"][:1]
                if not question:
                    settings.update(
                        question=base["fit"]["question"], operations=base["operations"]
                    )
                words = d["fresh_probes"] if bitword else prev["fit_probes"]
                tag = f"{grammar}{bitword}{question}"
                rows = []
                cases = list(make_cases(list(groups.values()), f, "fresh"))
                for ix in range(0, len(cases), 2):
                    jobs = [
                        (c, bit, t)
                        for c in cases[ix : ix + 2]
                        for bit in (False, True)
                        for t in branch_probe(c, groups[c["group"]], f, words, bit)
                    ]
                    scores = query(host, focus, [t for c, b, t in jobs], labels, False)
                    for (c, bit, t), score in zip(jobs, scores, strict=True):
                        r = [
                            c["key"],
                            int(bit),
                            t["label_order"],
                            t["gold"],
                            decoder.get(score[0]),
                            *score,
                            digest(t["source"]),
                            digest(t["question"]),
                        ]
                        rows.append(r)
                        stream.write(
                            json.dumps([tag, *r], separators=(",", ":")) + "\n"
                        )
                report[tag] = summarize(rows)
    finally:
        focus.close()
    store.write("cross.json", report)
    reader, _, _ = load_recipient(PARENT, host)
    archived = {}
    for line in (
        (ROOT / "provenance/two_flags_20261104/run/choices.jsonl")
        .read_text()
        .splitlines()
    ):
        r = json.loads(line)
        if r["arm"] == "parts" and r["panel"] == "names+grammar+question":
            archived[r["key"], r["world"], r["label_order"]] = [
                r["greedy_token"],
                r["A"],
                r["B"],
                r["source_sha256"],
                r["question_sha256"],
            ]
    with gzip.open(args.literal / "parent.jsonl.gz", "rt") as stream:
        latest = {
            tuple(r[2:5]): r[6:11]
            for r in map(json.loads, stream)
            if r[:2] == ["previous", "ordinary"]
        }
    f = fixture(base, prev, d, "previous")
    outcomes = {}
    try:
        prior, full, thin = {}, {}, {}
        for c in make_cases(list(groups.values()), f, "fresh"):
            ts = [t for w in range(4) for t in tasks_for(c, groups[c["group"]], f, w)]
            for j in (0, 4):
                batch = ts[j : j + 4]
                values = query(host, reader, batch, labels, True)
                for t, v in zip(batch, values):
                    prior[c["key"], t["world"], t["label_order"]] = [
                        *v,
                        digest(t["source"]),
                        digest(t["question"]),
                    ]
        jobs = list(panel_jobs(groups, f, prev["fresh_probes"], "previous"))
        for i in range(0, len(jobs), 8):
            batch = jobs[i : i + 8]
            ts = [r[-1] for r in batch]
            for target, legacy in ((full, True), (thin, False)):
                for (kind, k, w, o, t), v in zip(
                    batch, query(host, reader, ts, labels, legacy)
                ):
                    if kind == "ordinary":
                        target[k, w, o] = [
                            *v,
                            digest(t["source"]),
                            digest(t["question"]),
                        ]
        with gzip.open(store.path / "batch.jsonl.gz", "wt") as stream:
            for k in archived:
                stream.write(
                    json.dumps(
                        [*k, archived[k], prior[k], full[k], thin[k], latest[k]],
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        for tag, a, b in (
            ("old4_full_vs_archive", prior, archived),
            ("new8_selected_vs_current", thin, latest),
            ("new8_full_vs_selected", full, thin),
            ("old4_vs_new8_full", prior, full),
        ):
            if (
                set(a) != set(b)
                or len(a) != 1280
                or any(a[k][3:] != b[k][3:] for k in a)
            ):
                raise ValueError("Different scored prompts")
            outcomes[tag] = dict(
                n=len(a),
                identical=sum(a[k] == b[k] for k in a),
                winner_changes=sum(a[k][0] != b[k][0] for k in a),
                max_logprob_delta=max(
                    abs(a[k][j] - b[k][j]) for k in a for j in (1, 2)
                ),
            )
    finally:
        reader.close()
    store.write("batch.json", outcomes)
    return dict(
        status="FROZEN_WORDING_AUDIT_COMPLETE",
        reader=sha,
        plan=file_sha(STUDY / "WORDING_AUDIT.md"),
        compute=host.meter.snapshot(),
        training_updates=0,
    )


if __name__ == "__main__":
    p = parser()
    p.add_argument("--literal", type=Path, required=True)
    a = p.parse_args()
    store = RunStore(a.out)
    now = time.monotonic()
    try:
        r = run(a, store)
    except Exception as exc:
        store.write("terminal.json", dict(status="TECHNICAL_STOP", error=str(exc)))
        raise
    store.write("terminal.json", r | dict(elapsed_seconds=time.monotonic() - now))
