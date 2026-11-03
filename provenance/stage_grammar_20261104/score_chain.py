"""Reduce native eight-slot and separately decoded rule choices without pooling interfaces."""

import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from selffold.conditional_curriculum import make_cases, tasks_for
from selffold.records import digest, file_sha
from selffold.stage_grammar import fixture
from summarize import ROOT, above, old, ratio, read

RAW = (
    "source",
    "case_teacher",
    "rule_teacher",
    "g1",
    "g2_only",
    "both",
    "opposed",
    "equivalent",
    "mean",
    "zero",
)


def summarize(run):
    term = read(run / "terminal.json")
    closed = read(run / "maps_closed.json")
    if (
        term["status"] != "DEVELOPMENT_G_CHAIN_COMPLETE"
        or closed["artifact"] != file_sha(run / "maps.npz")
        or closed["fresh_captures"] != 0
    ):
        raise ValueError("Unbound chain")
    base, prior, design, action = [
        read(ROOT / "data" / x)
        for x in (
            "conditional_margin_20261103.json",
            "two_flags_20261104.json",
            "stage_grammar_20261104.json",
            "within_topic_wording_20261103.json",
        )
    ]
    groups = action["groups"]
    f = fixture(base, prior, design, "fresh")
    cs = [c for c in make_cases(groups, f, "fresh") if c["stage"] == "source"]
    target = read(run / "targets.json")
    states = read(run / "state_index.json")["routes"]
    if target["keys"] != [c["key"] for c in cs] or len(cs) != 64:
        raise ValueError("Incomplete source bank")
    labels = {
        v: k
        for k, v in read(ROOT / "provenance/two_flags_20261104/run/labels.json").items()
    }
    result = {}
    paired = {}
    gold_id = [2 * c["topic"] + c["pole"] for c in cs]
    with gzip.open(run / "decisions.jsonl.gz", "rt") as io:
        for mode in ("raw", "decoded"):
            records = {}
            tables = {}
            for route in (*RAW, *(("oracle",) if mode == "decoded" else ())):
                rows = []
                hits = []
                margins = []
                letters = []
                for i, c in enumerate(cs):
                    src = (
                        c
                        if mode == "raw"
                        else c | dict(flags=base["fit"]["flag_pairs"][0])
                    )
                    statekey = (
                        states[route][i]
                        if mode == "raw"
                        else (
                            int(gold_id[i])
                            if route == "oracle"
                            else target["predicted"][route][i]
                        )
                    )
                    for t in (
                        t
                        for w in range(4)
                        for t in tasks_for(src, groups[c["topic"]], f, w)
                    ):
                        r = json.loads(next(io))
                        if (
                            r[:6]
                            != [
                                mode,
                                route,
                                c["key"],
                                t["world"],
                                t["label_order"],
                                t["gold"],
                            ]
                            or r[9:] != [digest(t["question"]), statekey]
                            or not all(np.isfinite(r[7:9]))
                        ):
                            raise ValueError("Wrong native decision binding")
                        letter = labels.get(r[6])
                        rows.append(
                            dict(
                                key=c["key"],
                                world=t["world"],
                                label_order=t["label_order"],
                                gold=t["gold"],
                                greedy_label=letter,
                                A=r[7],
                                B=r[8],
                            )
                        )
                        hits.append(letter == t["gold"])
                        letters.append(letter)
                        margins.append(r[7] - r[8])
                metrics = old.prior.measure(rows)
                records[route] = {
                    k: metrics[k]
                    for k in (
                        "native",
                        "strict",
                        "coverage",
                        "forced",
                        "ties",
                        "per_world",
                    )
                }
                tables[route] = (np.array(hits), np.array(letters), np.array(margins))
            main, opp, same = (tables[k] for k in ("both", "opposed", "equivalent"))
            joint = (
                main[0] & ~opp[0] & np.isin(opp[1], ("A", "B")) & (main[2] * opp[2] < 0)
            )  # a valid opposing letter is required
            both_stages = main[0] & tables["g1"][0]

            def count(a):
                return [int(a.sum()), int(a.size)]

            paired[mode] = dict(
                opposed=count(joint),
                same=count(main[0] & same[0]),
                both_stages=count(both_stages),
                raw_margin_flip=count(main[2] * opp[2] < 0),
            )
            result[mode] = records
        if io.read():
            raise ValueError("Unexpected extra score")
    wins = {
        k: dict(
            correct=ratio(v > 0 for v in vs),
            ties=sum(v == 0 for v in vs),
            mean=float(np.mean(vs)),
        )
        for k, vs in target["target_gap"].items()
    }
    types = {
        k: dict(
            complete=ratio(x == gold for x, gold in zip(xs, gold_id)),
            wrong_topic=sum(x // 2 != g // 2 for x, g in zip(xs, gold_id)),
        )
        for k, xs in target["predicted"].items()
    }

    def rate(p):
        return p[0] / p[1]

    raw = result["raw"]
    full = raw["both"]
    vs = paired["raw"]
    gate = dict(
        teacher_all=all(
            above(raw[n]["native"], 90) and above(raw[n]["strict"], 75)
            for n in ("source", "case_teacher", "rule_teacher")
        ),
        final_native=above(full["native"], 90),
        final_strict=above(full["strict"], 75),
        opposed=above(vs["opposed"], 90),
        same=above(vs["same"], 90),
        both_stages=above(vs["both_stages"], 70),
        null_gap=rate(full["native"])
        - max(rate(raw[k]["native"]) for k in ("zero", "mean"))
        >= 0.1,
        case_target=above(wins["g1_case"]["correct"], 90),
        rule_target=above(wins["full_rule"]["correct"], 90),
        g1_added=rate(wins["g1_case"]["correct"]) - rate(wins["z0_case"]["correct"])
        >= 0.1,
        g2_added=rate(wins["full_rule"]["correct"]) - rate(wins["g1_rule"]["correct"])
        >= 0.1,
    )
    return dict(
        routes=result,
        pairs=paired,
        target_preference=wins,
        decoder=types,
        raw_checks=gate,
        raw_development_pass=all(gate.values()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    print(json.dumps(summarize(p.parse_args().run), indent=2))
