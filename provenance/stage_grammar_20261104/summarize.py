"""Regenerate every literal question and compare unchanged native, strict and intervention criteria."""

from collections import Counter, defaultdict
import argparse
import gzip
import importlib.util
import json
import math
from pathlib import Path

from selffold.conditional_curriculum import STAGES, make_cases
from selffold.records import digest
from selffold.stage_grammar import ARMS, PANELS, fixture, panel_jobs

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "twoflags_original", ROOT / "provenance/two_flags_20261104/summarize.py"
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def read(p):
    return json.loads(p.read_text())


def above(pair, q):
    return pair[0] * 100 >= q * pair[1]


def ratio(items):
    x = list(items)
    return [sum(x), len(x)]


def reduce(run):
    if read(run / "terminal.json")["status"] != "LITERAL_STAGE_RUN_COMPLETE":
        raise ValueError("Incomplete trial")
    base, prev, design, options = [
        read(ROOT / "data" / p)
        for p in (
            "conditional_margin_20261103.json",
            "two_flags_20261104.json",
            "stage_grammar_20261104.json",
            "within_topic_wording_20261103.json",
        )
    ]
    groups = {g["id"]: g for g in options["groups"]}
    ids = read(run / "labels.json")
    labels = {v: k for k, v in ids.items()}
    results = {}
    fixed = {}
    for arm in ("parent", *ARMS):
        result = {}
        with gzip.open(run / (arm + ".jsonl.gz"), "rt", encoding="utf8") as stream:
            for panel in PANELS:
                if arm == "parent" and panel not in ("fresh", "previous"):
                    continue
                f = fixture(base, prev, design, "train" if panel == "fit" else panel)
                split = "fit" if panel == "fit" else "fresh"
                bank = list(make_cases(list(groups.values()), f, split))
                cases = {c["key"]: c for c in bank}
                ordinary = defaultdict(list)
                world = {}
                branches = {}
                strict_original = {}
                err = Counter()
                words = (
                    prev["fresh_probes"]
                    if panel == "previous"
                    else design["fresh_probes"]
                )
                flags = defaultdict(list)
                for kind, key, index, order, t in panel_jobs(groups, f, words, panel):
                    row = json.loads(next(stream))
                    if (
                        row[:6] != [panel, kind, key, index, order, t["gold"]]
                        or row[9:] != [digest(t["source"]), digest(t["question"])]
                        or not all(math.isfinite(x) for x in row[7:9])
                    ):
                        raise ValueError("Unbound native score")
                    record = dict(
                        key=key,
                        world=index,
                        label_order=order,
                        gold=t["gold"],
                        greedy_label=labels.get(row[6]),
                        A=row[7],
                        B=row[8],
                    )
                    if kind == "ordinary":
                        c = cases[key]
                        stage = c["stage"]
                        ordinary[stage].append(record)
                        if record["greedy_label"] != t["gold"]:
                            err[stage, c["source_order"], t["gold"]] += 1
                        if panel == "fresh":
                            fixed[arm, key, index, order] = (
                                record["greedy_label"] == t["gold"]
                            )
                    elif kind == "branch":
                        branches[key, bool(index), order] = record
                    else:
                        pair, name, j = key.split("/")
                        world[int(pair), name, int(j), index, order] = record
                        flags[name + "_" + j, order].append(
                            record["greedy_label"] == t["gold"]
                        )
                ordinary_scores = {}
                for s in STAGES:
                    measures = old.prior.measure(ordinary[s])
                    ordinary_scores[s] = {
                        k: measures[k]
                        for k in (
                            "native",
                            "strict",
                            "coverage",
                            "forced",
                            "ties",
                            "per_world",
                        )
                    }
                    if panel == "fit":
                        strict_original[s] = old.prior.measure(
                            [
                                r
                                for r in ordinary[s]
                                if cases[r["key"]]["style"] in base["fit"]["styles"]
                            ]
                        )["strict"]
                data = dict(
                    ordinary=ordinary_scores,
                    errors={f"{s}/{o}/{g}": n for (s, o, g), n in sorted(err.items())},
                )
                if strict_original:
                    data["original_fit_strict"] = strict_original
                if world:
                    comp = old.mediated(bank, world, branches, f, groups)
                    probes = {
                        f"{k}/{order}": ratio(v) for (k, order), v in flags.items()
                    }
                    stage_pass = {
                        s: all(
                            above(comp[s][k], pct)
                            for k, pct in (
                                ("native", 90),
                                ("strict", 75),
                                ("correct_bit_exchange", 90),
                                ("correct_policy_exchange", 90),
                            )
                        )
                        for s in STAGES
                    }
                    data.update(
                        composed=comp,
                        world_components=probes,
                        stage_pass=stage_pass,
                        constructive=all(stage_pass.values())
                        and all(above(v, 90) for v in probes.values()),
                    )
                result[panel] = data
            if stream.read():
                raise ValueError("Extra decisions")
        if arm != "parent":

            def literalgate(panel, n, s):
                return all(
                    above(v["native"], n) and above(v["strict"], s)
                    for v in result[panel]["ordinary"].values()
                )

            result["literal_fit_gate"] = literalgate("fit", 95, 90)
            result["literal_fresh_gate"] = literalgate("fresh", 90, 75)
        results[arm] = result
    paired = {}
    for a, b in (("repeat", "vary"), ("vary", "focus"), ("parent", "focus")):
        keys = [(k, i, o) for arm, k, i, o in fixed if arm == a]
        paired[a + "_to_" + b] = dict(
            recovered=sum(not fixed[a, *k] and fixed[b, *k] for k in keys),
            lost=sum(fixed[a, *k] and not fixed[b, *k] for k in keys),
            n=len(keys),
        )
    train = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines()]
    by = {(r["arm"], r["step"]): r for r in train}
    if len(by) != len(train) or len(by) != len(ARMS) * design["steps"]:
        raise ValueError("Incomplete training")
    cost = {}
    for a in ARMS:
        rows = [by[a, i] for i in range(1, design["steps"] + 1)]
        totals = Counter()
        for r in rows:
            totals.update(r["counts"])
        cost[a] = dict(
            tasks=dict(totals),
            padded=sum(r["padded"] for r in rows),
            unmasked=sum(r["unmasked"] for r in rows),
            forwards=sum(r["forwards"] for r in rows),
        )
    if any(
        len({by[a, i]["padded"] for a in ARMS}) != 1
        for i in range(1, design["steps"] + 1)
    ):
        raise ValueError("Unequal padded training budget")
    return dict(results=results, paired_ordinary=paired, cost=cost)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    print(json.dumps(reduce(p.parse_args().run), indent=2))
