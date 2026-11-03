"""Registered component gates, retaining gold and inferred topics and native conditional worlds."""

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from collections import defaultdict
import json, sys

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from conditional_teacher_audit import summarize as audit_summary

s = spec_from_file_location(
    "previous_measures", STUDY.parent / "matched_cues_20261103/summarize.py"
)
prev = module_from_spec(s)
s.loader.exec_module(prev)
old = prev.old
CELLS = ("accept_before", "accept_after", "choice_before", "choice_after")


def action_metric(data, arm, mode):
    index = {
        (r["topic"], r["frame"], r["pair"], r["pole"], r["order"]): r for r in data
    }
    ok = {}
    unit = defaultdict(list)
    for k, r in index.items():
        t = r["topic"] if mode == "gold" else r["frozen"]["topic"]
        score = (
            (r["frozen"]["actions"] if arm == "old" else r[arm])[t]
            if t is not None
            else 0
        )
        good = t == r["topic"] and score * (1 - 2 * (r["pole"] ^ r["order"])) > 0
        ok[k] = good
        unit[k[:3]].append(good)
    assert len(index) == len(data) and all(len(v) == 4 for v in unit.values())
    return {
        "singletons": old.count(ok.values()),
        "strict": old.count(all(v) for v in unit.values()),
        "pairs": old.count(
            v and ok[t, f, p, 1 - pole, o]
            for (t, f, p, pole, o), v in ok.items()
            if pole == 0
        ),
    }


def policy_metric(data, arm, mode):
    keyed = {
        (r["topic"], r["frame"], r["pair"], r["pole"], r["order"]): r for r in data
    }
    ok = {}
    units = defaultdict(list)
    for k, r in keyed.items():
        t = r["topic"] if mode == "gold" else r["frozen"]["topic"]
        value = 0 if t is None else r[arm][t] * r["frozen"]["role"]
        ok[k] = t == r["topic"] and value * (1 - 2 * r["pole"]) > 0
        units[k[:3]].append(ok[k])
    return {
        "singletons": old.count(ok.values()),
        "strict": old.count(all(v) for v in units.values()),
        "opposed_role": old.count(
            v and ok[t, f, p, 1 - pole, 1 - o]
            for (t, f, p, pole, o), v in ok.items()
            if pole == 0
        ),
        "opposed_action": old.count(
            v and ok[t, f, p, 1 - pole, o]
            for (t, f, p, pole, o), v in ok.items()
            if pole == 0
        ),
    }


def source_summary(raw):
    panels = {}
    for split, total in (("fit", 4608), ("fresh", 2048)):
        data = [r for r in raw if r["split"] == split]
        assert len(data) == len({r["key"] for r in data}) == total
        for cell in ("all", *CELLS):
            subset = [r for r in data if cell == "all" or r["frame"] == cell]
            panels[split + "/" + cell] = {
                "action": {
                    arm + "/" + mode: action_metric(subset, arm, mode)
                    for arm in ("old", "swap", "invariant")
                    for mode in ("gold", "inferred")
                },
                "policy": {
                    arm + "/" + mode: policy_metric(subset, arm, mode)
                    for arm in ("swap", "invariant")
                    for mode in ("gold", "inferred")
                },
                "role": old.count(
                    r["frozen"]["role"] * (1 - 2 * r["order"]) > 0 for r in subset
                ),
                "topic": old.count(r["frozen"]["topic"] == r["topic"] for r in subset),
            }
    gates = {}
    for mode in ("gold", "inferred"):
        f = panels["fit/all"]["action"]["invariant/" + mode]
        passed = old.passes(f["singletons"], 95) and old.passes(f["strict"], 90)
        cell_gates = {}
        for cell in CELLS:
            a = panels["fresh/" + cell]["action"]
            w = a["invariant/" + mode]
            base = a["old/" + mode]
            cell_gates[cell] = (
                all(
                    old.passes(w[k], th)
                    for k, th in (("singletons", 90), ("strict", 75), ("pairs", 90))
                )
                and (w["singletons"][0] - base["singletons"][0]) * 100
                >= 10 * w["singletons"][1]
            )
        gates[mode] = {
            "fit": passed,
            "cells": cell_gates,
            "admission": passed and all(cell_gates.values()),
        }
    fresh = panels["fresh/all"]["action"]
    gates["covariance_effect"] = all(
        (fresh["invariant/gold"][k][0] - fresh["swap/gold"][k][0]) * 100
        >= 10 * fresh["invariant/gold"][k][1]
        for k in ("singletons", "pairs")
    )
    return {"panels": panels, "gates": gates}


def teacher_summary(data):
    stats = {}
    for mode in ("generic", "concrete"):
        for route in ("calibrated", "mean"):
            group = [r for r in data if r["mode"] == mode and r["route"] == route]
            idx = {
                (r["topic"], r["family"], r["pole"], r["label_order"]): r for r in group
            }
            assert len(idx) == len(group) == 128
            stats[mode + "/" + route] = old.native(group) | {
                "families": [
                    old.native([r for r in group if r["family"] == f])["native"]
                    for f in range(4)
                ],
                "topics": [
                    old.native([r for r in group if r["topic"] == t])["native"]
                    for t in range(8)
                ],
                "strict": old.count(
                    all(old.correct(r) for k, r in idx.items() if k[0] == t)
                    for t in range(8)
                ),
                "flips": old.count(
                    (r["A"] - r["B"]) * (idx[t, f, 1, o]["A"] - idx[t, f, 1, o]["B"])
                    < 0
                    for (t, f, p, o), r in idx.items()
                    if p == 0
                ),
            }
    concrete = stats["concrete/calibrated"]
    reliable = (
        old.passes(concrete["native"], 95)
        and all(old.passes(n, 95) for n in concrete["families"])
        and concrete["strict"][0] >= 7
        and old.passes(concrete["flips"], 90)
        and stats["concrete/mean"]["native"][0] <= 64
    )
    left = [r for r in data if r["mode"] == "concrete" and r["route"] == "calibrated"]
    right = [r for r in data if r["mode"] == "generic" and r["route"] == "calibrated"]
    paired = prev.paired(
        left, right, ("topic", "family", "pole", "label_order"), old.correct
    )
    fd = [
        concrete["families"][i][0] - stats["generic/calibrated"]["families"][i][0]
        for i in range(4)
    ]
    benefit = (
        paired["net"] * 100 >= 10 * paired["n"]
        and sum(d > 0 for d in fd) >= 3
        and sum(d > 0 for d in paired["by_topic_net"].values()) >= 5
    )
    return {
        "metrics": stats,
        "paired": paired,
        "by_family_net": fd,
        "reliability": reliable,
        "operation_benefit": benefit,
    }


def summary(folder):
    return {
        "source": source_summary(old.rows(folder / "source_scores.jsonl")),
        "teacher": teacher_summary(old.rows(folder / "teachers.jsonl")),
        "conditional": audit_summary(old.rows(folder / "audit_decisions.jsonl")),
    }


if __name__ == "__main__":
    print(
        json.dumps(
            summary(Path(sys.argv[1]) if len(sys.argv) > 1 else STUDY / "run"), indent=2
        )
    )
