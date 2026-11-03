"""Evaluate singleton policies and their opposed pairs without re-encoding any sources."""

from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import sys

STUDY = Path(__file__).resolve().parent
spec = spec_from_file_location(
    "previous_cue_measures", STUDY.parent / "matched_cues_20261103/summarize.py"
)
prev = module_from_spec(spec)
spec.loader.exec_module(prev)
old = prev.old
CELLS = ("accept_before", "accept_after", "choice_before", "choice_after")


def policy(t, score):
    return 2 * t + int(score < 0) if t is not None and score != 0 else None


def decisions(r):
    result = dict(r)
    for name in ("weighted", "ordinary"):
        h = r[name]
        t = h["topic"]
        a = h["actions"][t] if t is not None else 0.0
        result[name] = policy(t, h["role"] * a)
        result[name + "_signed"] = h["role"] * a
        if name == "weighted":
            result.update(
                {
                    "topic_ok": t == r["topic"],
                    "action_ok": t == r["topic"]
                    and a * (1 - 2 * (r["pole"] ^ r["order"])) > 0,
                    "role_ok": h["role"] * (1 - 2 * r["order"]) > 0,
                    "null_role_ok": h["shuffled_role"] * (1 - 2 * r["order"]) > 0,
                    "both_wrong": t == r["topic"]
                    and a * (1 - 2 * (r["pole"] ^ r["order"])) < 0
                    and h["role"] * (1 - 2 * r["order"]) < 0,
                    "direct": policy(t, h["directs"][t] if t is not None else 0.0),
                    "mention": policy(t, a),
                    "shuffled": policy(t, h["shuffled_role"] * a),
                    "oracle": policy(r["topic"], h["role"] * h["actions"][r["topic"]]),
                    "role_signed": h["role"],
                }
            )
    return result


def opposed(data, method="weighted"):
    by = {(r["topic"], r["frame"], r["pair"], r["pole"], r["order"]): r for r in data}
    if len(by) != len(data):
        raise ValueError("Duplicate singleton")
    result = []
    for (t, f, k, p, o), plus in by.items():
        if p:
            continue
        minus = by[t, f, k, 1, 1 - o]
        a, b = plus[method], minus[method]
        correct = a == 2 * t and b == 2 * t + 1
        inverted = a == 2 * t + 1 and b == 2 * t
        same = a == b and a in (2 * t, 2 * t + 1)
        eligible = plus["topic_ok"] and minus["topic_ok"]
        rank = eligible and plus[method + "_signed"] > minus[method + "_signed"]
        role_rank = (plus["role_signed"] - minus["role_signed"]) * (1 - 2 * o) > 0
        result.append(
            {
                "topic": t,
                "frame": f,
                "pair": k,
                "order": o,
                "correct": correct,
                "inverted": inverted,
                "same": same,
                "other": not (correct or inverted or same),
                "rank": rank,
                "rank_without_both": rank and not correct,
                "role_both": plus["role_ok"] and minus["role_ok"],
                "role_rank": role_rank,
                "tie": eligible
                and plus[method + "_signed"] == minus[method + "_signed"],
            }
        )
    return result


def metrics(data):
    pairs = opposed(data)
    methods = {
        k: old.source_measure(data, k)
        for k in (
            "weighted",
            "ordinary",
            "direct",
            "mention",
            "shuffled",
            "old",
            "oracle",
        )
    }
    return {
        "methods": methods,
        "component": {
            k: old.count(r[k] for r in data)
            for k in ("topic_ok", "action_ok", "role_ok", "null_role_ok", "both_wrong")
        }
        | {"both_correct": old.count(r["role_ok"] and r["action_ok"] for r in data)},
        "exchange": {
            k: old.count(r[k] for r in pairs)
            for k in (
                "correct",
                "inverted",
                "same",
                "other",
                "rank",
                "rank_without_both",
                "tie",
                "role_both",
                "role_rank",
            )
        },
    }


def gain(a, b, points):
    if a[1] != b[1]:
        raise ValueError("Paired denominators differ")
    return (a[0] - b[0]) * 100 >= points * a[1]


def summary(folder):
    data = [decisions(r) for r in old.rows(folder / "scores.jsonl")]
    panels = {}
    for split, n in (("fit", 4608), ("fresh", 2048)):
        block = [r for r in data if r["split"] == split]
        if len(block) != n or len({r["key"] for r in block}) != n:
            raise ValueError("Incomplete state panel")
        panels[split + "/all"] = metrics(block)
        for cell in CELLS:
            subset = [r for r in block if r["frame"] == cell]
            assert len(subset) * 4 == n
            panels[split + "/" + cell] = metrics(subset)
    fit = panels["fit/all"]["methods"]["weighted"]
    fit_gate = old.passes(fit["policy"], 95) and old.passes(fit["strict"], 90)
    cells = {}
    for f in CELLS:
        d = panels["fresh/" + f]
        m = d["methods"]
        w = m["weighted"]
        c = d["component"]
        cells[f] = {
            "policy": all(
                old.passes(w[k], threshold)
                for k, threshold in (
                    ("policy", 90),
                    ("strict", 75),
                    ("opposed_same_action_order", 90),
                    ("opposed_same_role_order", 90),
                )
            )
            and all(
                gain(w["policy"], m[j]["policy"], 10) for j in ("shuffled", "mention")
            ),
            "role_only": old.passes(c["role_ok"], 95)
            and old.passes(d["exchange"]["role_both"], 90)
            and gain(c["role_ok"], c["null_role_ok"], 20),
        }
    fresh = [r for r in data if r["split"] == "fresh"]
    correct = lambda r, m: r[m] == 2 * r["topic"] + r["pole"]
    weighted = [dict(r, ok=correct(r, "weighted")) for r in fresh]
    ordinary = [dict(r, ok=correct(r, "ordinary")) for r in fresh]
    cmp = {
        "singleton": prev.paired(weighted, ordinary, ("key",), lambda r: r["ok"]),
        "opposed": prev.paired(
            opposed(fresh),
            opposed(fresh, "ordinary"),
            ("topic", "frame", "pair", "order"),
            lambda r: r["correct"],
        ),
    }
    effect = all(d["net"] * 100 >= 10 * d["n"] for d in cmp.values())
    literal = old.rows(folder / "literal.jsonl")
    assert len(literal) == len({(r["key"], r["label_order"]) for r in literal}) == 4096
    lit = {}
    for name, group in [("all", literal)] + [
        (f, [r for r in literal if r["frame"] == f]) for f in CELLS
    ]:
        idx = {(r["key"], r["label_order"]): r for r in group}
        lit[name] = old.native(group) | {
            "both_labels": old.count(
                old.correct(r) and old.correct(idx[k, 1])
                for (k, o), r in idx.items()
                if o == 0
            )
        }
    return {
        "panels": panels,
        "fresh_by_topic": {
            g: metrics([r for r in fresh if r["group"] == g])
            for g in sorted({r["group"] for r in fresh})
        },
        "paired_weighted_minus_ordinary": cmp,
        "literal": lit,
        "gates": {
            "fit": fit_gate,
            "fresh_cells": cells,
            "source_recovery": fit_gate and all(d["policy"] for d in cells.values()),
            "role_only": all(d["role_only"] for d in cells.values()),
            "extra_weighting_benefit": effect,
        },
    }


if __name__ == "__main__":
    print(
        json.dumps(
            summary(Path(sys.argv[1]) if len(sys.argv) > 1 else STUDY / "run"), indent=2
        )
    )
