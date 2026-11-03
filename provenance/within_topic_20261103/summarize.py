"""Recompute the prospective within-topic screens from the raw native readouts."""

import json
from collections import defaultdict
from pathlib import Path
import sys

SIZES = {
    "fit": 2048,
    "questions": 2048,
    "source": 512,
    "joint": 512,
    "syntax": 256,
    "lexicon": 256,
}
FRESH = ("inclination", "decision")


def rows(file):
    return [json.loads(line) for line in file.read_text().splitlines()]


def ratio(n, d):
    return {"correct": int(n), "total": d, "fraction": n / d}


def key(row):
    return row["key"], row["wording"], row["label_order"]


def measure(data):
    units = defaultdict(list)
    for r in data:
        units[r["group"], r["unit"]].append(r)
    for unit in units.values():
        assert (
            len(unit)
            == len(
                {
                    (r["pole"], r["source_order"], r["wording"], r["label_order"])
                    for r in unit
                }
            )
            == 16
        )
    return {
        "native": ratio(sum(r["greedy_label"] == r["gold"] for r in data), len(data)),
        "forced": ratio(
            sum(r[r["gold"]] > r["B" if r["gold"] == "A" else "A"] for r in data),
            len(data),
        ),
        "coverage": sum(r["greedy_label"] in ("A", "B") for r in data),
        "forced_ties": sum(r["A"] == r["B"] for r in data),
        "strict": ratio(
            sum(
                all(r["greedy_label"] == r["gold"] for r in unit)
                for unit in units.values()
            ),
            len(units),
        ),
    }


def transfer_screen(routes):
    matched = routes["matched"]["native"]["fraction"]
    null = max(routes[r]["native"]["fraction"] for r in ("zero", "mean", "norm"))
    opposed = routes["opposed"]["native"]["fraction"]
    return (
        matched >= 0.9
        and routes["matched"]["strict"]["fraction"] >= 0.75
        and matched - null >= 0.1 - 1e-12
        and matched - opposed >= 0.3 - 1e-12
    )


def summary(folder):
    end = json.loads((folder / "terminal.json").read_text())
    if end["status"] != "COMPLETED_WITHIN_TOPIC_WORDING":
        return {"terminal": end}
    raw = rows(folder / "decisions.jsonl")
    assert len(raw) == 35328
    panels = defaultdict(list)
    for r in raw:
        panels[r["arm"], r["endpoint"], r["panel"], r["route"]].append(r)
    full, by_style, by_topic, swaps = {}, {}, {}, {}
    for (arm, endpoint, panel, route), group in sorted(panels.items()):
        assert len(group) == len({key(r) for r in group}) == SIZES[panel]
        name = "/".join((arm, endpoint, panel, route))
        full[name] = measure(group)
        if panel in ("joint", "syntax"):
            by_style[name] = {
                s: measure([r for r in group if r["style"] == s]) for s in FRESH
            }
            if panel == "joint" and route in ("matched", "teacher", "rendered_teacher"):
                by_topic[name] = {
                    t: measure([r for r in group if r["group"] == t])
                    for t in sorted({r["group"] for r in group})
                }
        if route == "opposed":
            other = {key(r): r for r in panels[arm, endpoint, panel, "matched"]}
            flips = sum(
                (r["A"] - r["B"]) * (other[key(r)]["A"] - other[key(r)]["B"]) < 0
                for r in group
            )
            swaps[name] = ratio(flips, len(group))
    screens = {}
    for arm in ("single", "many"):
        fit = full[f"{arm}/final/fit/matched"]
        verdict = {
            "fit": fit["native"]["fraction"] >= 0.95
            and fit["strict"]["fraction"] >= 0.9
        }
        for panel in ("joint", "syntax"):
            names = {
                r: f"{arm}/final/{panel}/{r}"
                for r in ("matched", "zero", "mean", "norm", "opposed")
            }
            verdict[panel] = {
                s: transfer_screen({r: by_style[n][s] for r, n in names.items()})
                for s in FRESH
            }
            verdict[panel]["combined"] = transfer_screen(
                {r: full[n] for r, n in names.items()}
            )
        verdict["within_topic_admission"] = verdict["fit"] and all(
            verdict["joint"].values()
        )
        verdict["syntax_only_screen"] = verdict["fit"] and all(
            verdict["syntax"].values()
        )
        screens[arm] = verdict
    single = {key(r): r for r in panels["single", "final", "joint", "matched"]}
    paired = {"both_correct": 0, "both_wrong": 0, "many_only": 0, "single_only": 0}
    for r in panels["many", "final", "joint", "matched"]:
        ref = single[key(r)]
        assert (r["source_sha256"], r["question_sha256"], r["gold"]) == (
            ref["source_sha256"],
            ref["question_sha256"],
            ref["gold"],
        )
        a, b = ref["greedy_label"] == r["gold"], r["greedy_label"] == r["gold"]
        paired[
            {
                (True, True): "both_correct",
                (False, False): "both_wrong",
                (False, True): "many_only",
                (True, False): "single_only",
            }[a, b]
        ] += 1
    difference = (paired["many_only"] - paired["single_only"]) / 512
    geometry = defaultdict(list)
    for r in rows(folder / "geometry.jsonl"):
        geometry[f"style_{r['style']}/{r['kind']}"].append(r["span_energy_fraction"])
    assignments = defaultdict(list)
    for r in rows(folder / "assignments.jsonl"):
        assignments["/".join((r["arm"], r["endpoint"], r["panel"]))].append(r)
    assignment_metrics = {}
    for name, group in assignments.items():
        assignment_metrics[name] = {
            "policy": ratio(
                sum(r["gold_class"] == r["predicted_class"] for r in group), len(group)
            ),
            "topic": ratio(
                sum(r["gold_class"] // 2 == r["predicted_class"] // 2 for r in group),
                len(group),
            ),
            "mean_true_weight": sum(r["true_weight"] for r in group) / len(group),
        }
    return {
        "terminal": end,
        "metrics": full,
        "by_fresh_style": by_style,
        "by_topic": by_topic,
        "opposed_margin_flips": swaps,
        "assignments": assignment_metrics,
        "screens": screens,
        "matched_style_contrast": paired
        | {"delta_native": difference, "ten_point_effect": difference >= 0.1},
        "old_span": {
            k: {"mean": sum(v) / len(v), "min": min(v), "max": max(v), "n": len(v)}
            for k, v in geometry.items()
        },
    }


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "run"
    print(json.dumps(summary(folder), indent=2, sort_keys=True))
