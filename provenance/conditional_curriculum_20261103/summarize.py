"""Reduce the sealed literal warmup and two native conditional-reader arms."""

from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import sys

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
STAGES = ("source", "cases", "rule")
TOPICS = (
    "archive",
    "route",
    "notes",
    "temperature",
    "length",
    "clock",
    "mail",
    "music",
)
ROUTES = ("matched", "opposed", "zero", "mean")


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def ratio(values):
    values = list(values)
    return [sum(values), len(values)]


def passes(count, percent):
    return count[1] > 0 and 100 * count[0] >= percent * count[1]


def metadata(row):
    parts = row["key"].split("/")
    if len(parts) != 7:
        raise ValueError(
            "Conditional key must bind split, group, flags, style, stage, pole and order"
        )
    split, group, flag, style, stage, pole, order = parts
    flag, pole, order = int(flag), int(pole), int(order)
    if (
        split not in ("fit", "fresh")
        or group not in TOPICS
        or stage not in STAGES
        or pole not in (0, 1)
    ):
        raise ValueError("Unknown conditional observation")
    if order not in ((0,) if stage == "rule" else (0, 1)):
        raise ValueError("Canonical rules have one genuine source realization")
    result = dict(
        split=split,
        group=group,
        flag_pair=flag,
        style=style,
        stage=stage,
        pole=pole,
        source_order=order,
    )
    for name, value in result.items():
        if name in row and row[name] != value:
            raise ValueError("Decision metadata differs from its bound source key")
    for name in ("world", "label_order"):
        if type(row[name]) is not int or row[name] not in (
            range(4) if name == "world" else (0, 1)
        ):
            raise ValueError("Invalid world or displayed option order")
    action = pole if row["world"] == 3 else 1 - pole
    if row["gold"] != ("A" if action == row["label_order"] else "B"):
        raise ValueError("Gold answer differs from the conjunction truth ledger")
    if not all(
        isinstance(row[k], (int, float)) and math.isfinite(row[k]) for k in ("A", "B")
    ):
        raise ValueError("Nonfinite native label score")
    return result


def correct(row):
    return row["greedy_label"] == row["gold"]


def score_counts(data):
    return {
        "native": ratio(correct(r) for r in data),
        "forced": ratio(
            r[r["gold"]] > r["B" if r["gold"] == "A" else "A"] for r in data
        ),
        "coverage": ratio(r["greedy_label"] in ("A", "B") for r in data),
        "ties": sum(r["A"] == r["B"] for r in data),
    }


def measure(data):
    """Measure complete units, requiring both policies, every world and genuine orders."""
    if not data:
        raise ValueError("Cannot measure an empty conditional panel")
    by_unit = defaultdict(dict)
    seen = set()
    for row in data:
        m = metadata(row)
        observation = row["key"], row["world"], row["label_order"]
        if observation in seen:
            raise ValueError("Duplicate conditional decision")
        seen.add(observation)
        unit = m["group"], m["flag_pair"], m["style"], m["stage"]
        by_unit[unit][
            m["pole"], m["source_order"], row["world"], row["label_order"]
        ] = row
    for unit, records in by_unit.items():
        orders = (0,) if unit[-1] == "rule" else (0, 1)
        expected = set(itertools.product((0, 1), orders, range(4), (0, 1)))
        if set(records) != expected:
            raise ValueError(
                "Strict conditional unit is missing a policy, world, order or label"
            )
    return score_counts(data) | {
        "strict": ratio(
            all(correct(r) for r in records.values()) for records in by_unit.values()
        ),
        "per_world": {
            str(w): score_counts([r for r in data if r["world"] == w]) for w in range(4)
        },
        "per_topic": {
            g: score_counts([r for r in data if metadata(r)["group"] == g])
            for g in TOPICS
            if any(metadata(r)["group"] == g for r in data)
        },
    }


def validate_panel(data, split, stage, fixture):
    settings = fixture[split]
    expected = set(
        itertools.product(
            TOPICS, range(len(settings["flag_pairs"])), settings["styles"]
        )
    )
    actual = {(m["group"], m["flag_pair"], m["style"]) for m in map(metadata, data)}
    total = (
        (768 if stage == "rule" else 1536)
        if split == "fit"
        else (256 if stage == "rule" else 512)
    )
    if actual != expected or len(data) != total:
        raise ValueError(
            "Conditional panel must retain every topic, flag pair and source style"
        )
    if any(
        metadata(r)["split"] != split or metadata(r)["stage"] != stage for r in data
    ):
        raise ValueError("Conditional panel stage or split differs")


def warm_gate(data):
    """Can be called at the training boundary before any slot fitting starts."""
    selected = [
        r
        for r in data
        if (r["arm"], r["endpoint"], r["split"], r["route"])
        == ("warm", "final", "fit", "literal")
    ]
    fixture = json.loads(
        (ROOT / "data/conditional_curriculum_20261103.json").read_text()
    )
    stages = {}
    for stage in STAGES:
        group = [r for r in selected if r["stage"] == stage]
        validate_panel(group, "fit", stage, fixture)
        m = measure(group)
        stages[stage] = passes(m["native"], 95) and passes(m["strict"], 90)
    total = score_counts(selected)
    return {
        "stages": stages,
        "total": passes(total["native"], 95),
        "passed": passes(total["native"], 95) and all(stages.values()),
    }


def flip_count(matched, opposed):
    by_key = {(r["key"], r["world"], r["label_order"]): r for r in matched}
    other = {(r["key"], r["world"], r["label_order"]): r for r in opposed}
    if (
        len(by_key) != len(matched)
        or len(other) != len(opposed)
        or by_key.keys() != other.keys()
    ):
        raise ValueError("Opposed route must have an exact matched counterpart")
    for key, r in other.items():
        ref = by_key[key]
        if (r["question_sha256"], r["gold"]) != (ref["question_sha256"], ref["gold"]):
            raise ValueError("Opposed route changed the question or evaluation gold")
    return ratio(
        (r["A"] - r["B"]) * (other[k]["A"] - other[k]["B"]) < 0
        for k, r in by_key.items()
    )


def summary(folder):
    data = rows(Path(folder) / "decisions.jsonl")
    fixture = json.loads(
        (ROOT / "data/conditional_curriculum_20261103.json").read_text()
    )
    panels = defaultdict(list)
    for row in data:
        metadata(row)
        key = tuple(row[k] for k in ("arm", "endpoint", "split", "stage", "route"))
        panels[key].append(row)
    warm = warm_gate(data)
    required = {
        ("warm", endpoint, split, stage, "literal")
        for endpoint, split in (
            ("initial", "fit"),
            ("final", "fit"),
            ("final", "fresh"),
        )
        for stage in STAGES
    }
    if warm["passed"]:
        required |= {
            (arm, "final", split, stage, route)
            for arm in ("frozen", "joint")
            for stage in STAGES
            for split, routes_ in (("fit", ("matched",)), ("fresh", ROUTES))
            for route in routes_
        }
    if set(panels) != required:
        raise ValueError("Attempted panels differ from the predeclared warmup branch")
    metrics = {}
    for (arm, endpoint, split, stage, route), group in sorted(panels.items()):
        validate_panel(group, split, stage, fixture)
        metrics["/".join((arm, endpoint, split, stage, route))] = measure(group)
    gates = {"warm": warm}
    flips = {}
    for arm in ("frozen", "joint"):
        if not warm["passed"]:
            gates[arm] = {"attempted": False, "admission": False}
            continue
        fitting, transfer = {}, {}
        for stage in STAGES:
            fit = metrics[f"{arm}/final/fit/{stage}/matched"]
            fitting[stage] = passes(fit["native"], 95) and passes(fit["strict"], 90)
            fresh = {
                route: metrics[f"{arm}/final/fresh/{stage}/{route}"] for route in ROUTES
            }
            matched = fresh["matched"]
            numerator, denominator = matched["native"]
            null = max(fresh[route]["native"][0] for route in ("zero", "mean"))
            opposed = fresh["opposed"]["native"][0]
            flip = flip_count(
                panels[arm, "final", "fresh", stage, "matched"],
                panels[arm, "final", "fresh", stage, "opposed"],
            )
            flips[arm + "/" + stage] = flip
            transfer[stage] = (
                passes(matched["native"], 90)
                and passes(matched["strict"], 75)
                and 100 * (numerator - null) >= 10 * denominator
                and 100 * (numerator - opposed) >= 30 * denominator
                and passes(flip, 90)
            )
        gates[arm] = {
            "attempted": True,
            "fit_stages": fitting,
            "fresh_stages": transfer,
            "admission": all(fitting.values()) and all(transfer.values()),
        }
    return {"metrics": metrics, "opposed_margin_flips": flips, "gates": gates}


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else STUDY / "run"
    print(json.dumps(summary(folder), indent=2))
