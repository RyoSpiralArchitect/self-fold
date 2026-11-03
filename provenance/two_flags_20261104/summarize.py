"""Verify the sealed answer ledger and score native choices and one-bit interventions."""

import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from selffold.conditional_curriculum import STAGES, make_cases, tasks_for
from selffold.two_flags import (
    ALL_NEW,
    ARM_NAMES,
    PANELS,
    branch_probe,
    panel_fixture,
    world_probe,
)

spec = importlib.util.spec_from_file_location(
    "native_panels", ROOT / "provenance/conditional_margin_20261103/summarize.py"
)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha_text(s):
    return hashlib.sha256(
        json.dumps(
            s, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def indexed(records, fields):
    data = {tuple(row[k] for k in fields): row for row in records}
    if len(data) != len(records):
        raise ValueError("Duplicate observation")
    return data


def checked(index, key, task, labels):
    row = index.pop(key)
    label = labels.get(row["greedy_token"])
    if (
        row["world"] != task["world"]
        or row["gold"] != task["gold"]
        or row["greedy_label"] != label
        or row["source_sha256"] != sha_text(task["source"])
        or row["question_sha256"] != sha_text(task["question"])
        or not all(math.isfinite(row[k]) for k in ("A", "B"))
    ):
        raise ValueError(
            "Native answer does not bind its rendered input, score or truth ledger"
        )
    return row


def rate(bits):
    return [sum(bits), len(bits)]


def compact(data):
    result = prior.measure(data)
    return {
        k: result[k]
        for k in ("native", "strict", "coverage", "forced", "ties", "per_world")
    }


def decode_branch(label, order):
    if label not in ("A", "B"):
        return None
    return label == ("A" if order == 0 else "B")


def mediated(cases, world, branches, fixture, groups):
    total = defaultdict(lambda: defaultdict(list))
    strict = defaultdict(lambda: defaultdict(list))
    for c in cases:
        stage = c["stage"]
        unit = (c["group"], c["flag_pair"], c["style"])
        opposed = c["key"].rsplit("/", 2)[0] + f"/{1-c['pole']}/{c['source_order']}"
        for w in range(4):
            truth = all(fixture["worlds"][w])
            tasks = tasks_for(c, groups[c["group"]], fixture, w)
            for read_order in (0, 1):
                bit = decode_branch(
                    world[c["flag_pair"], "and", 0, w, read_order]["greedy_label"],
                    read_order,
                )
                for task in tasks:
                    order, label = task["label_order"], task["gold"]
                    opposite_label = "B" if label == "A" else "A"
                    oracle = branches[c["key"], truth, order]["greedy_label"] == label
                    match = flip = swap = False
                    if bit is not None:
                        match = branches[c["key"], bit, order]["greedy_label"] == label
                        flip = (
                            branches[c["key"], not bit, order]["greedy_label"]
                            == opposite_label
                        )
                        swap = (
                            branches[opposed, bit, order]["greedy_label"]
                            == opposite_label
                        )
                    for name, result in [
                        ("native", match),
                        ("gold_bit", oracle),
                        ("correct_bit_exchange", match and flip),
                        ("correct_policy_exchange", match and swap),
                        ("invalid_first_label", bit is None),
                    ]:
                        total[stage][name].append(result)
                    strict[stage][unit].append(match)
    results = {}
    for s in STAGES:
        expected = (256 if s == "rule" else 512) * 2
        if (
            any(len(values) != expected for values in total[s].values())
            or len(strict[s]) != 16
        ):
            raise ValueError("Incomplete mediated panel")
        results[s] = {k: rate(v) for k, v in total[s].items()} | {
            "strict": rate([all(v) for v in strict[s].values()])
        }
    return results


def summary(run):
    design, base, groups_list = (
        read(ROOT / "data" / n)
        for n in (
            "two_flags_20261104.json",
            "conditional_margin_20261103.json",
            "within_topic_wording_20261103.json",
        )
    )
    groups = {g["id"]: g for g in groups_list["groups"]}
    labels = {v: k for k, v in read(run / "labels.json").items()}
    if sorted(labels.values()) != ["A", "B"]:
        raise ValueError("Native label IDs required")
    choices = indexed(
        rows(run / "choices.jsonl"), ("arm", "panel", "key", "world", "label_order")
    )
    worlds = indexed(
        rows(run / "world_answers.jsonl"),
        ("arm", "panel", "flag_pair", "kind", "flag_index", "world", "label_order"),
    )
    bindings = indexed(
        rows(run / "branch_choices.jsonl"), ("arm", "key", "bit", "label_order")
    )
    direct, components, composition, paired, gates = {}, {}, {}, {}, {}
    for arm in ARM_NAMES:
        direct[arm], components[arm], paired[arm] = {}, {}, {}
        for panel in ("fit", *PANELS):
            fixture = base if panel == "fit" else panel_fixture(base, design, panel)
            split = "fit" if panel == "fit" else "fresh"
            cases = list(make_cases(list(groups.values()), fixture, split))
            stages = defaultdict(list)
            branch_rows, world_rows = {}, {}
            words = design["fit_probes" if panel == "fit" else "fresh_probes"]
            for c in cases:
                for w in range(4):
                    for t in tasks_for(c, groups[c["group"]], fixture, w):
                        k = (c["key"], w, t["label_order"])
                        row = checked(choices, (arm, panel, *k), t, labels)
                        if any(
                            row[f] != c[f]
                            for f in (
                                "stage",
                                "group",
                                "style",
                                "source_order",
                                "pole",
                                "flag_pair",
                            )
                        ):
                            raise ValueError("Wrong grouping fields")
                        stages[c["stage"]].append(row)
                        if panel == ALL_NEW:
                            paired[arm][k] = (
                                c["group"],
                                row["greedy_label"] == t["gold"],
                            )
                if panel == ALL_NEW:
                    for bit in (False, True):
                        for t in branch_probe(
                            c, groups[c["group"]], fixture, words, bit
                        ):
                            k = (c["key"], bit, t["label_order"])
                            branch_rows[k] = checked(bindings, (arm, *k), t, labels)
            direct[arm][panel] = {s: compact(stages[s]) for s in STAGES}
            if panel in ("fit", ALL_NEW):
                probes = defaultdict(list)
                for pair, flags in enumerate(fixture[split]["flag_pairs"]):
                    for kind, index in [("flag", 0), ("flag", 1), ("and", 0)]:
                        for w in range(4):
                            for t in world_probe(flags, w, fixture, words, kind, index):
                                k = (pair, kind, index, w, t["label_order"])
                                row = checked(worlds, (arm, panel, *k), t, labels)
                                world_rows[k] = row
                                probes[f"{kind}_{index}"].append(row)
                components[arm][panel] = {
                    k: {
                        "native": rate([r["greedy_label"] == r["gold"] for r in vals]),
                        "by_order": [
                            rate(
                                [
                                    r["greedy_label"] == r["gold"]
                                    for r in vals
                                    if r["label_order"] == o
                                ]
                            )
                            for o in (0, 1)
                        ],
                    }
                    for k, vals in probes.items()
                }
            if panel == ALL_NEW:
                composition[arm] = mediated(
                    cases, world_rows, branch_rows, fixture, groups
                )
        above = prior.passes
        gates[arm] = {
            "literal_fit": all(
                above(m["native"], 95) and above(m["strict"], 90)
                for m in direct[arm]["fit"].values()
            ),
            "direct_fresh": all(
                above(m["native"], 90) and above(m["strict"], 75)
                for m in direct[arm][ALL_NEW].values()
            ),
            "world_components": all(
                above(r, 90)
                for m in components[arm][ALL_NEW].values()
                for r in m["by_order"]
            ),
            "bit_path_stages": {
                s: all(
                    above(m[k], n)
                    for k, n in [
                        ("native", 90),
                        ("strict", 75),
                        ("correct_bit_exchange", 90),
                        ("correct_policy_exchange", 90),
                    ]
                )
                for s, m in composition[arm].items()
            },
        }
        gates[arm]["typed_interface"] = gates[arm]["world_components"] and all(
            gates[arm]["bit_path_stages"].values()
        )
    if (
        choices
        or bindings
        or worlds
        or paired["direct"].keys() != paired["parts"].keys()
    ):
        raise ValueError("Extra records or unpaired panels")
    delta = Counter({g: 0 for g in groups})
    recovered = lost = 0
    for k, (g, baseline) in paired["direct"].items():
        treated = paired["parts"][k][1]
        recovered += int(treated and not baseline)
        lost += int(baseline and not treated)
        delta[g] += int(treated) - int(baseline)
    strict_gains = sum(
        direct["parts"][ALL_NEW][s]["strict"][0]
        - direct["direct"][ALL_NEW][s]["strict"][0]
        for s in STAGES
    )
    n = len(paired["direct"])
    comparison = dict(
        recovered=recovered,
        lost=lost,
        denominator=n,
        net=recovered - lost,
        topics=dict(delta),
        net_strict=strict_gains,
        benefit=100 * (recovered - lost) >= 5 * n
        and sum(v > 0 for v in delta.values()) >= 5
        and strict_gains >= 0,
    )
    trains = indexed(rows(run / "training.jsonl"), ("arm", "step"))
    schedule = read(run / "schedule.json")
    if len(schedule) != 384 or len(trains) != 768:
        raise ValueError("Incomplete schedule")
    costs = {a: Counter() for a in ARM_NAMES}
    for step, unit in enumerate(schedule, 1):
        a, b = (trains[arm, step] for arm in ARM_NAMES)
        if (
            a["units"] != unit
            or b["units"] != unit
            or a["padded_tokens"] != b["padded_tokens"]
            or a["forwards"] != 8
            or b["forwards"] != 8
            or a["counts"] != {"action": 32}
            or b["counts"] != {k: 8 for k in ("action", "flag", "and", "bind")}
        ):
            raise ValueError("Treatment and control budgets or training tasks differ")
        for row in (a, b):
            for name in ("padded_tokens", "unmasked_tokens", "forwards"):
                costs[row["arm"]][name] += row[name]
    return dict(
        direct=direct,
        components=components,
        composition=composition,
        comparison=comparison,
        gates=gates,
        training_cost=costs,
    )


if __name__ == "__main__":
    print(
        json.dumps(
            summary(Path(sys.argv[1]) if len(sys.argv) > 1 else STUDY / "run"),
            indent=2,
            sort_keys=True,
        )
    )
