"""Reduce the fixed development comparison from its per-case receipts (stdlib only)."""
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import random
from statistics import fmean

HERE = Path(__file__).resolve().parent
ARMS = ("copy", "opposition")
ROUTES = ("matched", "literal", "empty", "zero", "random", "unrelated")
FAMILIES = ("scope", "priority")
GROUPS = [f"{f}-{i:02d}" for f in FAMILIES for i in range(12)]


def rows(folder, name):
    return [json.loads(line) for line in (folder / name).read_text().splitlines()]


def prediction(row):
    return "A" if row["A"] > row["B"] else ("B" if row["B"] > row["A"] else None)


def group_metrics(items, worlds):
    lookup = {(x["pole"], x["wording"], x["world"], x["order"]): x for x in items}
    if len(lookup) != len(items) or len(items) != 8 * worlds:
        raise ValueError("Duplicate or missing functional cells")
    correct = {key: prediction(v) == v["gold"] for key, v in lookup.items()}
    opposed, wording, world_switch = [], [], []
    for (p, v, w, o), ok in correct.items():
        if p == "plus":
            other = ("minus", v, w, o)
            if lookup[other]["gold"] == lookup[p, v, w, o]["gold"]:
                raise ValueError("Expected opposed gold choices")
            opposed.append(ok and correct[other])
        if v == "p0":
            other = (p, "p1", w, o)
            if lookup[other]["gold"] != lookup[p, v, w, o]["gold"]:
                raise ValueError("Expected same-meaning variants")
            wording.append(ok and correct[other])
    for p in ("plus", "minus"):
        for v in ("p0", "p1"):
            for o in (0, 1):
                for a, b in combinations(range(worlds), 2):
                    keys = [(p, v, w, o) for w in (a, b)]
                    if lookup[keys[0]]["gold"] != lookup[keys[1]]["gold"]:
                        world_switch.append(all(correct[k] for k in keys))
    evidence = {"accuracy": list(correct.values()), "opposed": opposed, "wrappers": wording, "world_switch": world_switch}
    return {metric: (sum(values), len(values)) for metric, values in evidence.items()}


def aggregate(grouped, groups):
    result = {}
    for metric in grouped[groups[0]]:
        pairs = [grouped[g][metric] for g in groups]
        result[metric] = {"correct": sum(a for a, _ in pairs), "n": sum(n for _, n in pairs),
                          "macro": fmean(a / n for a, n in pairs)}
    return result


def strata(grouped):
    return {name: aggregate(grouped, groups) for name, groups in
            [("all", GROUPS), *((f, [g for g in GROUPS if g.startswith(f+"-")]) for f in FAMILIES)]}


def summarize(folder):
    term = json.loads((folder / "terminal.json").read_text())
    if term["status"] != "COMPLETED_PAIRED_DEVELOPMENT_COMPARISON":
        raise ValueError("Incomplete experiment")
    functional = defaultdict(list)
    for x in rows(folder, "functional.jsonl"):
        functional[x["arm"], x["route"], x["group"]].append(x)
    if set(functional) != {(a, r, g) for a in ARMS for r in ROUTES for g in GROUPS}:
        raise ValueError("Missing or unexpected endpoints")
    results, differences, lexical, probe = {}, {}, {}, {}
    for arm in ARMS:
        results[arm] = {}
        for route in ROUTES:
            per_group = {g: group_metrics(functional[arm, route, g], 2 if g.startswith("scope-") else 4) for g in GROUPS}
            results[arm][route] = strata(per_group)
            for cell in ("all", *FAMILIES):
                selected = GROUPS if cell == "all" else [g for g in GROUPS if g.startswith(cell+"-")]
                results[arm][route][cell]["ties"] = sum(prediction(x) is None for g in selected
                                                        for x in functional[arm, route, g])
            if route == "matched":
                differences[arm] = {g: a / n for g in GROUPS for a, n in [per_group[g]["accuracy"]]}
        by_case = {x["case"]: x for x in rows(folder, "prefix.jsonl") if x["arm"] == arm}
        expected = {f"{g}-{p}-p{i}" for g in GROUPS for p in ("plus", "minus") for i in range(2)}
        if set(by_case) != expected or len([x for x in rows(folder, "prefix.jsonl") if x["arm"] == arm]) != 96:
            raise ValueError("Missing or duplicate prefix endpoints")
        lexical[arm] = {}
        for cell in ("all", *FAMILIES):
            groups = GROUPS if cell == "all" else [g for g in GROUPS if g.startswith(cell+"-")]
            pairs = [[by_case[f"{g}-{p}-p{i}"] for p in ("plus", "minus")] for g in groups for i in range(2)]
            members = [x for pair in pairs for x in pair]
            if any(a["prefix_sha256"] != b["prefix_sha256"] or a["target_token"] != b["foil_token"]
                   or b["target_token"] != a["foil_token"] for a, b in pairs):
                raise ValueError("Misaligned prefix contrast")
            lexical[arm][cell] = {"strict_pairs": sum(all(x["margin"] > 0 for x in pair) for pair in pairs),
                "pairs": len(pairs), "correct": sum(x["margin"] > 0 for x in members), "n": len(members),
                "ties": sum(x["margin"] == 0 for x in members), "mean_nll": fmean(x["source_nll"] for x in members),
                "mean_signed_margin": fmean(x["margin"] for x in members),
                "max_replay_error": max(abs(x["margin"] - x["masked_suffix_replay_margin"]) for x in members)}
    raw = rows(folder, "raw_readout.jsonl")
    for sham in (False, True):
        cases = {x["case"]: x for x in raw if x["sham"] == sham}
        if set(cases) != expected or sum(x["sham"] == sham for x in raw) != 96:
            raise ValueError("Missing or duplicate raw probe decisions")
        strata_raw = {}
        for cell in ("all", *FAMILIES):
            groups = GROUPS if cell == "all" else [g for g in GROUPS if g.startswith(cell+"-")]
            pairs = [[cases[f"{g}-{p}-p{i}"] for p in ("plus", "minus")] for g in groups for i in range(2)]
            ordered = 0
            for pair in pairs:
                by_word = {x["word"]: x["copper_margin"] for x in pair}
                ordered += by_word["copper"] > by_word["silver"]
            examples = [x for pair in pairs for x in pair]
            strata_raw[cell] = {"correct": sum((1 if x["word"] == "copper" else -1) * x["copper_margin"] > 0 for x in examples),
                                "n": len(examples), "ordered_pairs": ordered, "pairs": len(pairs)}
        probe["sham" if sham else "train_labels"] = strata_raw
    delta = {g: differences["opposition"][g] - differences["copy"][g] for g in GROUPS}
    rng = random.Random(47)
    values = {f: [d for g, d in delta.items() if g.startswith(f+"-")] for f in FAMILIES}
    sampled = sorted(fmean(x for f in FAMILIES for x in rng.choices(values[f], k=12)) for _ in range(10000))
    compare = {"difference": fmean(delta.values()), "families": {k: fmean(v) for k, v in values.items()},
               "paired_site_interval_95": [sampled[249], sampled[9749]]}
    screens = {}
    for arm in ARMS:
        m = results[arm]["matched"]["all"]
        null = max(results[arm][r]["all"]["accuracy"]["macro"] for r in ("empty", "zero", "random", "unrelated"))
        screens[arm] = {"strongest_null": null, "functional": (m["accuracy"]["macro"] >= .8 and
                m["accuracy"]["macro"] - null >= .1 and m["opposed"]["macro"] >= .8 and m["wrappers"]["macro"] >= .9),
                "lexical": lexical[arm]["all"]["strict_pairs"] / lexical[arm]["all"]["pairs"] >= .8}
    return {"functional": results, "lexical": lexical, "raw_probe": probe, "comparison": compare, "screens": screens}


if __name__ == "__main__":
    import sys
    print(json.dumps(summarize(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "run"), indent=2, sort_keys=True))
