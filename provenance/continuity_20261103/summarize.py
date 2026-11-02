"""Reduce complete teacher-encoded recipient panels from native and A/B scores.

This endpoint qualifies the readout of independently encoded source and teacher
texts. It supplies no score for learned latent G composition.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import fmean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGES = ("source", "stage1", "stage2")
ROUTES = ("matched", "empty", "zero", "random", "unrelated", "opposed")
NULLS = ("empty", "zero", "random", "unrelated")
FRAMES = ("self", "other", "impersonal")


def _material():
    path = ROOT / "data/continuity_20261103/dev.groups.jsonl"
    groups = [json.loads(line) for line in path.read_text().splitlines()]
    if len(groups) != 24 or len({g["id"] for g in groups}) != 24:
        raise ValueError("Need exactly 24 development groups")
    cases = {}
    expected = set()
    for group in groups:
        for pole, details in group["poles"].items():
            for variant in details["variants"]:
                key = (group["id"], pole, variant["frame"], variant["order"])
                if key in cases:
                    raise ValueError("Duplicate source variant")
                cases[key] = variant["id"]
                for world_index in range(4):
                    for wording in (0, 1):
                        for order in (0, 1):
                            expected.add((*key, world_index, wording, order))
    if len(expected) != 4608:
        raise ValueError("Unexpected recipient denominator")
    return {g["id"]: g for g in groups}, cases, expected


def _world_index(value, ledger):
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(ledger):
        return value
    if isinstance(value, dict):
        matches = [i for i, entry in enumerate(ledger) if value == entry["world"]]
        if len(matches) == 1:
            return matches[0]
    raise ValueError("Unknown development world")


def _metrics(lookup, group_ids, frames):
    subset = {key: row for key, row in lookup.items() if key[0] in group_ids and key[2] in frames}
    per_group = defaultdict(list)
    compound = defaultdict(list)
    for key, row in subset.items():
        group, pole, frame, source_order, world, wording, label_order = key
        forced = "A" if row["A"] > row["B"] else ("B" if row["B"] > row["A"] else None)
        item = {"native": row["greedy_label"] == row["gold"], "forced": forced == row["gold"],
                "coverage": row["greedy_label"] in ("A", "B"), "tie": forced is None}
        per_group[group].append(item)
        compound[group, frame, world, wording, label_order].append(item)
    if set(per_group) != set(group_ids) or any(len(values) != 4 for values in compound.values()):
        raise ValueError("Incomplete policy/order composition")
    group_compound = defaultdict(list)
    for key, values in compound.items():
        group_compound[key[0]].append({"native": all(x["native"] for x in values),
                                       "forced": all(x["forced"] for x in values)})
    all_items = [x for values in per_group.values() for x in values]
    all_compound = [x for values in group_compound.values() for x in values]
    return {
        "n": len(all_items), "native_correct": sum(x["native"] for x in all_items),
        "forced_correct": sum(x["forced"] for x in all_items),
        "native_coverage_count": sum(x["coverage"] for x in all_items),
        "forced_ties": sum(x["tie"] for x in all_items),
        "native_accuracy": fmean(fmean(x["native"] for x in per_group[g]) for g in group_ids),
        "forced_accuracy": fmean(fmean(x["forced"] for x in per_group[g]) for g in group_ids),
        "native_coverage": fmean(fmean(x["coverage"] for x in per_group[g]) for g in group_ids),
        "compound_n": len(all_compound),
        "native_compound_correct": sum(x["native"] for x in all_compound),
        "forced_compound_correct": sum(x["forced"] for x in all_compound),
        "native_compound_accuracy": fmean(fmean(x["native"] for x in group_compound[g]) for g in group_ids),
        "forced_compound_accuracy": fmean(fmean(x["forced"] for x in group_compound[g]) for g in group_ids),
        "greedy_labels": dict(sorted(Counter(str(row["greedy_label"]) for row in subset.values()).items())),
    }


def _reduce(records, candidate, routes):
    groups, cases, expected = _material()
    lookup = defaultdict(dict)
    source_hashes = {}
    observed = 0
    for row in records:
        if row["candidate"] != candidate:
            continue
        stage, route = row["stage"], row["route"]
        if stage not in STAGES or route not in routes:
            raise ValueError("Unexpected candidate route or representation stage")
        group, pole = row["group"], row["pole"]
        if group not in groups or pole not in groups[group]["poles"]:
            raise ValueError("Unknown source group or pole")
        ledger = groups[group]["poles"][pole]["truth_ledger"]
        world = _world_index(row["world"], ledger)
        prefix = (group, pole, row["frame"], row["source_order"])
        key = (*prefix, world, row["wording"], row["label_order"])
        if key not in expected or row["case"] != cases[prefix] or key in lookup[route, stage]:
            raise ValueError("Duplicate or unknown recipient case")
        expected_gold = "A" if ledger[world]["choice"] == groups[group]["actions"][row["label_order"]] else "B"
        if row["gold"] != expected_gold or row["greedy_label"] not in (None, "A", "B"):
            raise ValueError("Gold or native output label disagrees with the contract")
        if not all(isinstance(row[k], (int, float)) and math.isfinite(row[k]) for k in ("A", "B")):
            raise ValueError("Nonfinite label score")
        fingerprint = row["state_sha256"]
        old = source_hashes.setdefault((route, stage, row["case"]), fingerprint)
        if old != fingerprint:
            raise ValueError("Task-dependent source state")
        lookup[route, stage][key] = row
        observed += 1
    if set(lookup) != {(r, s) for r in routes for s in STAGES}:
        raise ValueError("Missing recipient routes or stages")
    if any(set(cell) != expected for cell in lookup.values()):
        raise ValueError("Incomplete recipient panel")
    family_ids = {family: [g for g in groups if groups[g]["family"] == family] for family in sorted({g["family"] for g in groups.values()})}
    cells = {}
    for route in routes:
        cells[route] = {}
        for stage in STAGES:
            scores = lookup[route, stage]
            cells[route][stage] = {"all": _metrics(scores, list(groups), FRAMES),
                "frames": {frame: _metrics(scores, list(groups), (frame,)) for frame in FRAMES},
                "families": {family: _metrics(scores, ids, FRAMES) for family, ids in family_ids.items()}}
    return {"candidate": candidate, "logical_records": observed, "cells": cells}


def score_candidate(records, candidate):
    result = _reduce(records, candidate, ROUTES)
    final = result["cells"]["matched"]["stage2"]["all"]
    strongest = max(result["cells"][route]["stage2"]["all"]["native_accuracy"] for route in NULLS)
    result["recipient_screen"] = {
        "endpoint": "teacher_encoded_stage2_native_recipient_only",
        "native_accuracy": final["native_accuracy"], "strongest_null_native_accuracy": strongest,
        "native_gain_over_null": final["native_accuracy"] - strongest,
        "native_compound_accuracy": final["native_compound_accuracy"],
        "pass": final["native_accuracy"] >= .90 and final["native_accuracy"] - strongest >= .10
                and final["native_compound_accuracy"] >= .75,
    }
    return result


def summarize(folder):
    records = [json.loads(line) for line in (folder / "functional.jsonl").read_text().splitlines()]
    candidates = sorted({row["candidate"] for row in records if row["candidate"] != "base"})
    if not candidates:
        raise ValueError("No recipient candidate")
    output = {"candidates": {name: score_candidate(records, name) for name in candidates}}
    if any(row["candidate"] == "base" for row in records):
        output["literal_base"] = _reduce(records, "base", ("literal_base",))
    terminal = folder / "terminal.json"
    if terminal.exists():
        output["terminal"] = json.loads(terminal.read_text())
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.folder), indent=2, sort_keys=True))
