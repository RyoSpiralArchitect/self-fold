"""Exploratory saved-score diagnostics; never replaces the sealed native gates."""

from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

STUDY = Path(__file__).resolve().parent
DECISIONS_SHA256 = "83cbd32131bdbdfa3d7ee248de99564de59b34a448800a4bb0c0a5cf4bab5205"
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
SETTINGS = {
    "fit": (range(3), ("if_else", "condition_test")),
    "fresh": (range(2), ("joint_switch",)),
}


def ratio(values):
    values = list(values)
    return [sum(values), len(values)]


def native(row):
    return row["greedy_label"] == row["gold"]


def identity(row):
    return row["key"], row["world"], row["label_order"]


def validate(data):
    """Require the entire sealed panel ledger, including non-A/B native answers."""
    panels = defaultdict(dict)
    questions, sources = {}, {}
    for row in data:
        if row["arm"] != "warm" or row["route"] != "literal":
            raise ValueError("Only the closed literal study can be diagnosed")
        panel = row["endpoint"], row["split"]
        if panel not in (("initial", "fit"), ("final", "fit"), ("final", "fresh")):
            raise ValueError("Unexpected endpoint or split")
        parts = row["key"].split("/")
        if len(parts) != 7:
            raise ValueError("Incomplete source key")
        split, topic, flag, style, stage, pole, order = parts
        metadata = dict(
            split=split,
            group=topic,
            flag_pair=int(flag),
            style=style,
            stage=stage,
            pole=int(pole),
            source_order=int(order),
        )
        if any(row[k] != value for k, value in metadata.items()):
            raise ValueError("Metadata differs from its source key")
        for field, limit in (("world", 4), ("label_order", 2)):
            if type(row[field]) is not int or row[field] not in range(limit):
                raise ValueError("Invalid world or label mapping")
        correct_action = row["pole"] if row["world"] == 3 else 1 - row["pole"]
        if row["gold"] != ("A" if correct_action == row["label_order"] else "B"):
            raise ValueError("Gold differs from the fixed conditional action mapping")
        if not all(
            isinstance(row[k], (int, float)) and math.isfinite(row[k])
            for k in ("A", "B")
        ):
            raise ValueError("Nonfinite A/B score")
        if row["greedy_label"] not in ("A", "B", None):
            raise ValueError("Invalid native label")
        key = identity(row)
        if key in panels[panel]:
            raise ValueError("Duplicate native observation")
        panels[panel][key] = row
        qkey = split, topic, int(flag), style, row["world"], row["label_order"]
        if questions.setdefault(qkey, row["question_sha256"]) != row["question_sha256"]:
            raise ValueError(
                "Question changed across source positions, poles or endpoints"
            )
        if sources.setdefault(row["key"], row["source_sha256"]) != row["source_sha256"]:
            raise ValueError("Source changed across worlds, labels or endpoints")
        if row["injected_sha256"] != row["source_sha256"]:
            raise ValueError("Literal route no longer binds the complete source")
    if set(panels) != {("initial", "fit"), ("final", "fit"), ("final", "fresh")}:
        raise ValueError("Missing a complete study panel")
    for (_, split), panel in panels.items():
        flags, styles = SETTINGS[split]
        expected = {
            (f"{split}/{topic}/{flag}/{style}/{stage}/{pole}/{order}", world, label)
            for topic, flag, style, stage, pole in itertools.product(
                TOPICS, flags, styles, STAGES, (0, 1)
            )
            for order in ((0,) if stage == "rule" else (0, 1))
            for world, label in itertools.product(range(4), (0, 1))
        }
        if set(panel) != expected:
            raise ValueError("Incomplete source/world/order/label panel")
    return panels


def paired_order(data):
    """Symmetrize two displayed mappings externally; retain every source/world pair."""
    pairs, units = defaultdict(dict), defaultdict(list)
    for row in data:
        pairs[row["key"], row["world"]][row["label_order"]] = row
    correct, ties, native_pair_counts = [], 0, Counter()
    world_correct = defaultdict(list)
    for records in pairs.values():
        if set(records) != {0, 1}:
            raise ValueError("Missing the other displayed label mapping")
        a, b = records[0], records[1]
        if a["gold"] == b["gold"] or a["source_sha256"] != b["source_sha256"]:
            raise ValueError("Paired mapping changed source or failed to reverse gold")
        if a["question_sha256"] == b["question_sha256"]:
            raise ValueError("Displayed label swap failed to change the question")
        margin = ((a["A"] - a["B"]) - (b["A"] - b["B"])) / 2
        good = margin > 0 if a["gold"] == "A" else margin < 0
        correct.append(good)
        world_correct[a["world"]].append(good)
        ties += margin == 0
        native_pair_counts[str(sum(native(r) for r in records.values()))] += 1
        units[a["group"], a["flag_pair"], a["style"], a["stage"]].append(good)
    stage = data[0]["stage"]
    if any(len(v) != (8 if stage == "rule" else 16) for v in units.values()):
        raise ValueError("Paired strict units omit a policy, world or source order")
    return {
        "forced_correct": ratio(correct),
        "forced_strict": ratio(all(v) for v in units.values()),
        "forced_ties": ties,
        "native_correct": ratio(native(r) for r in data),
        "native_coverage": ratio(r["greedy_label"] in ("A", "B") for r in data),
        "native_ties": sum(r["A"] == r["B"] for r in data),
        "native_correct_mappings_per_pair": {
            str(k): native_pair_counts[str(k)] for k in range(3)
        },
        "per_world": {
            str(world): {
                "native_correct": ratio(native(r) for r in data if r["world"] == world),
                "native_coverage": ratio(
                    r["greedy_label"] in ("A", "B") for r in data if r["world"] == world
                ),
                "forced_correct": ratio(world_correct[world]),
            }
            for world in range(4)
        },
        "native_by_source_order": {
            str(order): ratio(native(r) for r in data if r["source_order"] == order)
            for order in ((0,) if stage == "rule" else (0, 1))
        },
        "excluded_native_non_ab": 0,
    }


def diagnose(data):
    panels = validate(data)
    before, after = panels["initial", "fit"], panels["final", "fit"]
    transitions = {}
    for stage in STAGES:
        worlds = {}
        for world in range(4):
            keys = [
                k
                for k, r in before.items()
                if r["stage"] == stage and r["world"] == world
            ]
            transitions_now = Counter(
                (native(before[k]), native(after[k])) for k in keys
            )
            initial, final = (
                sum(native(before[k]) for k in keys),
                sum(native(after[k]) for k in keys),
            )
            worlds[str(world)] = {
                "initial": [initial, len(keys)],
                "final": [final, len(keys)],
                "net_change": final - initial,
                "improved": transitions_now[False, True],
                "regressed": transitions_now[True, False],
                "still_correct": transitions_now[True, True],
                "still_wrong": transitions_now[False, False],
            }
        transitions[stage] = worlds
    errors = [r for r in after.values() if not native(r)]
    by_stage = {}
    for stage in STAGES:
        bad = [r for r in errors if r["stage"] == stage]
        all_on = [r for r in bad if r["world"] == 3]
        by_stage[stage] = {
            "errors": len(bad),
            "gold_A_predicted_B": sum(
                r["gold"] == "A" and r["greedy_label"] == "B" for r in bad
            ),
            "all_on_errors_by_source_order": {
                str(order): sum(r["source_order"] == order for r in all_on)
                for order in (0, 1)
            },
        }
    paired = {
        split: {
            stage: paired_order(
                [r for r in panels["final", split].values() if r["stage"] == stage]
            )
            for stage in STAGES
        }
        for split in ("fit", "fresh")
    }
    return {
        "classification": "exploratory_posthoc_external_forced_two_forward_readout",
        "native_gates_changed": False,
        "native_admission_established": False,
        "selection": "All sealed observations retained, including native non-A/B answers and ties; strict sign ties count incorrect.",
        "paired_formula": "((A-B)_label_order_0 - (A-B)_label_order_1) / 2; positive selects canonical action0; negative selects action1.",
        "denominator_note": "Two saved native forwards yield one external forced action decision; paired decision denominators are half native denominators. Strict units retain both poles, every world and every genuine source order.",
        "fit_error_asymmetry": {
            "errors": len(errors),
            "gold_A_predicted_B": sum(
                r["gold"] == "A" and r["greedy_label"] == "B" for r in errors
            ),
            "by_stage": by_stage,
        },
        "fit_native_initial": ratio(native(r) for r in before.values()),
        "fit_native_final": ratio(native(r) for r in after.values()),
        "fit_world_transitions": transitions,
        "paired_order": paired,
        "limits": [
            "External forced pairing cannot replace the native full-vocabulary admission gate or establish a causal effect of the margin objective.",
            "Fresh source names, wording families and question operations changed together; these scores do not isolate flag conjunction, source binding or question transfer.",
            "No source-learning branch or G was trained, and old/new fresh panels are not paired.",
        ],
    }


def summary(path):
    content = Path(path).read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    if sha != DECISIONS_SHA256:
        raise ValueError("The sealed native decisions changed")
    data = [json.loads(line) for line in content.splitlines()]
    return {"input_decisions_sha256": sha, "native_observations": len(data)} | diagnose(
        data
    )


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else STUDY / "run/decisions.jsonl"
    print(json.dumps(summary(path), indent=2, sort_keys=True))
