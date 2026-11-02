"""Authored finite policies and two supported representation changes.

Train/dev construction grammars and lexical bindings are public. The loader has
no qualification path. Natural-language usefulness still needs human review.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

FRAMES = ("self", "other", "impersonal")
OPERATIONS = {
    "cases": "Make every condition case explicit, preserving the current policy and its subject.",
    "reintegrate": "Reintegrate the cases as one current decision rule, preserving every prescribed action.",
}
TRAIN_FAMILIES = ("both_if", "either_if", "unless", "exclusive", "neither", "first_flag", "second_flag", "never_both")
DEV_FAMILIES = ("requirement", "override")
FAMILY_SPLITS = {**dict.fromkeys(TRAIN_FAMILIES, "train"), **dict.fromkeys(DEV_FAMILIES, "dev")}
# Bits enumerate (off,off), (off,on), (on,off), (on,on). These are authored
# independently of the AST evaluator and the rendered case/normal-form targets.
TRUTH_BITS = {"both_if": (0, 0, 0, 1), "either_if": (0, 1, 1, 1),
              "unless": (0, 0, 1, 0), "exclusive": (0, 1, 1, 0),
              "neither": (1, 0, 0, 0), "first_flag": (0, 0, 1, 1),
              "second_flag": (0, 1, 0, 1), "never_both": (1, 1, 1, 0),
              "requirement": (0, 0, 0, 1), "override": (0, 0, 1, 0)}
TRAIN_FLAGS = (("signal", "permit"), ("alarm", "check"), ("review", "rollback"))
DEV_FLAGS = (("visitor_mode", "request"), ("approval", "override"))
TRAIN_ACTIONS = (("the red bin", "the blue bin"), ("the left shelf", "the right shelf"),
                 ("the north gate", "the south gate"), ("the green folder", "the yellow folder"))
DEV_ACTIONS = (("the copper tray", "the silver tray"), ("the small box", "the large box"))
TRAIN_SITES = ("amber", "birch", "cedar", "coral", "elm", "fern", "hazel", "ivory", "jade", "linen", "maple", "olive")
DEV_SITES = ("acorn", "beacon", "canyon", "dune", "estuary", "flint", "grove", "harbor", "inlet", "juniper", "lagoon", "meadow")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def evaluate_condition(node, world):
    op = node["op"]
    if op == "flag":
        return bool(world[node["name"]])
    if op == "not":
        return not evaluate_condition(node["child"], world)
    if op == "and":
        return all(evaluate_condition(child, world) for child in node["children"])
    if op == "or":
        return any(evaluate_condition(child, world) for child in node["children"])
    if op == "xor":
        return sum(evaluate_condition(child, world) for child in node["children"]) == 1
    raise ValueError("Unknown finite condition")


def rule_ast(family, flags):
    x, y = ({"op": "flag", "name": name} for name in flags)
    both = {"op": "and", "children": [x, y]}
    rules = {
        "both_if": both,
        "either_if": {"op": "or", "children": [x, y]},
        "unless": {"op": "and", "children": [x, {"op": "not", "child": y}]},
        "exclusive": {"op": "xor", "children": [x, y]},
        "neither": {"op": "not", "child": {"op": "or", "children": [x, y]}},
        "first_flag": x,
        "second_flag": y,
        "never_both": {"op": "not", "child": both},
        "requirement": both,
        "override": {"op": "and", "children": [x, {"op": "not", "child": y}]},
    }
    return rules[family]


def frame_parts(frame, site):
    if frame == "self":
        return f"At the {site} station, my current policy is as follows.", "I choose"
    if frame == "other":
        return f"At the {site} station, Agent A has this current policy.", "Agent A chooses"
    if frame == "impersonal":
        return f"At the {site} station, the current selection policy is as follows.", "the policy selects"
    raise ValueError("Unknown attribution frame")


def condition_clauses(family, flags):
    x, y = flags
    return {
        "both_if": (f"If {x} and {y} are both on", f"If {x} is off or {y} is off"),
        "either_if": (f"If either {x} or {y} is on", f"If {x} and {y} are both off"),
        "unless": (f"When {x} is on, unless {y} is on", f"When {x} is off or {y} is on"),
        "exclusive": (f"If exactly one of {x} and {y} is on", f"If {x} and {y} have the same on/off setting"),
        "neither": (f"If neither {x} nor {y} is on", f"If {x} or {y} is on"),
        "first_flag": (f"Whenever {x} is on, regardless of {y}", f"Whenever {x} is off, regardless of {y}"),
        "second_flag": (f"Whenever {y} is on, regardless of {x}", f"Whenever {y} is off, regardless of {x}"),
        "never_both": (f"Unless {x} and {y} are both on", f"When {x} and {y} are both on"),
        "requirement": (f"When the two required checks, {x} and {y}, have both reached on", f"In any case with an off check among {x} and {y}"),
        "override": (f"When the active {x} request survives the {y} override, meaning {x} is on and {y} is off", f"When {y} overrides by being on, or when {x} is off"),
    }[family]


def worlds(flags):
    return [dict(zip(flags, bits)) for bits in itertools.product((False, True), repeat=2)]


def world_text(world):
    return " and ".join(f"{name} is {'on' if value else 'off'}" for name, value in world.items())


def render_cases(prefix, subject, ledger, order):
    entries = ledger if order == 0 else ledger[::-1]
    return prefix + " The cases are mutually exclusive and exhaustive. " + " ".join(
        f"When {world_text(entry['world'])}, {subject} {entry['choice']}." for entry in entries)


def normal_form(ledger, action):
    terms = []
    for entry in ledger:
        if entry["choice"] == action:
            terms.append({"op": "and", "children": [
                {"op": "flag", "name": name} if value else {"op": "not", "child": {"op": "flag", "name": name}}
                for name, value in entry["world"].items()]})
    return {"op": "or", "children": terms}


def render_reintegrated(prefix, subject, ledger, actions):
    positive = ["(" + world_text(entry["world"]) + ")" for entry in ledger if entry["choice"] == actions[0]]
    criterion = " or ".join(positive)
    return f"{prefix} {subject} {actions[0]} exactly when {criterion}. In every other case, {subject} {actions[1]}."


def make_group(family, index):
    split = FAMILY_SPLITS[family]
    if split == "train":
        actions = TRAIN_ACTIONS[index // 3]
        flags = TRAIN_FLAGS[index % 3]
        site = TRAIN_SITES[index]
    else:
        actions = DEV_ACTIONS[index // 6]
        flags = DEV_FLAGS[(index // 3) % 2]
        site = DEV_SITES[index]
    rule = rule_ast(family, flags)
    conditions = condition_clauses(family, flags)
    enumerated = worlds(flags)
    if tuple(int(evaluate_condition(rule, world)) for world in enumerated) != TRUTH_BITS[family]:
        raise ValueError("AST disagrees with independently authored truth bits")
    group_id = f"{family}-{index:02d}"
    no_op = index % 4 == 0
    poles = {}
    for pole in ("plus", "minus"):
        choices = actions if pole == "plus" else actions[::-1]
        ledger = [{"world": world, "choice": choices[0 if truth else 1]} for world, truth in zip(enumerated, TRUTH_BITS[family], strict=True)]
        normalized = normal_form(ledger, actions[0])
        if any((actions[0] if evaluate_condition(normalized, entry["world"]) else actions[1]) != entry["choice"] for entry in ledger):
            raise ValueError("Normal form altered an action")
        variants = []
        for frame in FRAMES:
            prefix, subject = frame_parts(frame, site)
            clauses = [f"{condition}, {subject} {action}." for condition, action in zip(conditions, choices, strict=True)]
            for order in (0, 1):
                stage1 = render_cases(prefix, subject, ledger, order)
                stage2 = render_reintegrated(prefix, subject, ledger, actions)
                source = stage1 if no_op else prefix + " " + " ".join(clauses[::1 if order == 0 else -1])
                variants.append({"id": f"{group_id}-{pole}-{frame}-o{order}", "frame": frame, "order": order,
                                 "source": source, "stage1": stage1, "stage2": stage2,
                                 "no_op_stage1": no_op, "initial_representation": "explicit_cases" if no_op else "conditional_clauses"})
        poles[pole] = {"truth_ledger": ledger, "stage2_condition_ast": normalized, "variants": variants}
    return {"id": group_id, "lineage_group": group_id, "split": split, "family": family,
            "flags": list(flags), "actions": list(actions), "site": site, "rule_ast": rule,
            "truth_bits": list(TRUTH_BITS[family]), "poles": poles,
            "formal_equivalence": "exhaustive_four_worlds", "human_transition_review": "pending"}


def public_groups():
    for family in (*TRAIN_FAMILIES, *DEV_FAMILIES):
        for index in range(12):
            yield make_group(family, index)


def split_audit(groups):
    audit = {}
    for split in ("train", "dev"):
        selected = [g for g in groups if g["split"] == split]
        signatures = Counter("".join(str(bit) for bit in g["truth_bits"]) for g in selected)
        audit[split] = {"groups": len(selected), "variants": len(selected) * 12,
                        "construction_families": sorted({g["family"] for g in selected}),
                        "flags": sorted({name for g in selected for name in g["flags"]}),
                        "actions": sorted({action for g in selected for action in g["actions"]}),
                        "base_truth_signatures": dict(sorted(signatures.items())),
                        "explicit_case_noop_groups": sum(g["poles"]["plus"]["variants"][0]["no_op_stage1"] for g in selected)}
    train = set(audit["train"]["base_truth_signatures"])
    dev = set(audit["dev"]["base_truth_signatures"])
    audit["overlap"] = {"base_truth_signatures": sorted(train & dev),
                         "source_grammar_exception": "explicit-case no-op sources share a canonical table grammar across splits",
                         "claim": "held-out authored conditional grammar and lexical bindings; shared Boolean semantics",
                         "two_stage_human_usefulness": "pending"}
    return audit


def write_public(out):
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=False)
    groups = list(public_groups())
    manifest = {"schema": "finite_two_stage_continuity_groups/v1", "status": "PUBLIC_DEVELOPMENT_MATERIAL",
                "operations": OPERATIONS, "human_transition_review": "pending", "files": {},
                "qualification": {"status": "new_two_stage_panel_not_authored", "content_in_checkout": False},
                "audit": split_audit(groups)}
    for split in ("train", "dev"):
        value = "".join(canonical(g) + "\n" for g in groups if g["split"] == split).encode()
        name = split + ".groups.jsonl"
        (folder / name).write_bytes(value)
        manifest["files"][split] = {"path": name, "sha256": sha256(value), **manifest["audit"][split]}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_groups(folder, split):
    if split not in {"train", "dev"}:
        raise ValueError("Only the public train and dev splits can be loaded")
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    entry = manifest["files"][split]
    if entry["path"] != split + ".groups.jsonl":
        raise ValueError("Unexpected split path")
    value = (folder / entry["path"]).read_bytes()
    if sha256(value) != entry["sha256"]:
        raise ValueError("Public continuity data changed")
    groups = [json.loads(line) for line in value.decode().splitlines()]
    if len(groups) != entry["groups"] or len({g["id"] for g in groups}) != len(groups):
        raise ValueError("Duplicate or missing groups")
    if any(g["split"] != split or FAMILY_SPLITS.get(g["family"]) != split for g in groups):
        raise ValueError("Cross-split construction")
    return groups


def examples(folder, split, stage="source"):
    if stage not in {"source", "stage1", "stage2"}:
        raise ValueError("Unknown representation stage")
    rows = []
    for group in load_groups(folder, split):
        for pole, detail in group["poles"].items():
            for variant in detail["variants"]:
                rows.append({**variant, "source": variant[stage], "text": variant[stage], "input_stage": stage,
                             "group": group["id"], "family": group["family"], "pole": pole,
                             "original_source": variant["source"], "variant_id": variant["id"]})
    return rows


def question_cases(group, pole=None):
    """Recipient-only inputs. Never pass these records to donor capture."""
    if pole is None:
        return [case for current in ("plus", "minus") for case in question_cases(group, current)]
    output = []
    for world_index, entry in enumerate(group["poles"][pole]["truth_ledger"]):
        names = list(entry["world"])
        states = {k: "on" if v else "off" for k, v in entry["world"].items()}
        stems = (
            f"Under the current policy, {world_text(entry['world'])}. Which item is selected?",
            f"For this decision, the states are {names[1]}={states[names[1]]} and {names[0]}={states[names[0]]}. Choose the prescribed item.",
        )
        for wording, stem in enumerate(stems):
            for order in (0, 1):
                first, second = group["actions"][::1 if order == 0 else -1]
                output.append({"group": group["id"], "pole": pole, "world_index": world_index,
                               "world": entry["world"], "wording": wording, "order": order,
                               "question": f"{stem}\nA = {first}. B = {second}.", "labels": ["A", "B"],
                               "answer": "A" if entry["choice"] == first else "B",
                               "gold": "A" if entry["choice"] == first else "B", "action": entry["choice"]})
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_public(args.out), sort_keys=True))
