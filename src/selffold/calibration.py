"""Authored finite policies for codec preparation, with explicit truth ledgers."""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path

OPERATORS = ("plain", "decompose", "boundary", "reintegrate")
FAMILIES = {"conditional": "train", "exception": "train", "conjunction": "train",
            "exclusion": "train", "scope": "dev", "priority": "dev"}
SITES = ("amber", "birch", "cedar", "coral", "elm", "fern", "hazel", "ivory", "jade", "linen", "maple", "olive")


def evaluate(rule, world):
    op = rule["op"]
    if op == "constant":
        return rule["value"]
    if op == "flag":
        test = world[rule["flag"]]
    elif op == "and":
        test = all(world[k] for k in rule["flags"])
    elif op == "at_least":
        test = sum(world[k] for k in rule["flags"]) >= rule["count"]
    elif op == "priority":
        test = world[rule["flags"][0]] and not world[rule["flags"][1]]
    else:
        raise ValueError("Unknown finite policy")
    return rule["true"] if test else rule["false"]


def make_group(family, index, split, flags, plus_rule, clauses):
    """clauses are independently authored source rules, never decoder outputs."""
    site = f"the {SITES[index]} {family} station"
    options = ["use the copper tray", "use the silver tray"]
    worlds = [dict(zip(flags, values)) for values in itertools.product((False, True), repeat=len(flags))]
    poles = {}
    for pole in ("plus", "minus"):
        rule = dict(plus_rule)
        if pole == "minus":
            for key in ("true", "false", "value"):
                if key in rule:
                    rule[key] = 1 - rule[key]
        choices = options if pole == "plus" else options[::-1]
        clause = clauses(choices)
        sources = [f"At {site}, my policy is this: {clause}", f"This is my rule for {site}: {clause}"]
        ledger = [{"world": w, "choice": options[evaluate(rule, w)]} for w in worlds]
        table = " ".join((", ".join(f"{k} is {'on' if v else 'off'}" for k, v in row["world"].items()) or "In every case")
                         + ": I " + row["choice"] + "." for row in ledger)
        targets = {op: (sources[0] if family == "exclusion" else f"My policy at {site} distinguishes these cases. {table}")
                   for op in OPERATORS}
        poles[pole] = {"sources": sources, "rule": rule, "truth_ledger": ledger, "operator_targets": targets,
                       "operator_target_review": "formal_truth_checked_human_usefulness_pending"}
    return {"id": f"{family}-{index:02d}", "family": family, "split": split, "site": site,
            "options": options, "flags": flags, "poles": poles, "lineage_group": f"{family}-{index:02d}",
            "claim_scope": "synthetic_explicit_policy_preservation"}


def public_groups():
    for family, split in FAMILIES.items():
        for i in range(12):
            if family in ("conditional", "exception", "scope"):
                flag = {"conditional": "signal", "exception": "alarm", "scope": "visitor_mode"}[family]
                rule, flags = {"op": "flag", "flag": flag, "true": 0, "false": 1}, [flag]
                if family == "exception":
                    clause = lambda c: f"I {c[1]}, except when alarm is on; then I {c[0]}."
                else:
                    clause = lambda c, flag=flag: f"When {flag} is on, I {c[0]}; when it is off, I {c[1]}."
            elif family == "conjunction":
                flags = ["signal", "permit"]
                rule = {"op": "and", "flags": flags, "true": 0, "false": 1}
                clause = lambda c: f"Only when signal and permit are both on do I {c[0]}; otherwise I {c[1]}."
            elif family == "exclusion":
                flags, rule = [], {"op": "constant", "value": 0}
                clause = lambda c: f"I {c[0]}. I never {c[1]}."
            else:
                flags = ["request", "override"]
                rule = {"op": "priority", "flags": flags, "true": 0, "false": 1}
                clause = lambda c: f"I {c[0]} when request is on, unless override is on. If request is off or override is on, I {c[1]}."
            yield make_group(family, i, split, flags, rule, clause)


def write_public(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    groups = list(public_groups())
    manifest = {"status": "FORMAL_CODEC_PREPARATION_NOT_SEMANTIC_QUALIFICATION", "schema": "finite_policy_groups",
                "human_transition_review": "pending", "files": {}, "qualification": {"status": "not_sealed"}}
    for split in ("train", "dev"):
        selected = [g for g in groups if g["split"] == split]
        name = split + ".groups.jsonl"
        value = "".join(json.dumps(g, sort_keys=True, ensure_ascii=False) + "\n" for g in selected)
        (out / name).write_text(value)
        manifest["files"][split] = {"path": name, "sha256": hashlib.sha256(value.encode()).hexdigest(),
                                   "groups": len(selected), "source_variants": 4 * len(selected),
                                   "families": sorted({g["family"] for g in selected})}
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def codec_examples(folder, split):
    if split not in {"train", "dev"}:
        raise ValueError("The codec loader cannot load qualification material")
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    entry = manifest["files"][split]
    if entry["path"] != split + ".groups.jsonl":
        raise ValueError("Unexpected split path")
    raw = (folder / entry["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("Calibration data changed")
    rows = []
    for group in map(json.loads, raw.decode().splitlines()):
        if group["split"] != split or FAMILIES.get(group["family"]) != split:
            raise ValueError("Cross-split family or group")
        for pole, details in group["poles"].items():
            for phrasing, source in enumerate(details["sources"]):
                # First codec learning reconstructs its source exactly. G teachers remain pending.
                rows.append({"id": f"{group['id']}-{pole}-p{phrasing}", "group": group["id"],
                             "source": source, "target": source})
    return rows
