"""Build the unscored clause-order panel and measure a deterministic template shortcut."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/codec_preparation_20261102"
CLAUSES = {"scope": ("visitor_mode is on", "visitor_mode is off"),
           "priority": ("request is on and override is off", "request is off or override is on")}


def trigger(family, world):
    if family == "scope":
        return world["visitor_mode"]
    if family == "priority":
        return world["request"] and not world["override"]
    raise ValueError("Unexpected construction")


def mentions(text):
    words = re.findall(r"use the (?:copper|silver) tray", text)
    if len(words) != 2 or len(set(words)) != 2:
        raise ValueError("Expected the two distinct actions once each")
    return words


def build():
    raw = (DATA / "dev.groups.jsonl").read_bytes()
    expected = json.loads((DATA / "manifest.json").read_text())["files"]["dev"]["sha256"]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Public dev bytes changed")
    output, counts = [], Counter()
    for group in map(json.loads, raw.decode().splitlines()):
        family = group["family"]
        for pole, detail in group["poles"].items():
            actions = group["options"][::1 if pole == "plus" else -1]
            clauses = [f"If {cond}, I {action}." for cond, action in zip(CLAUSES[family], actions, strict=True)]
            for order in (0, 1):
                description = f"At {group['site']}, I follow this policy. " + " ".join(clauses[::1 if order == 0 else -1])
                old, new = mentions(detail["sources"][order]), mentions(description)
                for entry in detail["truth_ledger"]:
                    i = 0 if trigger(family, entry["world"]) else 1
                    # Check the action assigned by this text, accounting for the stated order.
                    if new[i ^ order] != entry["choice"]:
                        raise ValueError("Compiled clause disagrees with the independent public truth ledger")
                    for label, guessed in (("original", old[i]), ("crossed", new[i])):
                        counts[family, label, "correct"] += guessed == entry["choice"]
                        counts[family, label, "total"] += 1
                output.append({"id": f"{group['id']}-{pole}-o{order}", "group": group["id"], "pole": pole,
                               "trigger_clause_first": order == 0, "source": description,
                               "truth_ledger": detail["truth_ledger"]})
    report = {family: {panel: {key: counts[family, panel, key] for key in ("correct", "total")}
                       for panel in ("original", "crossed")} for family in CLAUSES}
    report.update(source_dev_sha256=expected, groups=24, descriptions=len(output),
                  unit="one description and world, no duplicated label orders", model_scores=0)
    return output, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", action="store_true", help="Emit prospective description records as JSONL")
    args = parser.parse_args()
    panel, report = build()
    if args.panel:
        for row in panel:
            print(json.dumps(row, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
