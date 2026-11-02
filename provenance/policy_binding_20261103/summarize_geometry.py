"""Single-source and paired-order accuracy from the fixed geometry panel."""
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def summarize(folder):
    if json.loads((folder / "terminal.json").read_text())["status"] != "COMPLETED_SINGLE_SOURCE_GEOMETRY_DEV":
        raise ValueError("Incomplete geometry experiment")
    cells = defaultdict(dict)
    for line in (folder/"scores.jsonl").read_text().splitlines():
        x = json.loads(line)
        key = (x["panel"], x["label"], x["feature"], x["sham"])
        unit = (x["group"], x["pole"], x["variant"])
        if unit in cells[key] or not math.isfinite(x["copper_margin"]):
            raise ValueError("Duplicate or nonfinite geometry result")
        cells[key][unit] = (1 if x["gold_copper"] else -1) * x["copper_margin"]
    keys = itertools.product(("original", "crossed"), ("first_mention", "trigger_action"),
                             ("raw", "centered", "reference"), (False, True))
    out = {}
    for panel, label, feature, sham in keys:
        scores = cells[panel, label, feature, sham]
        variants = ("p0", "p1") if panel == "original" else ("o0", "o1")
        if len(scores) != 96:
            raise ValueError("Incomplete panel")
        entry = {}
        for family in ("all", "scope", "priority"):
            families = ["scope", "priority"] if family == "all" else [family]
            pairs = [[scores[f"{f}-{i:02d}", p, v] for v in variants]
                     for f in families for i in range(12) for p in ("plus", "minus")]
            margins = [m for pair in pairs for m in pair]
            entry[family] = {"correct": sum(m > 0 for m in margins), "n": len(margins),
                             "both_orders": sum(all(m > 0 for m in pair) for pair in pairs), "pairs": len(pairs),
                             "ties": sum(m == 0 for m in margins)}
        out.setdefault(panel, {}).setdefault(label, {})[feature + ("_sham" if sham else "")] = entry
    if len(cells) != 24:
        raise ValueError("Unexpected geometry conditions")
    screens = {}
    for label, variants in out["crossed"].items():
        screens[label] = {}
        for f in ("raw", "centered"):
            m = variants[f]["all"]
            null = max(variants[key]["all"]["correct"]/96 for key in (f + "_sham", "reference"))
            screens[label][f] = (m["correct"]/96 >= .8 and m["correct"]/96-null >= .1 and
                                 (label != "trigger_action" or m["both_orders"]/48 >= .8))
    return {"panels": out, "crossed_screens": screens}


if __name__ == "__main__":
    import sys
    print(json.dumps(summarize(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "geometry"), indent=2, sort_keys=True))
