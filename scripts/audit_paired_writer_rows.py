"""Audit parsed row fields and saved native judgment counts without a model or fit."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "provenance/paired_writer_20261123/produce/outcomes.json"
OLD_ARMS = ("parent", "individual", "paired", "L", "oracle")
NEXT_ARMS = ("parent", "mean", "row_ce", "agreement", "combined", "L", "oracle")
PANELS = {"fresh": (1, 3, 5, 7), "previous": (0, 2, 4, 6)}
PATTERNS = (3, 5, 6, 9, 10, 12)
COORDS = ("off/off", "off/on", "on/off", "on/on")
RELATIONS = ("agree_correct", "agree_wrong", "disagree", "one_missing", "both_missing")
ROLE_STATUS = ("correct_flip", "swapped", "same", "other_distinct", "missing")


def relation(left, right, gold):
    if left is None and right is None:
        return "both_missing"
    if left is None or right is None:
        return "one_missing"
    if left != right:
        return "disagree"
    return "agree_correct" if left == gold else "agree_wrong"


def role_status(values, gold):
    if None in values:
        return "missing"
    if values == gold:
        return "correct_flip"
    if values == gold[::-1]:
        return "swapped"
    return "same" if values[0] == values[1] else "other_distinct"


def index_records(records, arms):
    index = defaultdict(dict)
    for r in records:
        split, arm = r["split"], r["arm"]
        menu, pat, rot, donor = (r[k] for k in ("menu", "pattern", "rotation", "default"))
        if (
            any(type(x) is not int for x in (menu, pat, rot, donor))
            or menu not in PANELS.get(split, ())
            or arm not in arms
            or pat not in PATTERNS
            or rot not in range(4)
            or donor not in (0, 1)
            or r["id"] != f"{menu}/{pat}/{donor}/{rot}"
        ):
            raise ValueError(f"Unexpected panel key: {r['id']}, {split}, {arm}")
        fields = r["fields"]
        if len(fields) != 5 or any(
            v is not None and (type(v) is not int or v not in range(16)) for v in fields
        ):
            raise ValueError(f"Invalid parsed fields: {r['id']}, {arm}")
        if (
            r["total"] != 112
            or type(r["correct"]) is not int
            or not 0 <= r["correct"] <= 112
            or r["complete"] is not (r["correct"] == r["total"])
        ):
            raise ValueError(f"Invalid downstream count: {r['id']}, {arm}")
        pair = index[split, arm, menu, pat, rot]
        if donor in pair:
            raise ValueError(f"Duplicate donor: {r['id']}, {arm}")
        pair[donor] = r
    expected = {
        (s, a, m, p, r)
        for s, menus in PANELS.items()
        for a, m, p, r in product(arms, menus, PATTERNS, range(4))
    }
    if index.keys() != expected or any(p.keys() != {0, 1} for p in index.values()):
        raise ValueError("Incomplete panel, arm, rotation or donor")
    return index


def pair_outcome(menu, pattern, rotation, pair):
    gold = [2 * menu + ((pattern >> i) & 1) for i in range(4)]
    fields = [pair[i]["fields"] for i in (0, 1)]
    row_hits = [[v == g for v, g in zip(report[:4], gold)] for report in fields]
    roles = [v[-1] for v in fields]
    gold_roles = [2 * menu, 2 * menu + 1]
    role_hits = [v == g for v, g in zip(roles, gold_roles)]
    for i, record in pair.items():
        if row_hits[i] != record["rows_correct"] or role_hits[i] != record["role"]:
            raise ValueError(
                f"Saved scoring flags disagree with assignment: {record['id']}, {record['arm']}"
            )
    rel = [relation(fields[0][i], fields[1][i], g) for i, g in enumerate(gold)]
    four_rows = [all(x) for x in row_hits]
    exact = all(four_rows)
    shared_error = "agree_wrong" in rel
    equal_table = all(x in ("agree_correct", "agree_wrong") for x in rel)
    correct = [pair[d]["correct"] for d in (0, 1)]
    final = all(n == 112 for n in correct)
    return {
        "menu": menu,
        "pattern": pattern,
        "rotation": rotation,
        "gold": gold,
        "fields": fields,
        "relations": rel,
        "role_status": role_status(roles, gold_roles),
        "correct_choices": correct,
        "four_row_reports": sum(four_rows),
        "correct_defaults": sum(role_hits),
        "rows": exact,
        "role": all(role_hits),
        "final": final,
        "joint": exact and all(role_hits) and final,
        "any_shared_wrong": shared_error,
        "same_wrong_table": equal_table and shared_error,
        "final_with_row_error": final and not exact,
    }


PAIR_METRICS = (
    "rows",
    "role",
    "final",
    "joint",
    "any_shared_wrong",
    "same_wrong_table",
    "final_with_row_error",
)


def counts(pairs):
    return {"pairs": len(pairs), **{k: sum(p[k] for p in pairs) for k in PAIR_METRICS}}


def summarize_arm(pairs, details):
    relations = {c: Counter(dict.fromkeys(RELATIONS, 0)) for c in COORDS}
    cells = {c: [Counter(correct=0, wrong=0, missing=0) for _ in range(2)] for c in COORDS}
    positions = {i: Counter(correct=0, wrong=0, missing=0) for i in range(4)}
    role_counts = Counter(dict.fromkeys(ROLE_STATUS, 0))
    physical = defaultdict(list)
    by_menu, by_rotation = defaultdict(list), defaultdict(list)
    for p in pairs:
        m, pat, rot = p["menu"], p["pattern"], p["rotation"]
        physical[m, pat].append(p)
        by_menu[m].append(p)
        by_rotation[rot].append(p)
        role_counts[p["role_status"]] += 1
        for i, coord in enumerate(COORDS):
            relations[coord][p["relations"][i]] += 1
            for d, fields in enumerate(p["fields"]):
                val = fields[i]
                status = "missing" if val is None else "correct" if val == p["gold"][i] else "wrong"
                cells[coord][d][status] += 1
                positions[(i - rot) % 4][status] += 1
    tables = [
        dict(
            menu=m,
            pattern=pat,
            **counts(group),
            varying_donor_tables=sum(
                len({tuple(p["fields"][d][:4]) for p in group}) > 1 for d in (0, 1)
            ),
        )
        for (m, pat), group in sorted(physical.items())
    ]
    hist = {k: [sum(t[k] == n for t in tables) for n in range(5)] for k in PAIR_METRICS}
    result = dict(
        **counts(pairs),
        reports=2 * len(pairs),
        four_row_reports=sum(p["four_row_reports"] for p in pairs),
        correct_defaults=sum(p["correct_defaults"] for p in pairs),
        role_status=role_counts,
        correct_choices=sum(sum(p["correct_choices"]) for p in pairs),
        total_choices=224 * len(pairs),
        row_by_donor=cells,
        row_by_display_position=positions,
        donor_relations=relations,
        by_menu={k: counts(v) for k, v in by_menu.items()},
        by_rotation={k: counts(v) for k, v in by_rotation.items()},
        physical_assignments=len(tables),
        rotation_histogram=hist,
        donor_tables_vary_across_rotations=sum(t["varying_donor_tables"] for t in tables),
    )
    if details:
        result.update(assignments=tables, presentations=pairs)
    return result


def audit(records, arms=OLD_ARMS, details=False):
    index = index_records(records, arms)
    result = {}
    for split, menus in PANELS.items():
        result[split] = {}
        for arm in arms:
            pairs = [
                pair_outcome(m, p, r, index[split, arm, m, p, r])
                for m, p, r in product(menus, PATTERNS, range(4))
            ]
            result[split][arm] = summarize_arm(pairs, details)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outcomes", type=Path, nargs="?", default=ARCHIVE)
    parser.add_argument(
        "--next", action="store_true", help="expect the seven routes of one future seed"
    )
    parser.add_argument(
        "--details", action="store_true", help="include every assignment and donor pair"
    )
    args = parser.parse_args()
    data = args.outcomes.read_bytes()
    records = json.loads(data)
    print(
        json.dumps(
            {
                "input_sha256": hashlib.sha256(data).hexdigest(),
                "records": len(records),
                "panels": audit(records, NEXT_ARMS if args.next else OLD_ARMS, args.details),
            },
            indent=2,
            sort_keys=True,
        )
    )
