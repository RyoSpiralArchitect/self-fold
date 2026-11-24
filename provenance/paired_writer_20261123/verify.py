"""Reconstruct the joint endpoint from raw reports, assignment bits and native logits."""

from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import runpy
import sys

import numpy as np


def read(path):
    return json.loads(path.read_text())


def verify(folder):
    study = Path(__file__).resolve().parent
    root = study.parents[1]
    parse = runpy.run_path(str(study.parent / "structured_report_20261122/verify.py"))[
        "reports_fields"
    ]
    cfg = read(study / "design.json")
    vocab = [
        name.casefold()
        for g in read(root / "data/within_topic_wording_20261103.json")["groups"]
        for name in g["options"]
    ]
    raw = [
        json.loads(line) for line in (folder / "reports.jsonl").read_text().splitlines()
    ]
    outcomes = read(folder / "outcomes.json")
    by_id = {(row["id"], row["arm"]): row for row in outcomes}
    keyset = {
        (f"{m}/{pat}/{bit}/{rotation}", arm)
        for m, pat, bit, rotation, arm in product(
            cfg["menus"],
            cfg["test_patterns"],
            (0, 1),
            range(4),
            ("parent", "individual", "paired"),
        )
    }
    if (
        len(raw) != 1152
        or {(r["id"], r["arm"]) for r in raw} != keyset
        or len(by_id) != 1920
        or len(outcomes) != 1920
    ):
        raise ValueError("Archive coverage differs from design")
    data = {}
    for split in ("fresh", "previous"):
        parent = folder / ("native_" + split)
        with np.load(parent / "scores.npz", allow_pickle=False) as scores:
            indices = scores["dispatch"].reshape(4, 16, 2)
            assert sorted(indices.flatten()) == list(range(128))
            data[split] = (
                scores["top"][indices],
                scores["ab"][indices],
                read(parent / "native.json")["labels"],
            )
    results = defaultdict(list)
    for q in raw:
        m, pat, default, rotation = map(int, q["id"].split("/"))
        split = "fresh" if m in cfg["fresh"] else "previous"
        top, ab, labels = data[split]
        mi = cfg[split].index(m)
        gold_rows = [2 * m + ((pat >> w) & 1) for w in range(4)]
        gold_default = 2 * m + default
        values = parse(q["text"], vocab)
        correct = 0
        if None not in values and len({v // 2 for v in values}) == 1:
            action = (
                [values[-1], values[-1] ^ 1]
                if len(set(values[:4])) > 1
                else [values[0]] * 2
            )
            for mask in cfg["next_masks"]:
                for world, order in product(range(4), range(2)):
                    triggered = (mask >> world) & 1
                    idx = action[triggered]
                    expected = default ^ triggered ^ order
                    margin = ab[mi, idx, order]
                    correct += int(
                        top[mi, idx, order] == labels[expected]
                        and margin[expected] > margin[1 - expected]
                    )
        saved = by_id[q["id"], q["arm"]]
        assert (values, correct) == (saved["fields"], saved["correct"])
        results[split, q["arm"], m, pat, rotation].append(
            (
                default,
                values[:4] == gold_rows,
                values[-1] == gold_default,
                correct == 112,
            )
        )
    summary = read(folder / "summary.json")
    for split, arm in product(
        ("fresh", "previous"), ("parent", "individual", "paired")
    ):
        twins = [p for k, p in results.items() if k[:2] == (split, arm)]
        assert len(twins) == 96 and all(
            sorted(v[0] for v in p) == [0, 1] for p in twins
        )
        scores = summary[split]["arms"][arm]
        for idx, metric in enumerate(("rows", "role", "final"), 1):
            assert (
                sum(all(r[idx] for r in pair) for pair in twins)
                == scores["twins"][metric]
            )
        assert (
            sum(all(all(r[1:]) for r in pair) for pair in twins)
            == scores["twins"]["joint"]
        )
        assert sum(r[1] for p in twins for r in p) == scores["four_rows"]
    return dict(
        exact=True,
        free_reports=len(raw),
        native_judgments=len(raw) * 112,
        distinct_native_inputs=256,
        panels=2,
        writer_arms=3,
        refits=0,
    )


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]))))
