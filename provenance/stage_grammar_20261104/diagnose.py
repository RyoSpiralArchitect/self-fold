"""Descriptive grammar-only, name-only and incomplete-unit counts from the sealed native rows."""

from collections import Counter, defaultdict
import argparse
import gzip
import json
from pathlib import Path

from summarize import ARMS, old, read


def diagnostics(literal):
    labels = {token: label for label, token in read(literal / "labels.json").items()}
    output = {}
    for arm in ARMS:
        grouped = defaultdict(list)
        with gzip.open(literal / (arm + ".jsonl.gz"), "rt") as stream:
            for line in stream:
                panel, kind, k, w, o, g, token, a, b, _, _ = json.loads(line)
                if kind != "ordinary":
                    continue
                _, group, flag, style, stage, pole, order = k.split("/")
                row = dict(
                    key=k,
                    world=w,
                    label_order=o,
                    gold=g,
                    greedy_label=labels.get(token),
                    A=a,
                    B=b,
                )
                grouped[panel, stage].append(row)
                if panel == "fit" and int(flag) < 2 and style == "if_else":
                    grouped["familiar", stage].append(row)
        report = {}
        for (panel, stage), rows in grouped.items():
            unit = defaultdict(int)
            by_order = Counter()
            for row in rows:
                _, g, f, s, _, _, order = row["key"].split("/")
                fail = row["greedy_label"] != row["gold"]
                unit[g, f, s] += int(fail)
                if fail:
                    by_order[order + "/" + row["gold"]] += 1
            basic = old.prior.measure(rows)
            report[f"{panel}/{stage}"] = dict(
                native=basic["native"],
                strict=basic["strict"],
                misses_per_unit={
                    str(n): v for n, v in sorted(Counter(unit.values()).items())
                },
                missed_by_source_order_and_gold=dict(sorted(by_order.items())),
            )
        output[arm] = report
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("literal", type=Path)
    print(json.dumps(diagnostics(p.parse_args().literal), indent=2))
