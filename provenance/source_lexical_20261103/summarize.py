"""Reduce fixed source-map decisions and all counterfactual controls."""

from collections import defaultdict
import json
from pathlib import Path
import sys

PANELS = {
    "train_fit": 64,
    "train_new_question": 64,
    "new_style": 128,
    "previous_topics": 64,
    "new_topics": 64,
    "all": 384,
}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def ratio(correct, total):
    return {"correct": correct, "total": total, "fraction": correct / total}


def member(row, panel):
    if panel == "train_fit":
        return row["split"] == "seen_sources" and row["wording"] == 0
    if panel == "train_new_question":
        return row["split"] == "seen_sources" and row["wording"] == 1
    return panel == "all" or row["split"] == {
        "previous_topics": "previous_lexical_transfer",
        "new_topics": "new_transfer_topic",
    }.get(panel, panel)


def metrics(panel):
    by_unit, by_order, by_pole = defaultdict(list), defaultdict(list), defaultdict(dict)
    biases = defaultdict(list)
    for r in panel:
        by_unit[r["group"], r["style"]].append(r)
        by_order[
            r["group"], r["style"], r["pole"], r["wording"], r["label_order"]
        ].append(r)
        by_pole[r["group"], r["phrasing"], r["wording"], r["label_order"]][
            r["pole"]
        ] = r
        biases[r["group"]].append(
            (r["A"] - r["B"]) * (1 if r["label_order"] == 0 else -1)
        )
    questions = {r["wording"] for r in panel}
    expected = 8 * len(questions)
    for group in by_unit.values():
        assert len(group) == expected
        assert (
            len(
                {
                    (r["pole"], r["source_order"], r["wording"], r["label_order"])
                    for r in group
                }
            )
            == expected
        )
    assert all(
        len(group) == 2 and {r["source_order"] for r in group} == {0, 1}
        for group in by_order.values()
    )
    assert all(set(group) == {"plus", "minus"} for group in by_pole.values())
    gaps = []
    for pair in by_pole.values():
        first, second = pair["plus"], pair["minus"]
        orientation = 1 if first["label_order"] == 0 else -1
        gaps.append(
            orientation * ((first["A"] - first["B"]) - (second["A"] - second["B"]))
        )
    return {
        "native": ratio(sum(r["greedy_label"] == r["gold"] for r in panel), len(panel)),
        "forced": ratio(
            sum(r[r["gold"]] > r["B" if r["gold"] == "A" else "A"] for r in panel),
            len(panel),
        ),
        "forced_ties": sum(r["A"] == r["B"] for r in panel),
        "native_label_coverage": sum(r["greedy_label"] in ("A", "B") for r in panel),
        "strict_pairs_both_source_orders": ratio(
            sum(
                all(r["greedy_label"] == r["gold"] for r in group)
                for group in by_unit.values()
            ),
            len(by_unit),
        ),
        "order_margin_sign_agreement": ratio(
            sum(
                (group[0]["A"] - group[0]["B"]) * (group[1]["A"] - group[1]["B"]) > 0
                for group in by_order.values()
            ),
            len(by_order),
        ),
        "mean_pole_gap": sum(gaps) / len(gaps),
        "mean_plus_margin_by_topic": {
            key: sum(value) / len(value) for key, value in sorted(biases.items())
        },
    }


def summary(folder):
    terminal = json.loads((folder / "terminal.json").read_text())
    if terminal["status"] != "COMPLETED_SOURCE_LEXICAL_DEVELOPMENT":
        return {"terminal": terminal}
    decisions = rows(folder / "decisions.jsonl")
    assert len(decisions) == 384 * (3 + 2 * 6)
    result = {}
    for endpoint in ("controls", "ridge", "final"):
        result[endpoint] = {}
        routes = (
            ("raw_E", "literal", "lexical_teacher")
            if endpoint == "controls"
            else ("matched", "opposed", "mean", "zero", "reverse", "norm")
        )
        for route in routes:
            route_rows = [
                r
                for r in decisions
                if r["endpoint"] == endpoint and r["route"] == route
            ]
            assert len(route_rows) == 384
            group = {}
            for panel, count in PANELS.items():
                chosen = [r for r in route_rows if member(r, panel)]
                assert len(chosen) == count
                group[panel] = metrics(chosen)
            result[endpoint][route] = group
        if endpoint == "controls":
            continue
        matched = {
            (r["key"], r["wording"], r["label_order"]): r
            for r in decisions
            if r["endpoint"] == endpoint and r["route"] == "matched"
        }
        flip = {}
        for panel, count in PANELS.items():
            compared = [
                r
                for r in decisions
                if r["endpoint"] == endpoint
                and r["route"] == "opposed"
                and member(r, panel)
            ]
            assert len(compared) == count
            flips = 0
            for r in compared:
                original = matched[r["key"], r["wording"], r["label_order"]]
                flips += (r["A"] - r["B"]) * (original["A"] - original["B"]) < 0
            flip[panel] = ratio(flips, count)
        result[endpoint]["opposed_margin_sign_flip"] = flip
        screens = {}
        for panel in ("train_fit", "new_style", "new_topics"):
            m = result[endpoint]["matched"][panel]
            if panel == "train_fit":
                screens[panel] = (
                    m["native"]["fraction"] >= 0.95
                    and m["strict_pairs_both_source_orders"]["fraction"] >= 0.9
                )
            else:
                null = max(
                    result[endpoint][route][panel]["native"]["fraction"]
                    for route in ("mean", "zero", "norm")
                )
                opposed = result[endpoint]["opposed"][panel]["native"]["fraction"]
                screens[panel] = (
                    m["native"]["fraction"] >= 0.8
                    and m["strict_pairs_both_source_orders"]["fraction"] >= 0.75
                    and m["native"]["fraction"] - null >= 0.1 - 1e-12
                    and m["native"]["fraction"] - opposed >= 0.3 - 1e-12
                )
        result[endpoint]["screens"] = screens | {
            "interface_admission": all(screens.values())
        }
    return {"terminal": terminal, "metrics": result}


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "run"
    print(json.dumps(summary(folder), indent=2, sort_keys=True))
