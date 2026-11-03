"""Rebuild the frozen matched-cue effects from the two native and source panels."""

from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

STUDY = Path(__file__).resolve().parent
spec = spec_from_file_location(
    "previous_polarity_scores", STUDY.parent / "teacher_polarity_20261103/summarize.py"
)
old = module_from_spec(spec)
spec.loader.exec_module(old)


def indexed(rows, fields):
    result = {tuple(r[k] for k in fields): r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate paired observation")
    return result


def paired(left, right, fields, correct):
    a, b = indexed(left, fields), indexed(right, fields)
    if not a or a.keys() != b.keys():
        raise ValueError("Unmatched comparison")
    cells = Counter()
    topic_delta, topic_n = Counter(), Counter()
    for k, v in a.items():
        x, y = bool(correct(v)), bool(correct(b[k]))
        cells[x, y] += 1
        topic_delta[v["topic"]] += int(x) - int(y)
        topic_n[v["topic"]] += 1
    difference = cells[True, False] - cells[False, True]
    n = len(a)
    leave = [
        100 * (difference - topic_delta[t]) / (n - topic_n[t])
        for t in topic_n
        if n > topic_n[t]
    ]
    return {
        "both": cells[True, True],
        "first_only": cells[True, False],
        "second_only": cells[False, True],
        "neither": cells[False, False],
        "net": difference,
        "n": n,
        "by_topic_net": dict(sorted(topic_delta.items())),
        "delta_pp": 100 * difference / n,
        "leave_one_topic_out_pp": [min(leave), max(leave)] if leave else None,
    }


def teacher_panel(rows):
    sections, by_cond = {}, {}
    for header in ("bare", "topic", "placebo", "rotated"):
        by_cond[header] = [
            r for r in rows if r["header"] == header and r["route"] == "calibrated"
        ]
        for route in ("calibrated", "mean"):
            group = [r for r in rows if r["header"] == header and r["route"] == route]
            idx = indexed(group, ("topic", "stem", "pole", "label_order"))
            assert len(idx) == 128
            margins = {k: v["A"] - v["B"] for k, v in idx.items()}
            m = old.native(group)
            m.update(
                {
                    "by_stem": [
                        old.native([r for r in group if r["stem"] == q])["native"]
                        for q in range(4)
                    ],
                    "by_topic": [
                        old.native([r for r in group if r["topic"] == t])["native"]
                        for t in range(8)
                    ],
                    "strict": old.count(
                        all(old.correct(v) for k, v in idx.items() if k[0] == t)
                        for t in range(8)
                    ),
                    "flips": old.count(
                        d * margins[t, q, 1, o] < 0
                        for (t, q, p, o), d in margins.items()
                        if p == 0
                    ),
                }
            )
            sections[header + "/" + route] = m
    comp = {}
    fields = ("topic", "stem", "pole", "label_order")
    for ref in ("bare", "placebo", "rotated"):
        left, right = by_cond["topic"], by_cond[ref]
        p = paired(left, right, fields, old.correct)
        p["by_stem_net"] = [
            sections["topic/calibrated"]["by_stem"][q][0]
            - sections[ref + "/calibrated"]["by_stem"][q][0]
            for q in range(4)
        ]
        a, b = (indexed(v, fields) for v in (left, right))

        def signed(v):
            return (v["A"] - v["B"]) * (1 if v["gold"] == "A" else -1)

        p["mean_gold_margin_change"] = sum(
            signed(a[k]) - signed(b[k]) for k in a
        ) / len(a)
        comp[ref] = p
    primary = comp["bare"]
    effect = (
        primary["net"] * 100 >= 10 * 128
        and all(comp[r]["net"] * 100 >= 5 * 128 for r in ("placebo", "rotated"))
        and sum(d > 0 for d in primary["by_stem_net"]) >= 3
        and sum(d > 0 for d in primary["by_topic_net"].values()) >= 5
    )
    named = sections["topic/calibrated"]
    reliable = (
        old.passes(named["native"], 95)
        and all(old.passes(n, 95) for n in named["by_stem"])
        and named["strict"][0] >= 7
        and old.passes(named["flips"], 90)
        and sections["topic/mean"]["native"][0] <= 64
    )
    return {
        "conditions": sections,
        "topic_minus": comp,
        "topic_specific_benefit": effect,
        "topic_cued_reliable": reliable,
    }


def components(r):
    scores, t = r["scores"], r["topic"]
    role = scores["role"] * (1 - 2 * r["order"])
    action = scores["first_action"][t] * (1 - 2 * (r["pole"] ^ r["order"]))
    return scores["topic"] == t, action, role


def both_right(r):
    topic, a, b = components(r)
    return topic and a > 0 and b > 0


def policy_right(r):
    return r["factor"] == 2 * r["topic"] + r["pole"]


def source_panel(records, literal):
    conditions, comparisons = {}, {}
    fields = ("topic", "family", "pair", "pole", "order")
    by_pos = {
        p: [r for r in records if r["position"] == p] for p in ("before", "after")
    }
    assert (
        len(records) == 512
        and all(len(v) == 256 for v in by_pos.values())
        and len(literal) == 1024
    )
    for pos, group in by_pos.items():
        for family in ("all", "accept", "choice"):
            data = [r for r in group if family == "all" or r["family"] == family]
            txt = [
                r
                for r in literal
                if r["position"] == pos and (family == "all" or r["family"] == family)
            ]
            assert len(indexed(txt, fields + ("label_order",))) == len(data) * 2
            scored = {
                m: old.source_measure(data, m)
                for m in ("factor", "direct", "mention", "oracle_factor")
            }
            parts = [components(r) for r in data]
            factor = scored["factor"]
            conditions[pos + "/" + family] = scored | {
                "topic": old.count(t for t, a, b in parts),
                "first_action_oracle_topic": old.count(a > 0 for t, a, b in parts),
                "first_role": old.count(b > 0 for t, a, b in parts),
                "topic_and_both_signs": old.count(both_right(r) for r in data),
                "topic_and_two_wrong_signs": old.count(
                    t and a < 0 and b < 0 for t, a, b in parts
                ),
                "literal": old.native(txt),
                "source_screen": all(
                    old.passes(factor[k], th)
                    for k, th in (
                        ("policy", 90),
                        ("strict", 75),
                        ("opposed_same_role_order", 90),
                        ("opposed_same_action_order", 90),
                    )
                ),
            }
    for name, score in (("policy", policy_right), ("both_signs", both_right)):
        metrics = paired(by_pos["before"], by_pos["after"], fields, score)
        for key, levels in (("family", ("accept", "choice")), ("order", (0, 1))):
            metrics["by_" + key + "_net"] = [
                paired(
                    *([r for r in rows if r[key] == level] for rows in by_pos.values()),
                    fields,
                    score,
                )["net"]
                for level in levels
            ]
        comparisons[name] = metrics
    lit = paired(
        [r for r in literal if r["position"] == "before"],
        [r for r in literal if r["position"] == "after"],
        fields + ("label_order",),
        old.correct,
    )
    readable = (
        all(
            old.passes(conditions[pos + "/all"]["literal"]["native"], 90)
            for pos in by_pos
        )
        and abs(lit["net"]) * 100 <= 5 * lit["n"]
    )
    direction = None
    for sign, pos in ((1, "before"), (-1, "after")):
        if (
            all(
                sign * m["net"] * 100 >= 10 * m["n"]
                and all(sign * d >= 0 for d in m["by_family_net"] + m["by_order_net"])
                for m in comparisons.values()
            )
            and sum(
                sign * d > 0 for d in comparisons["both_signs"]["by_topic_net"].values()
            )
            >= 5
        ):
            direction = pos
    return {
        "conditions": conditions,
        "before_minus_after": comparisons,
        "literal_before_minus_after": lit,
        "registered_direction": direction,
        "literal_readability_control": readable,
        "reader_specific_direction": direction if readable else None,
    }


def summary(folder):
    return {
        "teacher": teacher_panel(old.rows(folder / "teachers.jsonl")),
        "source": source_panel(
            old.rows(folder / "sources.jsonl"), old.rows(folder / "literal.jsonl")
        ),
    }


if __name__ == "__main__":
    print(
        json.dumps(
            summary(Path(sys.argv[1]) if len(sys.argv) > 1 else STUDY / "run"), indent=2
        )
    )
