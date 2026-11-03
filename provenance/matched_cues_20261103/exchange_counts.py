"""Post-run descriptive counts on the registered opposed pairs. No new model calls."""

from collections import Counter
import json
from pathlib import Path
from summarize import old

RUN = Path(__file__).resolve().parent / "run"


def report():
    teachers = old.rows(RUN / "teachers.jsonl")
    teacher = {}
    for label in ("bare", "topic", "placebo", "rotated"):
        idx = {
            (r["topic"], r["stem"], r["label_order"], r["pole"]): r
            for r in teachers
            if r["route"] == "calibrated" and r["header"] == label
        }
        pairs = Counter()
        for (t, s, o, p), left in idx.items():
            if p:
                continue
            right = idx[t, s, o, 1]
            result = (left["greedy_label"], right["greedy_label"])
            if any(v not in ("A", "B") for v in result):
                cell = "outside_labels"
            elif result[0] == result[1]:
                cell = "same_label"
            else:
                cell = (
                    "correct_switch"
                    if old.correct(left) and old.correct(right)
                    else "inverted_switch"
                )
            pairs[cell] += 1
        teacher[label] = dict(pairs)
    sources = old.rows(RUN / "sources.jsonl")
    literal = old.rows(RUN / "literal.jsonl")
    native = {(r["key"], r["label_order"]): old.correct(r) for r in literal}
    source = {}
    for pos in ("before", "after"):
        idx = {
            (r["topic"], r["family"], r["pair"], r["pole"], r["order"]): r
            for r in sources
            if r["position"] == pos
        }
        table = Counter()
        crossed = Counter()
        differing_states = 0
        for (t, f, w, p, o), left in idx.items():
            if p:
                continue
            right = idx[t, f, w, 1, 1 - o]
            differing_states += left["state_sha256"] != right["state_sha256"]
            a, b = left["factor"], right["factor"]
            if (a, b) == (2 * t, 2 * t + 1):
                cell = "correct_switch"
            elif (a, b) == (2 * t + 1, 2 * t):
                cell = "inverted_switch"
            elif a is not None and a == b and a // 2 == t:
                cell = "same_policy"
            else:
                cell = "other_or_abstained"
            table[cell] += 1
            lit = all(
                native[r["key"], order] for r in (left, right) for order in (0, 1)
            )
            crossed[
                ("literal_complete" if lit else "literal_incomplete") + "/" + cell
            ] += 1
        source[pos] = {
            "all": dict(table),
            "opposed_E_hashes_differ": differing_states,
            "by_literal_both_poles_and_label_orders": dict(crossed),
        }
    return {
        "teacher_opposed_at_fixed_question": teacher,
        "source_opposed_at_fixed_action_order": source,
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
