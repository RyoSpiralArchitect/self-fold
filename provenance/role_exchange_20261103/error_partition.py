"""Descriptive attribution of fresh classification errors and a check on the ordinary role head."""

from collections import Counter
from pathlib import Path
import json

from summarize import CELLS, decisions, old

RUN = Path(__file__).resolve().parent / "run"


def partition():
    fresh = [r for r in old.rows(RUN / "scores.jsonl") if r["split"] == "fresh"]
    ledger = Counter()
    confusion = Counter()
    names = {r["topic"]: r["group"] for r in fresh}
    for raw in fresh:
        r = decisions(raw)
        if not r["topic_ok"]:
            ledger["wrong_topic"] += 1
            confusion[
                r["group"] + "/" + names.get(raw["weighted"]["topic"], "abstain")
            ] += 1
        elif r["action_ok"] and r["role_ok"]:
            ledger["both_components_correct"] += 1
        elif r["action_ok"]:
            ledger["role_only_wrong"] += 1
        elif r["role_ok"]:
            ledger["action_only_wrong"] += 1
        elif r["both_wrong"]:
            ledger["two_errors_cancel"] += 1
        else:
            ledger["tie"] += 1
    ordinary = {}
    for frame in ("all", *CELLS):
        data = [r for r in fresh if frame == "all" or r["frame"] == frame]
        keyed = {
            (r["topic"], r["frame"], r["pair"], r["pole"], r["order"]): r for r in data
        }
        ok = lambda r: r["ordinary"]["role"] * (1 - 2 * r["order"]) > 0
        ordinary[frame] = {
            "singletons": old.count(ok(r) for r in data),
            "both_members": old.count(
                ok(r) and ok(keyed[t, f, w, 1, 1 - o])
                for (t, f, w, p, o), r in keyed.items()
                if p == 0
            ),
        }
    return {
        "fresh_all": dict(ledger),
        "wrong_topic_counts": dict(sorted(confusion.items())),
        "ordinary_first_role": ordinary,
    }


if __name__ == "__main__":
    print(json.dumps(partition(), indent=2))
