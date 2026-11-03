"""Post-run descriptive pairing of the fixed source readers; no fitting or new model calls."""

import json
from pathlib import Path
from summarize import prev, old


def report():
    data = [
        r
        for r in old.rows(Path(__file__).parent / "run/source_scores.jsonl")
        if r["split"] == "fresh"
    ]

    def pred(r, arm):
        score = (r["frozen"]["actions"] if arm == "old" else r[arm])[r["topic"]]
        return score * (1 - 2 * (r["pole"] ^ r["order"])) > 0

    output = {}
    for left, right in (("invariant", "swap"), ("invariant", "old"), ("swap", "old")):
        a = [dict(r, ok=pred(r, left)) for r in data]
        b = [dict(r, ok=pred(r, right)) for r in data]
        output[left + "_minus_" + right] = prev.paired(
            a, b, ("key",), lambda r: r["ok"]
        )
    return output


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
