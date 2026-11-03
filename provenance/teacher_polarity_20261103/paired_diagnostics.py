"""Descriptive paired comparisons after the two registered screens were scored."""

import json
from collections import Counter
from summarize import STUDY, OLD, correct, rows


def paired(predictions):
    counts = Counter(predictions)
    return {
        key: counts[a, b]
        for (a, b), key in (
            ((True, True), "both"),
            ((True, False), "first_only"),
            ((False, True), "second_only"),
            ((False, False), "neither"),
        )
    }


def report():
    run = STUDY / "run"
    sources = [r for r in rows(run / "sources.jsonl") if r["split"] == "fresh"]
    native = rows(run / "teachers.jsonl")
    receipt = {"source": {}}
    for frame in ("priority", "verdict", "all"):
        data = [r for r in sources if frame == "all" or r["frame"] == frame]
        receipt["source"][frame] = {
            "first_action_and_first_role": paired(
                (
                    r["scores"]["first_action"][r["topic"]]
                    * (1 - 2 * (r["pole"] ^ r["order"]))
                    > 0,
                    r["scores"]["role"] * (1 - 2 * r["order"]) > 0,
                )
                for r in data
            ),
            "factor_and_direct": paired(
                (
                    r["factor"] == 2 * r["topic"] + r["pole"],
                    r["direct"] == 2 * r["topic"] + r["pole"],
                )
                for r in data
            ),
        }
    fresh = {
        (r["topic"], r["pole"], r["wording"], r["label_order"], r["route"]): r
        for r in native
        if r["split"] == "fresh"
    }
    receipt["archived_and_calibrated"] = paired(
        (correct(r), correct(fresh[t, p, q, o, "calibrated"]))
        for (t, p, q, o, route), r in fresh.items()
        if route == "archived"
    )
    old = {
        (r["group"], r["pole"], r["wording"], r["label_order"]): r
        for r in rows(OLD / "decisions.jsonl")
        if r["arm"] == "reference"
        and r["route"] == "teacher"
        and r["panel"] in ("source", "joint")
    }
    flips = []
    for r in native:
        if r["route"] == "canonical" and r["split"] == "fit":
            k = (
                r["group"],
                ("plus", "minus")[r["pole"]],
                r["wording"],
                r["label_order"],
            )
            earlier = old[k]
            assert (
                earlier["question_sha256"],
                earlier["injected_sha256"],
                earlier["gold"],
            ) == (r["question_sha256"], r["injected_sha256"], r["gold"])
            if earlier["greedy_token"] != r["greedy_token"]:
                flips.append(
                    {
                        "key": k,
                        "gold": r["gold"],
                        **{
                            label: {
                                "choice": row["greedy_label"],
                                "A_minus_B": row["A"] - row["B"],
                            }
                            for label, row in (
                                ("previous_batch_of_four", earlier),
                                ("current_batch_of_two", r),
                            )
                        },
                    }
                )
    receipt["old_canonical_paired_scores"] = {
        "total": len(old),
        "changed_native_choice": flips,
    }
    return receipt


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
