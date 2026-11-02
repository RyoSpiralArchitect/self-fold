"""Recompute all carrier endpoints from full label likelihoods using only the stdlib."""
from collections import defaultdict
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def summarize(trial):
    terminal = json.loads((trial / "terminal.json").read_text())
    if terminal["status"] != "COMPLETED_DEVELOPMENT_CARRIER_DIAGNOSTIC":
        return {"terminal": terminal, "scored_rows": len(read(trial / "scores.jsonl")) if (trial / "scores.jsonl").exists() else 0,
                "captured_donors": len(read(trial / "donors.jsonl")) if (trial / "donors.jsonl").exists() else 0,
                "semantic_result": None}
    scores, generations, donors = (read(trial / f) for f in ("scores.jsonl", "generations.jsonl", "donors.jsonl"))
    result = {}
    for route in sorted({r["route"] for r in scores}):
        by_control = defaultdict(lambda: defaultdict(list))
        by_question = defaultdict(lambda: defaultdict(list))
        paired = defaultdict(dict)
        counts = defaultdict(lambda: {"attempts": 0, "correct": 0, "ties": 0})
        for row in scores:
            if row["route"] != route:
                continue
            delta = row["scores"]["A"]["logprob"] - row["scores"]["B"]["logprob"]
            prediction = None if delta == 0 else ("A" if delta > 0 else "B")
            gold = row["plus_label"] if row["pole"] == "plus" else ("B" if row["plus_label"] == "A" else "A")
            correct = prediction == gold
            assert row["prediction"] == prediction and row["gold"] == gold and row["correct"] == correct
            control = row["control"]
            by_control[control][row["group"]].append(correct)
            by_question[(control, row["question_variant"])][row["group"]].append(correct)
            counts[control]["attempts"] += 1
            counts[control]["correct"] += correct
            counts[control]["ties"] += delta == 0
            if control == "carrier":
                paired[(row["group"], row["phrasing"], row["question_variant"], row["plus_label"])][row["pole"]] = delta if row["plus_label"] == "A" else -delta
        metrics = {}
        for control, groups in sorted(by_control.items()):
            per_group = {name: math.fsum(values) / len(values) for name, values in sorted(groups.items())}
            metrics[control] = {**counts[control], "group_macro_accuracy": math.fsum(per_group.values()) / len(per_group), "per_group": per_group}
        best_null = max(metrics[k]["group_macro_accuracy"] for k in ("uninjected", "zero", "random", "unrelated"))
        accuracy = metrics["carrier"]["group_macro_accuracy"]
        gaps = [p["plus"] - p["minus"] for p in paired.values()]
        assert all(set(p) == {"plus", "minus"} for p in paired.values())
        text = [r for r in generations if r["route"] == route]
        result[route] = {"controls": metrics, "best_fixed_null_accuracy": best_null,
            "carrier_minus_best_null": accuracy - best_null,
            "development_screen_pass": accuracy >= .8 and accuracy - best_null >= .1,
            "source_states": len([r for r in donors if r["route"] == route]),
            "distinct_source_state_hashes": len({r["state_sha256"] for r in donors if r["route"] == route}),
            "paired_pole_gaps": {"total": len(gaps), "positive": sum(g > 0 for g in gaps), "ties": sum(g == 0 for g in gaps), "mean_nats": math.fsum(gaps) / len(gaps)},
            "question_macro_accuracy": {str(q): math.fsum(math.fsum(v) / len(v) for v in gs.values()) / len(gs)
                                        for (control, q), gs in by_question.items() if control == "carrier"},
            "verbalizations": {"total": len(text), "distinct_strings": len({r["text"] for r in text}),
                               "empty": sum(not r["text"] for r in text), "censored": sum(r["censored"] for r in text),
                               "eos": sum(r["stop"] == "eos" for r in text)}}
    return {"terminal": json.loads((trial / "terminal.json").read_text()),
            "native_parity": json.loads((trial / "native_parity.json").read_text()),
            "neutral_replays": {"total": len(read(trial / "neutral.jsonl")), "all_exact": all(r["exact"] for r in read(trial / "neutral.jsonl"))},
            "routes": result}


if __name__ == "__main__":
    print(json.dumps({p.name: summarize(p) for p in sorted((ROOT / "trials").iterdir()) if p.is_dir()},
                     ensure_ascii=False, sort_keys=True, indent=2))
