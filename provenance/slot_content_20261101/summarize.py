"""Recompute the archived-pair and six-contrast tables from the raw label likelihoods."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median


HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE.parents[1] / "data/slot_contrasts.dev.json").read_text())
ROUTES = ("text", "E", "G", "B", "G_reverse", "G_magnitude")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def recompute(example, fixtures):
    group = fixtures[example["group"]]
    plus, minus = group["options"]
    margins = []
    assert len(example["orders"]) == 2
    for order, (plus_label, first, second) in zip(example["orders"], (("A", plus, minus), ("B", minus, plus))):
        assert order["plus_label"] == plus_label
        assert order["question"] == f"{group['question']}\nA = {first}. B = {second}. Use the supplied description."
        a, b = (order["scores"][k] for k in ("A", "B"))
        assert a["tokens"] == b["tokens"] == 1
        margin = (a["logprob"] - b["logprob"]) * (1 if plus_label == "A" else -1)
        assert margin == order["plus_margin"]
        margins.append(margin)
    result = mean(margins)
    assert result == example["plus_margin"]
    return result


def route_stats(margins, route, groups, phrasings=2):
    correct = tied = 0
    all_gaps = []
    straddles = 0
    per_group = {}
    for group in groups:
        gaps, opposite, same = [], [], []
        for p in range(phrasings):
            pos = margins.get((group, "plus", p, route))
            neg = margins.get((group, "minus", p, route))
            for value, sign in ((pos, 1), (neg, -1)):
                if value is not None:
                    correct += sign * value > 0
                    tied += value == 0
            if pos is not None and neg is not None:
                gaps.append(pos - neg)
                opposite.append(abs(pos - neg))
                straddles += pos > 0 > neg
        for pole in ("plus", "minus"):
            pair = [margins.get((group, pole, p, route)) for p in (0, 1)]
            if None not in pair:
                same.append(abs(pair[0] - pair[1]))
        all_gaps.extend(gaps)
        per_group[group] = {"pole_gaps": gaps, "mean_absolute_opposite": mean(opposite) if opposite else None,
                            "mean_absolute_paraphrase": mean(same) if same else None}
    return {"ordered": sum(g > 0 for g in all_gaps), "tied_pairs": all_gaps.count(0),
            "pairs_observed": len(all_gaps), "pairs_planned": phrasings * len(groups),
            "all_phrasings_ordered_groups": sum(len(g["pole_gaps"]) == phrasings and min(g["pole_gaps"]) > 0
                                                for g in per_group.values()),
            "both_donor_signs_correct_pairs": straddles, "correct_donor_signs": correct,
            "tied_donor_signs": tied, "donors_observed": sum(k[3] == route for k in margins),
            "median_signed_pole_gap": median(all_gaps) if all_gaps else None, "groups": per_group}


def replay_stats(trial, groups):
    neutral = rows(trial / "neutral.jsonl")
    interventions = rows(trial / "interventions.jsonl")
    earlier = rows((trial.parent / trial.name.replace("replay-", "alignment-")) / "scores.jsonl")
    reference = {(r["group"], r["pole"], r["route"], o["plus_label"]): o["scores"] for r in earlier for o in r["orders"]}
    assert len(neutral) == 12 and len(interventions) == 24
    routes = ("native_words", "slots_only", "suffix_only", "slots_and_suffix")
    orders = {}
    old_diffs = []
    for row in interventions:
        assert row["max_abs_logit_difference_from_native"]["native_words"] == 0
        for route in routes:
            scores = row["scores"][route]
            margin = (scores["A"] - scores["B"]) * (1 if row["plus_label"] == "A" else -1)
            orders.setdefault((row["group"], row["pole"], 0, route), {})[row["plus_label"]] = margin
        old = reference[row["group"], row["pole"], "native_words", row["plus_label"]]
        old_diffs.extend(abs(row["scores"]["native_words"][k] - old[k]["logprob"]) for k in ("A", "B"))
    for row in neutral:
        old = reference[row["group"], None, "uninjected", row["plus_label"]]
        old_diffs.extend(abs(row["native_scores"][k] - old[k]["logprob"]) for k in ("A", "B"))
    assert len(orders) == 48 and all(set(o) == {"A", "B"} for o in orders.values())
    margins = {k: mean(o.values()) for k, o in orders.items()}
    checks = {}
    for geo in ("short", "same_length"):
        checks[geo] = {
            "max_abs_label_logprob": max(abs(r[geo]["scores"][k] - r["native_scores"][k]) for r in neutral for k in ("A","B")),
            "max_abs_slot_difference": max(r[geo]["max_abs_slot_difference"] for r in neutral),
            "max_abs_vocab_logit": max(r[geo]["max_abs_logit_difference"] for r in neutral),
        }
    return {"neutral_geometries": checks, "prior_native_label_score_max_abs": max(old_diffs),
            "native_vs_opaque_slot_max_abs": max(r["native_vs_opaque_slot_max_abs"] for r in interventions),
            "full_span_exact_vocab_replays": sum(r["max_abs_logit_difference_from_native"]["slots_and_suffix"] == 0 for r in interventions),
            "same_sign_as_native_out_of_12": {
                r: sum((margins[g, p, 0, r] > 0, margins[g, p, 0, r] < 0) ==
                       (margins[g, p, 0, "native_words"] > 0, margins[g, p, 0, "native_words"] < 0)
                       for g in groups for p in ("plus", "minus")) for r in routes},
            "max_abs_vocab_logit_from_native": {route: max(r["max_abs_logit_difference_from_native"][route] for r in interventions)
                                                for route in routes},
            "donor_orders": len(interventions), "neutral_orders": len(neutral),
            "routes": {r: route_stats(margins, r, groups, 1) for r in routes}}


def summarize(name):
    trial = HERE / "trials" / name
    terminal = json.loads((trial / "terminal.json").read_text())
    src = json.loads((trial / "source.json").read_text())
    request = json.loads((trial / "request.json").read_text())
    assert not src["git_status"] and src["git_commit"]
    result = {"source_commit": src["git_commit"], "model": request["model"], "status": terminal["status"]}
    if terminal["status"] != "COMPLETED_DEVELOPMENT_PROBE":
        return result
    is_archive = request["study"] == "archive"
    aligned = request["study"] == "alignment"
    groups = {g["id"]: g for g in ([DATA["archive"]] if is_archive else DATA["groups"])}
    if request["study"] == "replay":
        result.update(replay_stats(trial, groups))
        return result
    scores = rows(trial / "scores.jsonl")
    margins = {(r["group"], r["pole"], r["phrasing"], r["route"]): recompute(r, groups) for r in scores}
    assert len(margins) == len(scores)
    controls = ("uninjected", "neutral_replay" if aligned else "zero")
    result.update(measurements=len(scores), label_likelihoods=len(scores)*4,
                  null_margins={g: {route: margins[g, None, None, route] for route in controls}
                                for g in groups})
    if is_archive:
        assert len(scores) == 14
        result["donor_margins"] = {route: {pole: margins["caution", pole, 0, route] for pole in ("plus", "minus")}
                                   for route in ("text", "E", "L1", "L2", "B1", "B2")}
        return result
    cases = rows(trial / "cases.jsonl")
    if aligned:
        assert len(cases) == 12 and len(scores) == 60
        result["routes"] = {r: route_stats(margins, r, groups, 1)
                            for r in ("native_words", "E_words", "aligned", "aligned_reverse")}
        nulls = {(r["group"], r["route"]): r["orders"] for r in scores if r["pole"] is None}
        result["neutral_replay_max_abs_logprob_difference"] = max(
            abs(nulls[g, "uninjected"][i]["scores"][label]["logprob"] -
                nulls[g, "neutral_replay"][i]["scores"][label]["logprob"])
            for g in groups for i in (0, 1) for label in ("A", "B"))
        result["native_phrase_token_counts"] = {c["case"]: len(c["token_ids"]) for c in cases}
        return result
    assert len({c["case"] for c in cases}) == len(cases) == 24
    missing_b = sum(not c["roundtrip_available"] for c in cases)
    assert len(scores) == 156 - missing_b
    result["missing_roundtrips"] = missing_b
    result["distinct_stored_states"] = {route: len({r["state_sha256"] for r in scores if r["route"] == route})
                                        for route in ("E", "G", "B")}
    result["max_relative_norm_error"] = max(c["max_relative_slot_norm_error"] for c in cases)
    result["routes"] = {route: route_stats(margins, route, groups) for route in ROUTES}
    generations = rows(trial / "generations.jsonl")
    assert len(generations) == 50
    result["null_generations"] = {r["route"]: r["text"] for r in generations if r["case"] is None}
    result["free_readouts"] = {}
    for route in ("E", "G"):
        reads = {r["case"]: r for r in generations if r["route"] == route}
        assert len(reads) == 24
        result["free_readouts"][route] = {
            "distinct_strings": len({r["text"] for r in reads.values()}),
            "equal_matched_pole_strings": sum(reads[f"{g}-plus-p{p}"]["text"] == reads[f"{g}-minus-p{p}"]["text"]
                                              for g in groups for p in (0, 1)),
            "stops": dict(Counter(r["stop"] for r in reads.values())),
        }
    result["all_stops"] = dict(Counter(r["stop"] for r in generations))
    return result


if __name__ == "__main__":
    studies = ("archive-mistral", "contrasts-mistral", "contrasts-gemma", "alignment-mistral", "alignment-gemma",
               "replay-mistral", "replay-gemma")
    print(json.dumps({name: summarize(name) for name in studies}, indent=2, ensure_ascii=False))
