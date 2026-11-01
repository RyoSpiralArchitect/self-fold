"""Reproduce the localization and context-swap tables from paired raw label scores."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sign(value):
    return int(value > 0) - int(value < 0)


def signed_label(scores):
    return scores["A"] - scores["B"]


def panel_metrics(records, select):
    per_topic = {}
    signs_correct = native_agreement = donor_ties = 0
    for topic in sorted({r["group"] for r in records}):
        group = [r for r in records if r["group"] == topic]
        assert {(r["pole"], r["plus_label"]) for r in group} == {
            (p, o) for p in ("plus", "minus") for o in ("A", "B")} and len(group) == 4
        if any(select(r) is None for r in group):
            continue
        margins = {}
        for pole in ("plus", "minus"):
            donors = [r for r in group if r["pole"] == pole]
            def margin(get):
                return mean(signed_label(get(r)) * (1 if r["plus_label"] == "A" else -1) for r in donors)
            m = margin(select)
            native = margin(lambda r: r["reference"]["native"]["scores"])
            full = margin(lambda r: r["reference"]["suffix"]["scores"])
            margins[pole] = (m, full)
            signs_correct += sign(m) == (1 if pole == "plus" else -1)
            native_agreement += sign(m) == sign(native)
            donor_ties += m == 0
        per_topic[topic] = {"gap": margins["plus"][0] - margins["minus"][0],
                            "reference_gap": margins["plus"][1] - margins["minus"][1]}
    gap = sum(t["gap"] for t in per_topic.values())
    den = sum(t["reference_gap"] for t in per_topic.values())
    loo = [(gap-t["gap"])/(den-t["reference_gap"]) for t in per_topic.values() if den-t["reference_gap"] > 0]
    return {"topics": len(per_topic), "measured_orders": sum(select(r) is not None for r in records),
            "positive_gap": sum(t["gap"] > 0 for t in per_topic.values()),
            "tied_gap": sum(t["gap"] == 0 for t in per_topic.values()),
            "donor_ties": donor_ties, "correct": signs_correct, "native_sign_agreement": native_agreement,
            "gap_sum_nats": gap, "reference_sum_nats": den, "fraction": gap/den if den > 0 else None,
            "leave_one_topic_out": [min(loo), max(loo)] if loo else None,
            "median_gap_nats": median(t["gap"] for t in per_topic.values()) if per_topic else None,
            "per_topic": per_topic}


def context_scores(row, name, window, method):
    result = row["contexts"][name]["windows"].get(window, {})
    return result.get(method) if result.get("status") == "measured" else None


def mapping_conflicts(records, window, method):
    tally = Counter()
    for row in records:
        scores = context_scores(row, "reversed", window, method)
        if scores is None:
            continue
        recipient = sign(signed_label(row["reference"]["native"]["scores"]))
        source = sign(signed_label(row["contexts"]["reversed"]["source_native_scores"]))
        if recipient * source != -1:
            continue
        choice = sign(signed_label(scores))
        tally["source" if choice == source else "recipient" if choice == recipient else "tie"] += 1
    return {k: tally[k] for k in ("source", "recipient", "tie")}


def summarize(trial):
    terminal = read(trial/"terminal.json")
    source = read(trial/"source.json")
    req = read(trial/"request.json")
    assert source["git_status"] == "" and source["git_commit"]
    result = {"source_commit": source["git_commit"], "status": terminal["status"],
              "model": req["model"], "panel": req["panel"], "study": req["study"]}
    if result["status"] != "COMPLETED_DEVELOPMENT_PROBE":
        return result
    records = rows(trial/"interventions.jsonl")
    layouts = rows(trial/"prompts.jsonl")
    fixture = "slot_contrasts.dev.json" if req["panel"] == "original" else "span_topics.prospective.json"
    topics = {g["id"] for g in read(REPO/"data"/fixture)["groups"]}
    assert len(records) == 24 and len(layouts) == 12 and {r["group"] for r in records} == topics
    settings = read(trial/"span_settings.json")
    by_order = {(p["group"], p["plus_label"]): p for p in layouts}
    for r in records:
        p = by_order[r["group"], r["plus_label"]]
        bounds = [p["windows"][k] for k in settings["partition"]]
        assert bounds[0][0] == p["slot_positions"][-1]+1 and bounds[-1] == [len(p["recipient_ids"])-2, len(p["recipient_ids"])-1]
        assert all(a[1] == b[0] for a,b in zip(bounds, bounds[1:]))
        assert r["reference"]["neutral_replay_max_abs"] == r["reference"]["slots_and_suffix"]["max_abs_vocab_from_native"] == 0
    keys = records[0]["regions"]
    result["reference"] = {k: panel_metrics(records, lambda r: r["reference"][k]["scores"])
                           for k in ("native", "neutral", "suffix")}
    result["regions"] = {}
    for key in keys:
        widths = [p["windows"][key][1]-p["windows"][key][0] for p in layouts]
        result["regions"][key] = {"width_range": [min(widths), max(widths)],
                                  **{mode: panel_metrics(records, lambda r: r["regions"][key][mode]["scores"])
                                     for mode in ("only", "omit", "native_knockout")}}
    result["contexts"] = {}
    for name in records[0]["contexts"]:
        result["contexts"][name] = {}
        for win in settings["context_windows"]:
            result["contexts"][name][win] = {}
            for method in ("raw", "delta"):
                m = panel_metrics(records, lambda r: context_scores(r, name, win, method))
                if name == "reversed":
                    m["native_letter_conflicts"] = mapping_conflicts(records, win, method)
                result["contexts"][name][win][method] = m
    result["max_pre_question_abs"] = {k: max((r["contexts"][k]["pre_question_max_abs"] for r in records
                                             if r["contexts"][k]["status"]=="matched_length"), default=None)
                                      for k in records[0]["contexts"]}
    result["matched_raw_delta_max_abs_vocab"] = max(w["raw_delta_max_abs_vocab"] for r in records
                                                    for w in r["contexts"]["same"]["windows"].values())
    if req["panel"] == "original":
        family = "mistral" if req["model"] == "mistral7b" else "gemma"
        previous = REPO/"provenance/slot_content_20261101/trials"/f"replay-{family}"
        old = {(r["group"],r["pole"],r["plus_label"]):r for r in rows(previous/"interventions.jsonl")}
        result["archived_score_max_abs"] = max(
            abs(r["reference"][new]["scores"][label] - old[r["group"],r["pole"],r["plus_label"]]["scores"][prior][label])
            for r in records for label in ("A","B") for new,prior in (("native","native_words"),("suffix","suffix_only")))
    if req["study"] == "clauses":
        family = "mistral" if req["model"] == "mistral7b" else "gemma"
        first_stage = HERE/"trials"/f"{req['panel']}-{family}"
        old = {(r["group"],r["pole"],r["plus_label"]): r for r in rows(first_stage/"interventions.jsonl")}
        result["repeated_window_score_max_abs"] = max(
            abs(r["regions"][win][mode]["scores"][label] -
                old[r["group"],r["pole"],r["plus_label"]]["regions"][win][mode]["scores"][label])
            for r in records for win in ("score","closing") for mode in ("only","omit","native_knockout") for label in ("A","B"))
        result["punctuation_also_in_phrase"] = {
            k: sum(p["windows"][k+"_words"][1] > p["windows"][k+"_punct"][0] for p in layouts) for k in ("A","B")}
        result["same_A_B_punctuation_id_orders"] = sum(
            p["recipient_ids"][p["windows"]["A_punct"][0]] == p["recipient_ids"][p["windows"]["B_punct"][0]] for p in layouts)
    return result


if __name__ == "__main__":
    names = [f"{panel}-{model}" for panel in ("original", "prospective") for model in ("mistral", "gemma")]
    names += ["clauses-"+name for name in names]
    print(json.dumps({name: summarize(HERE/"trials"/name) for name in names}, indent=2, ensure_ascii=False))
