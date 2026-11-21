"""Tests for source isolation, coordinate binding and the matched decision rule."""
import copy
import json
from pathlib import Path

import pytest

from selffold.lm_writer_reports import fingerprint
from selffold.oracle_row_readout import HEADINGS, ORDERS, WORLD, analyze, requests, score


ROOT = Path(__file__).resolve().parents[1]
DESIGN = json.loads((ROOT / "data/oracle_row_readout_20261122.json").read_text())
GROUPS = json.loads((ROOT / "data/within_topic_wording_20261103.json").read_text())["groups"]


@pytest.fixture(scope="module")
def panel():
    return requests(DESIGN, GROUPS)


def literal_answer(q, heading=None, first=False):
    # Read the *visible* table, independently of the bit-mask gold calculation.
    heading = heading or HEADINGS[q["question"]]
    line = q["prompt"].split(heading + "\n")[1].splitlines()[0]
    entries = dict(entry.split(" -> ") for entry in line.split("; "))
    return next(iter(entries.values())) if first else entries[WORLD[q["row"]]]


def answer(q, text):
    return dict(id=q["id"], request_sha256=fingerprint(q), text=text,
                generation=dict(stop="eos", token_count=3, prompt_tokens=120, budget=48,
                                stop_token_id=107, censored=False))


def test_single_has_only_target_and_matches_full_query_byte_for_byte(panel):
    canonical = {(q["topic"], q["target"], q["rotation"], q["row"], q["question"]): q
                 for q in panel if q["context"] == "single"}
    assert len(canonical) == 768 and len(panel) == 3840
    for q in panel:
        single = canonical[q["topic"], q["target"], q["rotation"], q["row"], q["question"]]
        other = HEADINGS["actual" if q["question"] == "instruction" else "instruction"]
        assert "PREVIOUS TABLE:\n" not in single["prompt"] and other + "\n" not in single["prompt"]
        assert single["prompt"].count(" -> ") == 4
        assert single["prompt"].split("Question: ")[1] == q["prompt"].split("Question: ")[1]
        assert literal_answer(single) == literal_answer(q)
        if q["context"] in ORDERS:
            assert q["prompt"].count(" -> ") == 12
            assert (q["prompt"].index(HEADINGS["instruction"]) < q["prompt"].index(HEADINGS["actual"])) == (q["context"] == ORDERS[0])


def test_visible_table_reader_passes_and_has_expected_denominators(panel):
    perfect = [answer(q, literal_answer(q)) for q in panel]
    summary, _ = analyze(DESIGN, panel, perfect)
    assert summary["decision"] == "both_contexts_reliable_on_panel"
    assert summary["full_both_questions_both_orders"] == [384, 384]
    for p in summary["profiles"].values():
        assert p["metrics"]["overall"] == [768, 768]
        assert p["metrics"]["all_four_rows"] == [192, 192]
        assert p["metrics"]["both_sides_of_swap"] == [768, 768]
        assert p["passed"]


def test_printed_first_row_control_cannot_pass_oracle(panel):
    data = [answer(q, literal_answer(q, first=True)) for q in panel]
    summary, _ = analyze(DESIGN, panel, data)
    oracle = summary["profiles"]["single/single"]["metrics"]
    assert oracle["overall"] == [384, 768] and oracle["all_four_rows"] == [0, 192]
    assert summary["decision"] == "row_readout_first"


@pytest.mark.parametrize("wrong_orders,expected", [
    (set(ORDERS), "investigate_competing_records"),
    ({ORDERS[0]}, "resolve_order_or_context_dependence"),
])
def test_pairing_counts_once_per_order_and_detects_other_record(panel, wrong_orders, expected):
    data = []
    for q in panel:
        other = HEADINGS["actual" if q["question"] == "instruction" else "instruction"]
        text = literal_answer(q, other) if q["context"] in wrong_orders else literal_answer(q)
        data.append(answer(q, text))
    summary, _ = analyze(DESIGN, panel, data)
    assert summary["decision"] == expected
    assert summary["single_both_questions"] == [384, 384]
    for layout in wrong_orders:
        pairs = summary["matched_conflicts"][layout]
        assert pairs["single_only"] == 384 and pairs["net_gain"] == [384, 384]
        assert summary["profiles"][layout + "/agree"]["passed"]


def test_capped_correct_foreign_and_incomplete(panel):
    q = panel[0]
    a = answer(q, literal_answer(q))
    a["generation"].update(stop="length", token_count=48, censored=True, stop_token_id=None)
    assert score(q, a, 48, 1024)["correct"]
    a["text"] = "I think it is " + a["text"]
    assert not score(q, a, 48, 1024)["valid_name"]
    a["request_sha256"] = fingerprint(copy.copy(q) | {"row": 100})
    with pytest.raises(ValueError, match="bound"):
        score(q, a, 48, 1024)
    with pytest.raises(ValueError, match="incomplete"):
        analyze(DESIGN, panel, [a])
