"""Regression controls for compound native-output recipient scoring."""
import importlib.util
from pathlib import Path

import pytest

from selffold.continuity_data import load_groups, question_cases

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("continuity_summary", ROOT / "provenance/continuity_20261103/summarize.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def records():
    rows = []
    for group in load_groups(ROOT / "data/continuity_20261103", "dev"):
        for pole, details in group["poles"].items():
            questions = question_cases(group, pole)
            for variant in details["variants"]:
                for question in questions:
                    for stage in summary.STAGES:
                        for route in summary.ROUTES:
                            native = question["gold"] if route == "matched" else "A"
                            rows.append({"candidate": "test", "route": route, "stage": stage,
                                "case": variant["id"], "group": group["id"], "pole": pole,
                                "frame": variant["frame"], "source_order": variant["order"],
                                "world": question["world_index"], "wording": question["wording"],
                                "label_order": question["order"], "gold": question["gold"],
                                "A": 2. if native == "A" else 1., "B": 2. if native == "B" else 1.,
                                "greedy_token": 10 if native == "A" else 20, "greedy_label": native,
                                "state_sha256": "a" * 64})
    return rows


@pytest.fixture(scope="module")
def complete():
    return records()


def test_native_panel_and_compound_require_both_poles_and_orders(complete):
    result = summary.score_candidate(complete, "test")
    assert result["logical_records"] == 82944
    good = result["cells"]["matched"]["stage2"]["all"]
    assert (good["n"], good["native_correct"], good["compound_n"], good["native_compound_correct"]) == (4608, 4608, 1152, 1152)
    empty = result["cells"]["empty"]["stage2"]["all"]
    assert empty["native_accuracy"] == .5 and empty["native_compound_accuracy"] == 0
    assert result["recipient_screen"]["pass"]
    assert result["recipient_screen"]["endpoint"] == "teacher_encoded_stage2_native_recipient_only"
    for frame in summary.FRAMES:
        assert result["cells"]["matched"]["stage2"]["frames"][frame]["n"] == 1536


def test_forced_scores_can_be_right_while_native_output_fails(complete):
    changed = [{**row, "greedy_token": 99, "greedy_label": None} if row["route"] == "matched" else row for row in complete]
    result = summary.score_candidate(changed, "test")
    cell = result["cells"]["matched"]["stage2"]["all"]
    assert cell["forced_accuracy"] == 1 and cell["native_accuracy"] == cell["native_coverage"] == 0
    assert not result["recipient_screen"]["pass"]


def test_incomplete_duplicate_and_wrong_gold_panels_are_rejected(complete):
    with pytest.raises(ValueError, match="Incomplete"):
        summary.score_candidate(complete[1:], "test")
    with pytest.raises(ValueError, match="Duplicate"):
        summary.score_candidate(complete + complete[:1], "test")
    wrong = [{**complete[0], "gold": "B" if complete[0]["gold"] == "A" else "A"}, *complete[1:]]
    with pytest.raises(ValueError, match="Gold"):
        summary.score_candidate(wrong, "test")


def test_questions_cannot_change_the_source_hash(complete):
    changed = [{**complete[0], "state_sha256": "b" * 64}, *complete[1:]]
    with pytest.raises(ValueError, match="Task-dependent"):
        summary.score_candidate(changed, "test")
