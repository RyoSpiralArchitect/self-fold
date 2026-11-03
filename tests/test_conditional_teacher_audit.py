"""Formal truth, distinct stage states and counterbalanced native questions."""

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from conditional_teacher_audit import (
    DATA,
    ACTIONS,
    action_index,
    audit_cases,
    tasks_for,
    world_text,
)
from selffold.records import file_sha


def test_conditional_audit_has_distinct_stages_and_all_worlds_independent_of_order():
    fixture = json.loads(DATA.read_text())
    groups = json.loads(ACTIONS.read_text())["groups"]
    assert fixture["action_fixture_sha256"] == file_sha(ACTIONS)
    cases = list(audit_cases(groups, fixture))
    assert (
        len(cases)
        == len({c["source"] for c in cases})
        == len({c["key"] for c in cases})
        == 80
    )
    assert Counter(c["stage"] for c in cases) == {"source": 32, "cases": 32, "rule": 16}
    assert Counter((c["stage"], c["group"]) for c in cases) == {
        (s, g["id"]): (2 if s == "rule" else 4)
        for s in ("source", "cases", "rule")
        for g in groups
    }
    for c in cases:
        g = groups[c["topic"]]
        flags = fixture["groups"][g["id"]]["flags"]
        assert "Current indicators:" not in c["source"]
        if c["stage"] == "cases":
            table = [
                f"When {world_text(flags, w)}, I choose {g['options'][action_index(c['pole'], w)]}."
                for w in fixture["worlds"]
            ]
            expected = table if c["source_order"] == 0 else table[::-1]
            assert c["source"].endswith(" ".join(expected))
        if c["stage"] == "rule":
            assert c["source_order"] == 0
            # Canonical plus rule uses the conjunction; minus uses its exact complement.
            assert (
                ("exactly when (" + flags[0] + " is on and " + flags[1] + " is on)")
                in c["source"]
            ) == (c["pole"] == 0)


def test_conditional_questions_require_world_dependent_actions_and_reverse_gold_labels():
    f = json.loads(DATA.read_text())
    groups = json.loads(ACTIONS.read_text())["groups"]
    cases = list(audit_cases(groups, f))
    count = 0
    for c in cases:
        answers = []
        for w in range(4):
            tasks = tasks_for(c, groups[c["topic"]], f, w)
            assert len(tasks) == 2 and tasks[0]["gold"] != tasks[1]["gold"]
            assert (
                tasks[0]["question"].split("\nA =", 1)[0]
                == tasks[1]["question"].split("\nA =", 1)[0]
            )
            assert tasks[0]["source"] == tasks[1]["source"] == c["source"]
            expected = c["pole"] if w == 3 else 1 - c["pole"]
            assert tasks[0]["gold"] == ("A" if expected == 0 else "B")
            answers.append(expected)
            count += len(tasks)
        assert len(set(answers)) == 2 and answers.count(c["pole"]) == 1
    assert count == 640
