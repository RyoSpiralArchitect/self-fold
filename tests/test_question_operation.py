"""Verify the whole-topic operation contrast and its native evaluation denominators."""

from collections import Counter
import json
from pathlib import Path
import sys

import pytest

from selffold import prompts
from selffold.records import digest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import teacher_operation_panel as panel
from fit_teacher_and_polarity import make_tasks

FIXTURE = json.loads((ROOT / "data/question_operation_20261103.json").read_text())
GROUPS = json.loads((ROOT / "data/within_topic_wording_20261103.json").read_text())[
    "groups"
]


def test_operation_questions_keep_the_heading_and_hold_out_whole_question_forms():
    expected_cues = json.loads((ROOT / "data/matched_cues_20261103.json").read_text())[
        "topic_cues"
    ]
    assert FIXTURE["topic_cues"] == expected_cues
    result = panel.question_groups(GROUPS, FIXTURE)
    assert all("fresh_questions" not in g for g in GROUPS)
    prior = json.loads((ROOT / "data/teacher_polarity_20261103.json").read_text())
    exposed = {q for g in GROUPS for q in g["questions"]}
    exposed.update(prior["generic_questions"])
    exposed.update(
        q for questions in prior["specific_questions"].values() for q in questions
    )
    exposed.update(
        json.loads((ROOT / "data/matched_cues_20261103.json").read_text())["questions"]
    )
    new = [q for g in result for q in g["fresh_questions"]]
    assert len(new) == len(set(new)) == 64
    assert exposed.isdisjoint(q.split("\n", 1)[1] for q in new)
    for g in result:
        cue = FIXTURE["topic_cues"][g["id"]]
        concrete = FIXTURE["concrete_operations"][g["id"]]
        generic = FIXTURE["generic_operation"]
        assert all(option not in concrete for option in g["options"])
        for family, scaffold in enumerate(FIXTURE["scaffolds"]):
            a, b = g["fresh_questions"][2 * family : 2 * family + 2]
            assert a == cue + "\n" + scaffold.format(operation=generic)
            assert b == cue + "\n" + scaffold.format(operation=concrete)
            assert a.replace(generic, "{operation}") == b.replace(
                concrete, "{operation}"
            )


def test_question_operation_scores_all_modes_and_routes_with_both_poles_and_labels(
    monkeypatch,
):
    groups = panel.question_groups(GROUPS, FIXTURE)

    class Host:
        def render(self, content):
            return list(content.encode()), {prompts.INPUT_MARKER: list(range(2, 10))}

    class Store:
        def __init__(self):
            self.records = {}

        def append(self, name, row):
            self.records.setdefault(name, []).append(row)

    class PairRecorder:
        def __init__(self, host, adapter, groups, store):
            self.groups = groups

        def score(self, topic, pole, split, wording, state):
            return [
                {k: v for k, v in task.items() if k not in ("source", "question")}
                | {
                    "question_sha256": digest(task["question"]),
                    "injected_sha256": digest(state),
                }
                for task in make_tasks(self.groups, topic, pole, split, wording)
            ]

    monkeypatch.setattr(panel, "NativePairs", PairRecorder)
    store = Store()
    teachers = ["teacher-" + str(i) for i in range(16)]
    panel.evaluate(Host(), None, teachers, "mean", groups, store)
    rows = store.records["teachers.jsonl"]
    assert len(rows) == 512
    assert Counter((r["mode"], r["route"]) for r in rows) == {
        (mode, route): 128 for mode in panel.MODES for route in ("calibrated", "mean")
    }
    assert (
        len(
            {
                (
                    r["topic"],
                    r["family"],
                    r["mode"],
                    r["route"],
                    r["pole"],
                    r["label_order"],
                )
                for r in rows
            }
        )
        == 512
    )
    for row in rows:
        assert row["gold"] == ("A" if row["pole"] == row["label_order"] else "B")
        expected = (
            "mean"
            if row["route"] == "mean"
            else teachers[2 * row["topic"] + row["pole"]]
        )
        assert row["injected_sha256"] == digest(expected)
    layouts = store.records["teacher_question_layout.jsonl"]
    assert len(layouts) == 64
    assert all(
        r["input_slots"][0] == r["input_slots"][1] and len(r["question_sha256"]) == 2
        for r in layouts
    )
    with pytest.raises(ValueError, match="all eight"):
        panel.evaluate(Host(), None, teachers[:-1], "mean", groups, store)
