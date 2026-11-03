from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest

from selffold.matched_cues import (
    HEADERS,
    header_geometry,
    preference_records,
    question_groups,
)
from selffold.records import digest, file_sha

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "data/matched_cues_20261103.json").read_text())
ACTIONS = ROOT / "data/within_topic_wording_20261103.json"
GROUPS = json.loads(ACTIONS.read_text())["groups"]
spec = spec_from_file_location(
    "cue_scores", ROOT / "provenance/matched_cues_20261103/summarize.py"
)
scores = module_from_spec(spec)
spec.loader.exec_module(scores)


def test_cue_design_matches_questions_actions_roles_and_all_eight_topics():
    assert file_sha(ACTIONS) == FIXTURE["action_fixture_sha256"]
    prior = json.loads((ROOT / "data/teacher_polarity_20261103.json").read_text())
    exposed = (
        {q for g in GROUPS for q in g["questions"]}
        | set(prior["generic_questions"])
        | {q for qs in prior["specific_questions"].values() for q in qs}
    )
    assert exposed.isdisjoint(FIXTURE["questions"])
    for i, g in enumerate(question_groups(GROUPS, FIXTURE)):
        for q in range(4):
            bare, topic, placebo, rotated = g["fresh_questions"][q * 4 : q * 4 + 4]
            assert bare == FIXTURE["questions"][q]
            assert [v.split("\n", 1) for v in (topic, placebo, rotated)] == [
                [w, bare]
                for w in (
                    FIXTURE["topic_cues"][g["id"]],
                    FIXTURE["placebo"],
                    FIXTURE["topic_cues"][GROUPS[(i + 1) % 8]["id"]],
                )
            ]
    sources = list(preference_records(GROUPS, FIXTURE))
    assert len(sources) == len({r["source"] for r in sources}) == 512
    old = {
        json.loads(l)["source_sha256"]
        for l in (ROOT / "provenance/teacher_polarity_20261103/run/sources.jsonl")
        .read_text()
        .splitlines()
    }
    assert old.isdisjoint({digest(c["source"]) for c in sources})
    idx = {
        (c["topic"], c["pair"], c["family"], c["pole"], c["order"], c["position"]): c[
            "source"
        ]
        for c in sources
    }
    for (t, w, f, p, o, pos), v in idx.items():
        if pos == "before":
            clauses = v[:-1].split(". ")
            assert (
                len(clauses) == 2
                and idx[t, w, f, p, 1 - o, pos][:-1].split(". ") == clauses[::-1]
            )
            actions = (GROUPS[t]["plus"][w], GROUPS[t]["minus"][w])
            expected = []
            for role in (o, 1 - o):
                cue = FIXTURE["verdicts"][f][role]
                action = actions[p ^ role]
                assert f"[ {cue} ] [ {action} ]" in clauses[len(expected)]
                expected.append(f"My preference record: [ {action} ] [ {cue} ].")
            assert " ".join(expected) == idx[t, w, f, p, o, "after"]
            assert Counter(v.split()) == Counter(idx[t, w, f, 1 - p, o, pos].split())


def test_geometry_guard_and_paired_denominators():
    assert (
        header_geometry([[1, 9, 7], [1, 4, 5, 9, 7], [1, 3, 5, 9, 7], [1, 2, 5, 9, 7]])[
            "header_index"
        ]
        == 1
    )
    with pytest.raises(ValueError):
        header_geometry([[1, 9, 7], [1, 4, 5, 9, 7], [1, 3, 5, 9, 7], [1, 2, 5, 6, 7]])
    first = [{"topic": i, "ok": v} for i, v in enumerate([True, True, False, False])]
    second = [
        dict(r, ok=v) for r, v in zip(first, [True, False, True, False], strict=True)
    ]
    p = scores.paired(first, second[::-1], ("topic",), lambda r: r["ok"])
    assert [
        p[k] for k in ("both", "first_only", "second_only", "neither", "net", "n")
    ] == [1, 1, 1, 1, 0, 4]
    with pytest.raises(ValueError):
        scores.paired(first, second[:-1], ("topic",), lambda r: r["ok"])


def test_correct_policy_can_contain_two_wrong_signs():
    row = {
        "topic": 0,
        "pole": 0,
        "order": 0,
        "factor": 0,
        "scores": {"topic": 0, "role": -1.0, "first_action": [-1.0]},
    }
    assert scores.policy_right(row) and not scores.both_right(row)
