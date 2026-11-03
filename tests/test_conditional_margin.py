"""The margin study preserves exposed fitting inputs and seals new conditional evaluation text."""

from collections import Counter
import json
from pathlib import Path

from selffold.conditional_curriculum import (
    action_index,
    make_cases,
    tasks_for,
    world_text,
)
from selffold.records import file_sha

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/conditional_curriculum_20261103.json"
NEW = ROOT / "data/conditional_margin_20261103.json"
ACTIONS = ROOT / "data/within_topic_wording_20261103.json"


def test_margin_fixture_preserves_every_fitting_source_and_native_question():
    old, new = (json.loads(p.read_text()) for p in (OLD, NEW))
    groups = json.loads(ACTIONS.read_text())["groups"]
    assert (
        old["action_fixture_sha256"]
        == new["action_fixture_sha256"]
        == file_sha(ACTIONS)
    )
    assert (
        new["fit"] == old["fit"]
        and new["operations"] == old["operations"]
        and new["worlds"] == old["worlds"]
    )
    left, right = (list(make_cases(groups, f, "fit")) for f in (old, new))
    assert left == right and len(left) == 480
    for a, b in zip(left, right, strict=True):
        for world in range(4):
            assert tasks_for(a, groups[a["topic"]], old, world) == tasks_for(
                b, groups[b["topic"]], new, world
            )


def test_margin_fresh_names_styles_and_queries_are_wholly_unexposed():
    old, new = (json.loads(p.read_text()) for p in (OLD, NEW))
    groups = json.loads(ACTIONS.read_text())["groups"]
    previous_flags = {
        name
        for split in ("fit", "fresh")
        for pair in old[split]["flag_pairs"]
        for name in pair
    }
    original = json.loads((ROOT / "data/conditional_audit_20261103.json").read_text())
    previous_flags |= {name for g in original["groups"].values() for name in g["flags"]}
    fresh_flags = {name for pair in new["fresh"]["flag_pairs"] for name in pair}
    assert len(fresh_flags) == 4 and fresh_flags.isdisjoint(previous_flags)
    assert len(new["fresh"]["styles"]) == 1 and set(new["fresh"]["styles"]).isdisjoint(
        old["styles"]
    )
    assert (
        new["fresh"]["question"] != old["fresh"]["question"] != old["fit"]["question"]
    )
    assert all(
        new["fresh"]["operations"][g["id"]] != old["operations"][g["id"]]
        for g in groups
    )
    prior_cases = [
        c for split in ("fit", "fresh") for c in make_cases(groups, old, split)
    ]
    prior_sources = {c["source"] for c in prior_cases}
    prior_questions = {
        t["question"]
        for c in prior_cases
        for w in range(4)
        for t in tasks_for(c, groups[c["topic"]], old, w)
    }
    fresh = list(make_cases(groups, new, "fresh"))
    assert (
        len(fresh)
        == len({c["source"] for c in fresh})
        == len({c["key"] for c in fresh})
        == 160
    )
    assert {c["source"] for c in fresh}.isdisjoint(prior_sources)
    questions = {
        t["question"]
        for c in fresh
        for w in range(4)
        for t in tasks_for(c, groups[c["topic"]], new, w)
    }
    assert questions.isdisjoint(prior_questions)
    assert Counter(c["stage"] for c in fresh) == {"source": 64, "cases": 64, "rule": 32}


def test_margin_fresh_stage_ledger_is_balanced_and_world_dependent():
    new = json.loads(NEW.read_text())
    groups = json.loads(ACTIONS.read_text())["groups"]
    fresh = list(make_cases(groups, new, "fresh"))
    verdicts = Counter()
    idx = {
        (c["topic"], c["flag_pair"], c["stage"], c["pole"], c["source_order"]): c
        for c in fresh
    }
    for c in fresh:
        g = groups[c["topic"]]
        style = new["styles"][c["style"]]
        assert "present situation" not in c["source"]
        if c["stage"] == "rule":
            assert c["source_order"] == 0
        else:
            opposite_order = idx[
                c["topic"], c["flag_pair"], c["stage"], c["pole"], 1 - c["source_order"]
            ]
            assert opposite_order["source"] != c["source"]
        if c["stage"] == "cases":
            table = [
                style["case"].format(
                    world=world_text(c["flags"], w),
                    action=g["options"][action_index(c["pole"], w)],
                )
                for w in new["worlds"]
            ]
            assert c["source"].endswith(
                " ".join(table if c["source_order"] == 0 else table[::-1])
            )
        answers = []
        other = idx[
            c["topic"], c["flag_pair"], c["stage"], 1 - c["pole"], c["source_order"]
        ]
        for world in range(4):
            pair = tasks_for(c, g, new, world)
            flipped = tasks_for(other, g, new, world)
            answer = c["pole"] if world == 3 else 1 - c["pole"]
            assert (
                pair[0]["gold"] == ("A" if answer == 0 else "B")
                and pair[0]["gold"] != pair[1]["gold"]
            )
            assert [t["question"] for t in pair] == [t["question"] for t in flipped]
            assert all(
                a["gold"] != b["gold"] for a, b in zip(pair, flipped, strict=True)
            )
            assert all(t["source"] == c["source"] for t in pair)
            assert new["fresh"]["operations"][c["group"]] in pair[0]["question"]
            answers.append(answer)
            verdicts.update(t["gold"] for t in pair)
        assert len(set(answers)) == 2 and answers.count(c["pole"]) == 1
    assert verdicts == {"A": 640, "B": 640}
