"""The curriculum's stage texts and later questions share one counterbalanced truth ledger."""

from collections import Counter
import itertools
import json
from pathlib import Path

import pytest

from selffold.conditional_curriculum import (
    STAGES,
    action_index,
    make_cases,
    tasks_for,
    world_text,
)
from selffold.records import file_sha

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/conditional_curriculum_20261103.json"
ACTIONS = ROOT / "data/within_topic_wording_20261103.json"


def material():
    return json.loads(DATA.read_text()), json.loads(ACTIONS.read_text())["groups"]


def test_curriculum_seals_new_flags_wording_and_distinct_stage_order_units():
    fixture, groups = material()
    assert fixture["action_fixture_sha256"] == file_sha(ACTIONS)
    original = json.loads((ROOT / "data/conditional_audit_20261103.json").read_text())
    prior_flags = {flag for g in original["groups"].values() for flag in g["flags"]}
    fitting_flags = {f for pair in fixture["fit"]["flag_pairs"] for f in pair}
    fresh_flags = {f for pair in fixture["fresh"]["flag_pairs"] for f in pair}
    assert len(fitting_flags) == 6 and len(fresh_flags) == 4
    assert fitting_flags.isdisjoint(fresh_flags)
    assert prior_flags.isdisjoint(fitting_flags | fresh_flags)
    assert set(fixture["fit"]["styles"]).isdisjoint(fixture["fresh"]["styles"])
    assert fixture["fit"]["question"] != fixture["fresh"]["question"]
    banks = {s: list(make_cases(groups, fixture, s)) for s in ("fit", "fresh")}
    assert len(banks["fit"]) == 480 and len(banks["fresh"]) == 160
    assert {c["source"] for c in banks["fit"]}.isdisjoint(
        c["source"] for c in banks["fresh"]
    )
    for split, bank in banks.items():
        assert (
            len(bank)
            == len({c["key"] for c in bank})
            == len({c["source"] for c in bank})
        )
        expected = (
            {"source": 192, "cases": 192, "rule": 96}
            if split == "fit"
            else {"source": 64, "cases": 64, "rule": 32}
        )
        assert Counter(c["stage"] for c in bank) == expected
        index = {
            (
                c["topic"],
                c["flag_pair"],
                c["style"],
                c["stage"],
                c["pole"],
                c["source_order"],
            ): c
            for c in bank
        }
        for c in bank:
            assert (
                "Current indicators:" not in c["source"]
                and "Observed indicators" not in c["source"]
            )
            assert c["flags"] == fixture[split]["flag_pairs"][c["flag_pair"]]
            if c["stage"] == "rule":
                assert c["source_order"] == 0
            else:
                k = (
                    c["topic"],
                    c["flag_pair"],
                    c["style"],
                    c["stage"],
                    c["pole"],
                    1 - c["source_order"],
                )
                assert index[k]["source"] != c["source"]


def test_every_stage_covers_four_worlds_opposed_policies_and_both_label_orders():
    fixture, groups = material()
    assert fixture["worlds"] == [
        list(w) for w in itertools.product((False, True), repeat=2)
    ]
    for split in ("fit", "fresh"):
        bank = list(make_cases(groups, fixture, split))
        counts = Counter()
        by_key = {c["key"]: c for c in bank}
        for c in bank:
            g = groups[c["topic"]]
            style = fixture["styles"][c["style"]]
            truth = []
            if c["stage"] == "cases":
                table = [
                    style["case"].format(
                        world=world_text(c["flags"], w),
                        action=g["options"][action_index(c["pole"], w)],
                    )
                    for w in fixture["worlds"]
                ]
                assert c["source"].endswith(
                    " ".join(table if c["source_order"] == 0 else table[::-1])
                )
            for w, world in enumerate(fixture["worlds"]):
                pair = tasks_for(c, g, fixture, w)
                expected = c["pole"] if w == 3 else 1 - c["pole"]
                assert [t["gold"] for t in pair] == (
                    ["A", "B"] if expected == 0 else ["B", "A"]
                )
                assert pair[0]["source"] == pair[1]["source"] == c["source"]
                assert (
                    pair[0]["question"].split("\nA =", 1)[0]
                    == pair[1]["question"].split("\nA =", 1)[0]
                )
                assert world_text(c["flags"], world) in pair[0]["question"]
                assert (
                    g["options"][0] in pair[0]["question"]
                    and g["options"][1] in pair[0]["question"]
                )
                assert all(t["key"] in by_key for t in pair)
                truth.append(expected)
                counts.update(t["gold"] for t in pair)
            assert len(set(truth)) == 2 and truth.count(c["pole"]) == 1
        assert counts == {"A": len(bank) * 4, "B": len(bank) * 4}
    for invalid in ("train", "test", "qualification"):
        with pytest.raises(ValueError, match="Unknown"):
            list(make_cases(groups, fixture, invalid))
    with pytest.raises(ValueError):
        action_index(0, [True, 1])
