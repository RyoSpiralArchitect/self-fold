"""Check task separation, label interventions and accounting of a failed first call."""

import importlib.util
import json
from pathlib import Path

from selffold.conditional_curriculum import make_cases, tasks_for
from selffold.two_flags import (
    ALL_NEW,
    PANELS,
    branch_probe,
    panel_fixture,
    training_pairs,
    world_probe,
)

ROOT = Path(__file__).resolve().parents[1]
read = lambda p: json.loads((ROOT / p).read_text())
BASE = read("data/conditional_margin_20261103.json")
DESIGN = read("data/two_flags_20261104.json")
GROUPS = {g["id"]: g for g in read("data/within_topic_wording_20261103.json")["groups"]}
spec = importlib.util.spec_from_file_location(
    "two_flags_score", ROOT / "provenance/two_flags_20261104/summarize.py"
)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def test_crossed_panels_and_intermediate_information_boundaries():
    fixtures = {p: panel_fixture(BASE, DESIGN, p) for p in PANELS}
    assert len(fixtures) == 8 and all(
        f["fit"] == BASE["fit"] for f in fixtures.values()
    )
    previous_names = {
        s
        for f in [BASE, read("data/conditional_curriculum_20261103.json")]
        for split in ("fit", "fresh")
        for pair in f[split]["flag_pairs"]
        for s in pair
    }
    assert not previous_names.intersection(sum(DESIGN["new_flags"], []))
    for panel, f in fixtures.items():
        case = list(make_cases(list(GROUPS.values()), f, "fresh"))
        assert len(case) == len({c["key"] for c in case}) == 160
        assert (f["fresh"]["styles"] == [DESIGN["new_style_name"]]) == (
            "grammar" in panel
        )
        assert (f["fresh"]["flag_pairs"] == DESIGN["new_flags"]) == ("names" in panel)
        assert (f["fresh"]["question"] == DESIGN["question"]) == ("question" in panel)
    f = fixtures[ALL_NEW]
    c = next(make_cases(list(GROUPS.values()), f, "fresh"))
    g = GROUPS[c["group"]]
    for w, values in enumerate(f["worlds"]):
        a = tasks_for(c, g, f, w)
        b = branch_probe(c, g, f, DESIGN["fresh_probes"], all(values))
        other = branch_probe(c, g, f, DESIGN["fresh_probes"], not all(values))
        assert (
            [r["gold"] for r in b]
            == [r["gold"] for r in a]
            == [r["gold"] for r in reversed(other)]
        )
        assert all(
            c["flags"][0] not in r["question"] and c["flags"][1] not in r["question"]
            for r in b
        )
        for kind, ix, truth in [
            ("and", 0, all(values)),
            ("flag", 0, values[0]),
            ("flag", 1, values[1]),
        ]:
            pair = world_probe(c["flags"], w, f, DESIGN["fresh_probes"], kind, ix)
            assert (
                score.decode_branch(pair[0]["gold"], 0)
                == truth
                == score.decode_branch(pair[1]["gold"], 1)
            )
            assert all(
                act not in r["source"] + r["question"]
                for act in g["options"]
                for r in pair
            )
    case = next(make_cases(list(GROUPS.values()), BASE, "fit"))
    for step in range(12):
        for unit in range(2):
            p = [
                training_pairs(case, GROUPS[case["group"]], BASE, DESIGN, step, unit, w)
                for w in range(4)
            ]
            assert {k for k, _, _ in p} == {"action", "bind", "flag", "and"}
            assert all(
                {t["gold"] for t in x} == {"A", "B"} for _, d, b in p for x in (d, b)
            )


def test_invalid_world_answer_never_disappears_from_composed_action_or_controls():
    f = panel_fixture(BASE, DESIGN, ALL_NEW)
    cases = list(make_cases(list(GROUPS.values()), f, "fresh"))
    world = {
        (i, "and", 0, w, o): {
            "greedy_label": world_probe(flags, w, f, DESIGN["fresh_probes"], "and")[o][
                "gold"
            ]
        }
        for i, flags in enumerate(f["fresh"]["flag_pairs"])
        for w in range(4)
        for o in (0, 1)
    }
    branches = {
        (c["key"], bit, r["label_order"]): {"greedy_label": r["gold"]}
        for c in cases
        for bit in (False, True)
        for r in branch_probe(c, GROUPS[c["group"]], f, DESIGN["fresh_probes"], bit)
    }
    perfect = score.mediated(cases, world, branches, f, GROUPS)
    world[0, "and", 0, 2, 1]["greedy_label"] = None
    failed = score.mediated(cases, world, branches, f, GROUPS)
    for stage in perfect:
        n = perfect[stage]["native"][1]
        assert (
            perfect[stage]["native"]
            == perfect[stage]["correct_bit_exchange"]
            == perfect[stage]["correct_policy_exchange"]
            == [n, n]
        )
        assert failed[stage]["gold_bit"] == [n, n] and failed[stage]["strict"] == [
            8,
            16,
        ]
        assert (
            failed[stage]["native"]
            == failed[stage]["correct_bit_exchange"]
            == [n - n // 16, n]
        )
