import json
from collections import Counter
from pathlib import Path

from selffold.conditional_curriculum import make_cases
from selffold.stage_grammar import ARMS, fixture, panel_jobs, training_pair

ROOT = Path(__file__).resolve().parents[1] / "data"
BASE, OLD, DESIGN, GROUPS = [
    json.loads((ROOT / f).read_text())
    for f in (
        "conditional_margin_20261103.json",
        "two_flags_20261104.json",
        "stage_grammar_20261104.json",
        "within_topic_wording_20261103.json",
    )
]
GROUPS = {g["id"]: g for g in GROUPS["groups"]}


def test_panel_widths_and_fresh_material_distinct():
    train = fixture(BASE, OLD, DESIGN, "train")
    seen = {c["source"] for c in make_cases(list(GROUPS.values()), train, "fit")}
    assert len(seen) == 960
    for panel in ("names", "grammar", "fresh", "previous"):
        f = fixture(BASE, OLD, DESIGN, panel)
        cs = list(make_cases(list(GROUPS.values()), f, "fresh"))
        assert len(cs) == 160 and (
            panel == "previous" or not seen.intersection(c["source"] for c in cs)
        )
        words = OLD["fresh_probes"] if panel == "previous" else DESIGN["fresh_probes"]
        counts = Counter(row[0] for row in panel_jobs(GROUPS, f, words, panel))
        assert counts["ordinary"] == 1280 and counts["branch"] == (
            640 if panel in ("fresh", "previous") else 0
        )


def test_every_semantic_block_keeps_both_mappings_and_exact_task_budget():
    f = fixture(BASE, OLD, DESIGN, "train")
    c = next(make_cases(list(GROUPS.values()), f, "fit"))
    for a in ARMS:
        for step in range(16):
            count = Counter()
            for u in range(2):
                for p in range(2):
                    for w in range(4):
                        k, pair = training_pair(
                            c, GROUPS[c["group"]], f, OLD, DESIGN, step, u, p, w, a
                        )
                        assert {t["gold"] for t in pair} == {"A", "B"} and [
                            t["label_order"] for t in pair
                        ] == [0, 1]
                        count[k] += 1
            assert count == (
                {"action": 3, "bind": 7, "and": 5, "flag": 1}
                if a == "focus"
                else dict.fromkeys(("action", "bind", "and", "flag"), 4)
            )
