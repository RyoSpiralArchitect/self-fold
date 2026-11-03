"""Ensure training schedules never draw a fresh source or break a conditional unit."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fit_conditional_curriculum import make_schedule
from selffold.conditional_curriculum import STAGES, make_cases


def test_warmup_stage_order_and_branch_units_are_fixed_before_scoring():
    groups = json.loads((ROOT / "data/within_topic_wording_20261103.json").read_text())[
        "groups"
    ]
    f = json.loads((ROOT / "data/conditional_curriculum_20261103.json").read_text())
    fit = list(make_cases(groups, f, "fit"))
    phases = ["cases"] * 64 + ["rule"] * 64 + [STAGES[i % 3] for i in range(256)]
    warm = make_schedule(fit, phases, 47)
    assert warm == make_schedule(fit, phases, 47) and len(warm) == 384
    available = {
        tuple(
            c[k]
            for k in ("group", "stage", "flag_pair", "style", "source_order", "pole")
        ): c
        for c in fit
    }
    for stage, units in zip(phases, warm, strict=True):
        assert len(units) == len(set(units)) == 2 and all(u[1] == stage for u in units)
        records = [available[(*u, pole)] for u in units for pole in (0, 1)]
        assert len({r["key"] for r in records}) == 4 and all(
            r["split"] == "fit" for r in records
        )
    branches = make_schedule(fit, [STAGES[i % 3] for i in range(256)], 947)
    assert len(branches) == 256 and branches != warm[:256]
