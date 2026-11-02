"""Check the registered interventions without a pretrained model."""

import importlib.util
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "probe_slot_readout", ROOT / "scripts/probe_slot_readout.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_case_permutations_keep_every_action_assignment_and_training_uses_both_orders():
    groups = {"train": probe.load_groups(probe.DATA, "train")[:2]}
    original = probe.material(groups)
    forms = probe.order_material(groups)
    for key, row in forms["forward"].items():
        assert row["source"] == original["forward"][key]["source"]
        assert forms["reverse"][key]["source"] == original["reverse"][key]["source"]
        clauses = re.findall(r"When ([^,]+), I choose ([^.]+)\.", row["source"])
        for name, order in probe.ORDERS.items():
            actual = re.findall(
                r"When ([^,]+), I choose ([^.]+)\.", forms[name][key]["source"]
            )
            assert actual == [clauses[i] for i in order]
            assert dict(actual) == dict(clauses)
        assert (
            forms["mixed"][key]
            == forms["reverse" if key.endswith("-o1") else "forward"][key]
        )


def test_gain_is_centered_on_the_neutral_replay_and_preserves_the_lexical_endpoint():
    neutral = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    lexical = neutral + torch.arange(3, dtype=torch.float32)
    assert torch.equal(probe.gain_slots(neutral, lexical, 0), neutral)
    assert torch.equal(probe.gain_slots(neutral, lexical, 1), lexical)
    assert torch.equal(
        probe.gain_slots(neutral, lexical, 4) - neutral, 4 * (lexical - neutral)
    )
