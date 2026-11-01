from pathlib import Path
import json
import runpy

import torch

from selffold.records import RunStore, State


ROOT = Path(__file__).resolve().parents[1]
PROBE = runpy.run_path(str(ROOT / "scripts/probe_slot_content.py"))


def test_label_order_cancels_fixed_label_bias(tmp_path):
    class LabelBias:
        def score_choices(self, state, question, labels):
            return {"A": {"logprob": -1, "tokens": 1}, "B": {"logprob": -4, "tokens": 1}}

    store = RunStore(tmp_path / "trial")
    group = {"id": "unit", "question": "Which?", "options": ["roads", "rivers"]}
    PROBE["score"](LabelBias(), store, group, "uninjected", None)
    row = json.loads((store.path / "scores.jsonl").read_text())
    assert [x["plus_margin"] for x in row["orders"]] == [3, -3]
    assert row["plus_margin"] == 0 and row["state_sha256"] is None
    assert "A = rivers. B = roads." in row["orders"][1]["question"]


def test_null_control_preserves_the_injected_prompt(host, monkeypatch):
    original = host.forward
    calls = []

    def trace(ids, **kwargs):
        calls.append((ids.copy(), kwargs["input_positions"], kwargs["state"] is None))
        return original(ids, **kwargs)

    monkeypatch.setattr(host, "forward", trace)
    zeros = State(slots=torch.zeros(host.slots, host.width))
    for state in (None, zeros):
        host.score_choices(state, "Which description fits?", ["A", "B"])
    for blank, injected in zip(calls[:2], calls[2:]):
        assert blank[:2] == injected[:2] and len(blank[1]) == host.slots
        assert blank[2] and not injected[2]


def test_magnitude_control_keeps_slot_norms_and_reuses_directions():
    slots = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    first, err = PROBE["magnitude_control"](State(slots=slots))
    second, _ = PROBE["magnitude_control"](State(slots=-slots))
    assert torch.equal(first.slots, second.slots) and err < 1e-6
    assert torch.allclose(first.slots.norm(dim=1), slots.norm(dim=1))


def test_fixture_has_six_balanced_groups():
    groups = json.loads((ROOT / "data/slot_contrasts.dev.json").read_text())["groups"]
    assert len({g["id"] for g in groups}) == len(groups) == 6
    for group in groups:
        assert len(set(group["options"])) == 2
        assert len(group["plus"]) == len(group["minus"]) == 2
        assert len(set(group["plus"] + group["minus"])) == 4
