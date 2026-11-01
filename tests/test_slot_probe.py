from pathlib import Path
import json
import runpy
import subprocess
import sys

import torch
import pytest

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


def test_aligned_control_uses_same_positions_and_native_token_path(host, tmp_path):
    group = {"id": "unit", "question": "Which?", "options": ["a", "b"]}
    store = RunStore(tmp_path / "alignment")
    PROBE["lexical_alignment"](host, store, [group])
    rows = [json.loads(s) for s in (store.path / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == 10
    neutral = [host.placeholder_id] * host.slots
    assert host.score_choices(None, "Which?", ["A"], slot_token_ids=neutral) == host.score_choices(None, "Which?", ["A"])
    with pytest.raises(ValueError):
        host.score_choices(State(text="a"), "Which?", ["A"], slot_token_ids=neutral)
    cases = [json.loads(s) for s in (store.path / "cases.jsonl").read_text().splitlines()]
    assert all(len(c["padded_ids"]) == host.slots for c in cases)
    assert all(c["padded_ids"][:1] == c["token_ids"] for c in cases)


def test_full_span_reconstructs_native_lexical_readout(host, tmp_path):
    from selffold.replay_probe import run_replay_probe

    store = RunStore(tmp_path / "replay")
    run_replay_probe(host, store, [{"id": "unit", "question": "Which?", "options": ["a", "b"]}])
    controls = [json.loads(s) for s in (store.path / "neutral.jsonl").read_text().splitlines()]
    assert len(controls) == 2 and all(r["same_length"]["max_abs_logit_difference"] == 0 for r in controls)
    rows = [json.loads(s) for s in (store.path / "interventions.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert all(r["native_vs_opaque_slot_max_abs"] == 0 for r in rows)
    assert all(r["max_abs_logit_difference_from_native"]["slots_and_suffix"] == 0 for r in rows)
    assert not host.layers[host.cut]._forward_pre_hooks


def test_published_probe_summary_recomputes_from_the_scored_trials():
    study = ROOT / "provenance/slot_content_20261101"
    regenerated = subprocess.check_output([sys.executable, str(study / "summarize.py")])
    assert regenerated == (study / "summary.json").read_bytes()
