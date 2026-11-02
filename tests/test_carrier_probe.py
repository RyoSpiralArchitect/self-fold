import json
import torch
import pytest
from selffold.carrier_probe import capture_description, neutral_replay, run_carrier_probe
from selffold.records import RunStore


def test_description_capture_is_repeatable_and_has_no_question_input(host):
    for route in ("workspace", "native_tail"):
        text = "I organize my notebook with the newest notes first."
        first, metadata = capture_description(host, text, route)
        second, _ = capture_description(host, text, route)
        assert first.sha() == second.sha()
        assert metadata["forward_length"] == 256
        assert metadata["source_text"] == text
        assert len(metadata["capture_positions"]) == host.slots
    with pytest.raises(ValueError, match="too short"):
        capture_description(host, "x", "native_tail")
    with pytest.raises(ValueError, match="no truncation"):
        capture_description(host, "x" * 300, "workspace")


def test_neutral_replay_matches_the_actual_full_label_forward(host):
    assert neutral_replay(host, "Choose A or B.")["exact"]
    assert not host.layers[host.cut]._forward_pre_hooks


def test_all_readouts_reuse_saved_carriers_and_keep_full_denominators(host, tmp_path):
    groups = [{"id": name, "question": "Which?", "options": ["newest", "oldest"],
               "plus": ["I put the newest notes first."], "minus": ["I put the oldest notes first."]}
              for name in ("one", "two")]
    store = RunStore(tmp_path / "carriers")
    run_carrier_probe(host, store, groups, routes=("native_tail",))
    donors = [json.loads(x) for x in (store.path / "donors.jsonl").read_text().splitlines()]
    scores = [json.loads(x) for x in (store.path / "scores.jsonl").read_text().splitlines()]
    assert len(donors) == 4 and len(scores) == 4 * 8 * 2 * 2
    for donor in donors:
        actual = [x for x in scores if x["case"] == donor["case"] and x["control"] == "carrier"]
        assert len(actual) == 4
        assert {x["state_sha256"] for x in actual} == {donor["state_sha256"]}
    assert all(x["correct"] == (x["prediction"] == x["gold"]) for x in scores)
    assert len((store.path / "neutral.jsonl").read_text().splitlines()) == 8
    assert all(not p.requires_grad for p in host.model.parameters())
