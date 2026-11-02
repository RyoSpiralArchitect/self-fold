import json
from pathlib import Path
import subprocess
import sys
import hashlib
import numpy as np
import torch
import pytest
from selffold.calibration import public_groups, write_public, codec_examples
from selffold import prompts
from selffold.records import State
from selffold.codec_learning import EncoderBridge, codec_loss, save_bridge, install_bridge, load_bridge
from selffold.carrier_probe import capture_description


def test_family_splits_and_formal_truth_are_independent_of_any_decoder(tmp_path):
    groups = list(public_groups())
    assert len(groups) == 72 and len({g["lineage_group"] for g in groups}) == 72
    assert {g["family"] for g in groups if g["split"] == "train"}.isdisjoint(
        {g["family"] for g in groups if g["split"] == "dev"})
    for g in groups:
        assert all(a["choice"] != b["choice"] for a, b in zip(g["poles"]["plus"]["truth_ledger"], g["poles"]["minus"]["truth_ledger"]))
    out = tmp_path / "data"
    write_public(out)
    assert len(codec_examples(out, "train")) == 192 and len(codec_examples(out, "dev")) == 96
    assert all(row["source"] == row["target"] for row in codec_examples(out, "train"))
    with pytest.raises(ValueError, match="cannot load"):
        codec_examples(out, "qualification")
    file = out / "train.groups.jsonl"
    file.write_text(file.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        codec_examples(out, "train")


def test_real_gradient_path_changes_only_bridge_and_cleans_hooks(host, tmp_path):
    state = host.encode("I place the newest notes first.")
    fingerprint = state.sha()
    bridge = EncoderBridge(host.width, 3)
    assert torch.equal(bridge(state.slots), state.slots)
    original = {n: p.detach().clone() for n, p in host.model.named_parameters()}
    loss = codec_loss(host, bridge, state.slots, "new notes", end_token_id=2)
    loss.backward()
    assert bridge.up.grad is not None and bridge.up.grad.abs().sum() > 0
    assert all(p.grad is None for p in host.model.parameters())
    torch.optim.SGD(bridge.parameters(), lr=.01).step()
    assert not torch.equal(bridge(state.slots), state.slots)
    assert fingerprint == state.sha()
    assert all(torch.equal(p, original[n]) for n, p in host.model.named_parameters())
    assert not host.layers[host.cut]._forward_pre_hooks
    # The differentiable teacher-forcing objective agrees with the inference host.
    expected_loss = codec_loss(host, bridge, state.slots, "new notes", end_token_id=2).detach()
    with torch.no_grad():
        adapted = State(slots=bridge(state.slots))
    ids, positions = host.render(prompts.read_prompt())
    target = host.tokenizer.encode("new notes", add_special_tokens=False) + [2]
    logits, _ = host.forward(ids + target, state=adapted, input_positions=positions[prompts.INPUT_MARKER])
    actual_loss = torch.nn.functional.cross_entropy(logits[0, len(ids)-1:-1].float(), torch.tensor(target))
    assert abs(float(actual_loss) - float(expected_loss)) < 1e-5
    artifact = tmp_path / "encoder.npz"
    save_bridge(artifact, bridge, host.identity, dataset_sha256="a"*64, source_sha256="b"*64, training={"steps": 1})
    with pytest.raises(ValueError, match="another host"):
        load_bridge(artifact, {**host.identity, "cut": 99}, device="cpu")
    expected = bridge(capture_description(host, "a source", "workspace")[0].slots).detach()
    install_bridge(host, artifact)
    assert torch.equal(host.encode("a source").slots, expected)
    assert host.identity["encoder_bridge_sha256"]
    with pytest.raises(ValueError, match="already"):
        install_bridge(host, artifact)


def test_invalid_training_context_cannot_leave_a_hook(host):
    state = host.encode("A source description.")
    with torch.inference_mode(), pytest.raises(ValueError, match="gradient"):
        codec_loss(host, EncoderBridge(host.width, 3), state.slots, "target", end_token_id=2)
    assert not host.layers[host.cut]._forward_pre_hooks


def test_published_learning_is_replayable_bound_and_keeps_qualification_sealed():
    root = Path(__file__).resolve().parents[1]
    evidence = root / "provenance/carriers_20261102"
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0"))
    for line in (evidence / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        file = evidence / relative
        assert str(file.relative_to(root)) in tracked
        assert hashlib.sha256(file.read_bytes()).hexdigest() == expected
    regenerated = subprocess.check_output([sys.executable, str(evidence / "summarize.py")])
    assert regenerated == (evidence / "summary.json").read_bytes()
    data = root / "data/codec_preparation_20261102/manifest.json"
    manifest = json.loads(data.read_text())
    assert manifest["qualification"]["status"] == "sealed_before_fitting"
    assert not list((root / "data").rglob("qualification.groups.jsonl"))
    fit = evidence / "learning/gemma-codec"
    source = json.loads((fit / "source.json").read_text())
    with np.load(fit / "encoder_step_064.npz", allow_pickle=False) as selected:
        metadata = json.loads(selected["metadata_utf8"].tobytes())
        assert metadata["dataset_sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
        assert metadata["source_sha256"] == source["source_sha256"]
        assert selected["down"].shape == (2304, 16) and selected["up"].shape == (16, 2304)
        with np.load(fit / "encoder_step_000.npz", allow_pickle=False) as initial:
            assert not np.array_equal(initial["up"], selected["up"])
    summary = json.loads(regenerated)["codec_development_fit"]
    assert summary["backbone_unchanged"] and summary["qualification_outcomes"] == "not_observed"
