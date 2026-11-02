import importlib.util
import json
import subprocess
import sys
import numpy as np
from pathlib import Path
import pytest
import torch
from selffold import prompts
from selffold.codec_learning import EncoderBridge, codec_loss, codec_objectives
from selffold.policy_binding import first_divergence, pair_schedule
from selffold.calibration import codec_examples
from selffold.records import State, digest, file_sha

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/policy_binding_20261103"
DATA = ROOT / "data/codec_preparation_20261102"


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_common_prefix_loss_uses_only_visible_prefix_and_reaches_bridge(host):
    tok = host.tokenizer
    bridge = EncoderBridge(host.width, 3)
    slots = host.encode("A source").slots
    left, right = [tok.encode(text, add_special_tokens=False) + [2] for text in ("a red", "a tan")]
    i = first_divergence(left, right)
    nll, margin = codec_objectives(host, bridge, slots, left, contrast=(i, right[i]))
    assert nll.item() == codec_loss(host, bridge, slots, "a red", end_token_id=2).item()
    torch.nn.functional.softplus(-margin).backward()
    assert bridge.up.grad.abs().sum() > 0
    assert all(p.grad is None for p in host.model.parameters())
    ids, pos = host.render(prompts.read_prompt())
    masked = ids + left[:i] + [host.placeholder_id] * (len(left)-i)
    logits, _ = host.forward(masked, state=State(slots=bridge(slots)), input_positions=pos[prompts.INPUT_MARKER])
    at = logits[0, len(ids)+i-1].float()
    assert margin.item() == pytest.approx((at[left[i]]-at[right[i]]).item(), abs=1e-5)


def test_pairs_and_schedule_keep_opposites_together():
    for left, right in [([1, 2], [1, 2]), ([1, 2], [1, 3]), ([1, 2], [1, 4, 2])]:
        with pytest.raises(ValueError):
            first_divergence(left, right)
    pairs = [(2*i, 2*i+1) for i in range(96)]
    batch = pair_schedule(pairs)
    assert batch == pair_schedule(pairs)
    assert len(batch) == 64 and all(len(row) == 8 for row in batch)
    assert all(b == a+1 for row in batch for a, b in zip(row[::2], row[1::2]))
    assert len({x for row in batch[:24] for x in row}) == 192


def test_ignoring_opposed_donor_can_get_half_accuracy_and_zero_switches():
    path = Path(__file__).resolve().parents[1] / "provenance/policy_binding_20261103/summarize.py"
    spec = importlib.util.spec_from_file_location("binding_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    items = []
    for pole in ("plus", "minus"):
        for wording in ("p0", "p1"):
            for world in (0, 1):
                for order in (0, 1):
                    gold = "A" if bool(world) ^ bool(order) ^ (pole == "plus") else "B"
                    items.append(dict(pole=pole, wording=wording, world=world, order=order, A=1, B=0, gold=gold))
    assert module.group_metrics(items, 2) == {"accuracy": (8, 16), "opposed": (0, 8),
                                              "wrappers": (4, 8), "world_switch": (0, 8)}
    with pytest.raises(ValueError):
        module.group_metrics(items + items[:1], 2)


def test_public_policy_manifests_and_summaries_reproduce():
    root, study = ROOT, STUDY
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0"))
    entries = {}
    for line in (study / "SHA256SUMS").read_text().splitlines():
        sha, p = line.split("  ", 1)
        assert p not in entries
        entries[p] = sha
        assert file_sha(study / p) == sha
    assert set(entries) == {str((root / p).relative_to(study)) for p in tracked
                            if p.startswith(str(study.relative_to(root)) + "/") and not p.endswith("/SHA256SUMS")}
    for program, name in (("summarize.py", "summary.json"), ("summarize_geometry.py", "geometry_summary.json"),
                          ("order_controls.py", "order_audit.json")):
        assert subprocess.check_output([sys.executable, str(study / program)]) == (study / name).read_bytes()
    assert subprocess.check_output([sys.executable, str(study / "order_controls.py"), "--panel"]) == (study / "order_controls.jsonl").read_bytes()


def test_two_fits_share_start_order_and_compute_and_bind_serialized_endpoints():
    run, study, data = STUDY / "run", STUDY, DATA
    seal = json.loads((run / "source.json").read_text())
    assert digest(seal["files"]) == seal["source_sha256"] and seal["git_status"] == ""
    assert seal["git_commit"] == "5e4bd86f9ca45fd0e3f0192fb165cc3eac76c7e5"
    host = json.loads((run / "host.json").read_text())
    training = lines(run / "training.jsonl")
    fits = lines(run / "fits.jsonl")
    assert len(fits) == 2 and {x["arm"] for x in fits} == {"copy", "opposition"}
    for fit in fits:
        history = [x for x in training if x["arm"] == fit["arm"]]
        assert [x["step"] for x in history] == list(range(1, 65))
        assert all(len(x["cases"]) == len(x["nll"]) == len(x["opposed_token_ce"]) == 8
                   and x["gradient_norm"] > 0 and np.isfinite(x["gradient_norm"]) for x in history)
        assert digest([x["cases"] for x in history]) == fit["schedule_sha256"]
        assert fit["forwards"] == 512 and fit["processed_tokens"] == 44804
        artifact = run / (fit["arm"] + ".npz")
        assert file_sha(artifact) == fit["sha256"]
        with np.load(artifact, allow_pickle=False) as z:
            meta = json.loads(z["metadata_utf8"].tobytes())
            assert (z["down"].shape, z["up"].shape) == ((2304, 16), (16, 2304))
            assert meta["base_host_sha256"] == digest(host) and meta["source_sha256"] == seal["source_sha256"]
            assert meta["dataset_sha256"] == file_sha(data / "manifest.json")
            assert meta["training"]["plan_sha256"] == file_sha(study / "PLAN.md")
    for key in ("schedule_sha256", "initial_parameters_sha256"):
        assert fits[0][key] == fits[1][key]
    assert fits[0]["final_parameters_sha256"] != fits[1]["final_parameters_sha256"]


def test_functional_cases_reuse_historical_donors_and_independent_world_golds():
    root, data, run = ROOT, DATA, STUDY / "run"
    source_sha = {x["id"]: digest(x["source"]) for s in ("train", "dev") for x in codec_examples(data, s)}
    donors = {x["case"]: x for x in lines(run / "donors.jsonl")}
    assert {k: v["source_sha256"] for k, v in donors.items()} == source_sha
    previous = root / "provenance/carriers_20261102/learning/gemma-codec/cached_sources.jsonl"
    assert {k: v["state_sha256"] for k, v in donors.items()} == {r["case"]: r["state_sha256"] for r in lines(previous)}
    prefix = {(x["arm"], x["case"]): x for x in lines(run / "prefix.jsonl")}
    groups = {x["id"]: x for x in lines(data / "dev.groups.jsonl")}
    for x in lines(run / "functional.jsonl"):
        g = groups[x["group"]]
        target = g["poles"][x["pole"]]["truth_ledger"][x["world"]]["choice"]
        assert x["gold"] == ("A" if g["options"][x["order"]] == target else "B")
        if x["route"] == "matched":
            assert x["state_sha256"] == prefix[x["arm"], x["case"]]["state_sha256"]
    for t in lines(run / "neutral.jsonl"):
        assert t["exact"]


def test_geometry_replay_binds_original_states_and_single_source_controls():
    study, run, geom = STUDY, STUDY / "run", STUDY / "geometry"
    donors = {x["case"]: x for x in lines(run / "donors.jsonl")}
    terminal = json.loads((run / "terminal.json").read_text())
    assert all(json.loads((r / "native_parity.json").read_text())["exact"] for r in (geom, run))
    geom_terminal = json.loads((geom / "terminal.json").read_text())
    assert geom_terminal["optimizer_updates"] == 0 and geom_terminal["prior_margin_max_error"] <= 1e-5
    assert terminal["backbone_sha256_before"] == terminal["backbone_sha256_after"] == geom_terminal["backbone_sha256"]
    assert not terminal["G_teacher_used_by_loss"] and not terminal["qualification_content_loaded"] and not geom_terminal["qualification_content_loaded"]
    assert json.loads((geom / "request.json").read_text())["panel_sha256"] == file_sha(study / "order_controls.jsonl")
    assert json.loads((geom / "source.json").read_text())["git_commit"] == "65457166eb22d450802c2cc8f18ba09560ddb44a"
    captures = lines(geom / "captures.jsonl")
    assert len(captures) == 384
    for x in captures:
        if x["panel"] != "crossed":
            assert x["state_sha256"] == donors[x["case"]]["state_sha256"]
    assert len(lines(geom / "scores.jsonl")) == 2304
