"""Validate the scope and sealed records of the E-to-lexical experiment."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from selffold.records import digest, file_sha

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/source_lexical_20261103"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_source_map_archive_membership_and_reducer_replay():
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", str(STUDY.relative_to(ROOT))], cwd=ROOT
    )
    names = {
        Path(x.decode()).relative_to(STUDY.relative_to(ROOT)).as_posix()
        for x in tracked.split(b"\0")
        if x
    }
    lines = (STUDY / "SHA256SUMS").read_text().splitlines()
    manifest = {name: sha for sha, name in (line.split("  ", 1) for line in lines)}
    assert len(lines) == len(manifest) and set(manifest) == names - {"SHA256SUMS"}
    for name, sha in manifest.items():
        assert hashlib.sha256((STUDY / name).read_bytes()).hexdigest() == sha
    assert (
        subprocess.check_output([sys.executable, str(STUDY / "summarize.py")])
        == (STUDY / "summary.json").read_bytes()
    )


def test_source_map_training_scope_and_frozen_modules_are_bound():
    run = STUDY / "run"
    seal, request, terminal = [
        read(run / (name + ".json")) for name in ("source", "request", "terminal")
    ]
    assert seal["git_commit"] == "9c0c4965e0d555d7ebc20c9f8a3581273c56e0b2"
    assert seal["git_status"] == "" and digest(seal["files"]) == seal["source_sha256"]
    assert request["dataset_sha256"] == file_sha(
        ROOT / "data/source_lexical_20261103.json"
    )
    assert request["plan_sha256"] == file_sha(STUDY / "PLAN.md")
    assert request["loss_weights"] == {
        "native_binary": 1.0,
        "representation": 0.1,
        "variance_floor": 0.01,
    }
    assert (
        terminal["status"] == "COMPLETED_SOURCE_LEXICAL_DEVELOPMENT"
        and terminal["updates"] == 64
    )
    assert terminal["backbone_sha256_before"] == terminal["backbone_sha256_after"]
    assert (
        terminal["lexical_parameter_sha256_before"]
        == terminal["lexical_parameter_sha256_after"]
    )
    assert not terminal["G_trained"] and not terminal["qualification_content_loaded"]
    sources = {r["key"]: r for r in rows(run / "sources.jsonl")}
    teachers = {(r["group"], r["pole"]): r for r in rows(run / "teachers.jsonl")}
    assert len(sources) == 96 and len(teachers) == 24
    assert all(
        len(t["native_ids"]) == 256 and len(t["capture_positions"]) == 8
        for t in teachers.values()
    )
    fit = read(run / "ridge_fit.json")
    training = {
        key for key, value in sources.items() if value["split"] == "seen_sources"
    }
    assert set(fit["source_keys"]) == training and len(training) == 32
    assert fit["source_state_sha256"] == digest(
        [sources[key]["state_sha256"] for key in fit["source_keys"]]
    )
    assert fit["teacher_state_sha256"] == digest(
        [
            teachers[sources[key]["group"], sources[key]["pole"]]["state_sha256"]
            for key in fit["source_keys"]
        ]
    )
    history = rows(run / "training.jsonl")
    assert len(history) == 64 and [r["step"] for r in history] == list(range(1, 65))
    assert all(
        np.isfinite(r["gradient_norm"]) and r["gradient_norm"] > 0 for r in history
    )
    seen = Counter()
    for record in history:
        keys = record["source_keys"]
        assert len(keys) == 8 and set(keys) <= training
        assert Counter(sources[key]["pole"] for key in keys) == {"plus": 4, "minus": 4}
        seen.update(keys)
    assert set(seen.values()) == {16}
    assert read(run / "training_compute.json")["forwards"] == 256


def test_source_map_artifacts_and_previous_lexical_replay_match_their_bindings():
    run = STUDY / "run"
    seal = read(run / "source.json")
    frozen_fields = (
        "source_mean",
        "source_scale",
        "anchors",
        "coefficients",
        "teacher_mean",
    )
    snapshots = {}
    for fit in rows(run / "fits.jsonl"):
        path = run / fit["artifact"]
        assert fit["serialization_exact"] and file_sha(path) == fit["sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            meta = json.loads(arrays["metadata_utf8"].tobytes())
            assert meta["source_sha256"] == seal["source_sha256"]
            assert meta["base_host_sha256"] == digest(read(run / "host.json"))
            assert meta["dataset_sha256"] == file_sha(
                ROOT / "data/source_lexical_20261103.json"
            )
            assert (
                meta["lexical_sha256"] == read(run / "request.json")["lexical_sha256"]
            )
            assert arrays["anchors"].shape == (32, 18432)
            assert arrays["down"].shape == (18432, 16) and arrays["up"].shape == (
                16,
                18432,
            )
            snapshots[fit["endpoint"]] = {
                key: arrays[key].copy() for key in (*frozen_fields, "up")
            }
    assert set(snapshots) == {"ridge", "final"}
    assert all(
        np.array_equal(snapshots["ridge"][key], snapshots["final"][key])
        for key in frozen_fields
    )
    assert not snapshots["ridge"]["up"].any() and snapshots["final"]["up"].any()
    prior = ROOT / "provenance/slot_readout_20261103/run/lexical"
    old = {
        (r["group"], r["pole"], r["label_order"]): r
        for r in rows(prior / "decisions.jsonl")
        if r["route"] == "bridge_matched"
    }
    replay = rows(run / "lexical_replay.jsonl")
    assert len(replay) == len(old) == 24
    for r in replay:
        assert all(
            r[key] == old[r["group"], r["pole"], r["label_order"]][key]
            for key in ("A", "B", "greedy_token", "greedy_label")
        )
