"""Public provenance bindings for the native-recipient and numerical-audit runs."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from selffold.records import digest, file_sha

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/continuity_20261103"
DATA = ROOT / "data/continuity_20261103/manifest.json"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_continuity_manifest_covers_only_the_complete_tracked_public_study():
    tracked = subprocess.check_output(["git", "ls-files", "-z", "--", str(STUDY.relative_to(ROOT))], cwd=ROOT)
    names = {str((ROOT / p.decode()).relative_to(STUDY)) for p in tracked.split(b"\0") if p}
    expected = names - {"SHA256SUMS"}
    entries = {}
    for line in (STUDY / "SHA256SUMS").read_text().splitlines():
        sha, name = line.split("  ", 1)
        assert name not in entries and not Path(name).is_absolute() and ".." not in Path(name).parts
        assert "__pycache__" not in Path(name).parts and not name.endswith(".pyc")
        assert "qualification.groups.jsonl" not in name
        entries[name] = sha
        assert file_sha(STUDY / name) == sha
    assert set(entries) == expected


@pytest.mark.parametrize("folder,summary", [("recipient", "summary.json"), ("binary", "binary_summary.json")])
def test_published_recipient_summaries_reproduce_from_all_native_decisions(folder, summary):
    regenerated = subprocess.check_output([sys.executable, str(STUDY / "summarize.py"), str(STUDY / folder)])
    assert regenerated == (STUDY / summary).read_bytes()


def test_recipient_artifacts_and_training_history_bind_each_declared_objective():
    fits = {}
    histories = {}
    for dirname in ("recipient", "binary"):
        folder = STUDY / dirname
        source, host = read(folder / "source.json"), read(folder / "host.json")
        request, schedule = read(folder / "request.json"), read(folder / "schedule.json")
        assert source["git_status"] == "" and digest(source["files"]) == source["source_sha256"]
        assert request["data_sha256"] == file_sha(DATA) and len(schedule) == 256
        assert request["checkpoint_sha256"] == host["checkpoint_sha256"]
        training = rows(folder / "training.jsonl")
        for fit in rows(folder / "fits.jsonl"):
            candidate = fit["candidate"]
            assert candidate not in fits
            fits[candidate] = fit
            history = [r for r in training if r["candidate"] == candidate]
            histories[candidate] = history
            assert [r["step"] for r in history] == list(range(1, 257))
            assert all(np.isfinite(r["gradient_norm"]) and r["gradient_norm"] > 0 for r in history)
            assert all({key: r[key] for key in entry} == entry for r, entry in zip(history, schedule, strict=True))
            assert fit["training_forwards"] == 256 * request["effective_batch"] // request["microbatch"]
            artifact = folder / fit["artifact"]
            assert artifact.parent == folder and file_sha(artifact) == fit["sha256"]
            with np.load(artifact, allow_pickle=False) as archive:
                meta = json.loads(archive["metadata_utf8"].tobytes())
                architecture = meta["architecture"]
                assert meta["dataset_sha256"] == file_sha(DATA)
                assert meta["source_sha256"] == source["source_sha256"] and meta["base_host_sha256"] == digest(host)
                assert meta["training"]["schedule_sha256"] == digest(schedule)
                assert meta["training"]["loss"] == request["loss"]
                assert architecture["rank"] == 16 and architecture["last_layers"] == 4
                assert architecture["parameter_dtype"] == "float32" and architecture["scope"] == "explicit_task_context_only"
                expected = {"metadata_utf8"}
                count = 0
                for index, target in enumerate(architecture["targets"]):
                    assert target["layer"] >= host["cut"] and target["name"] in ("q_proj", "v_proj")
                    shapes = {"down": (target["inputs"], 16), "up": (16, target["outputs"])}
                    for part, shape in shapes.items():
                        name = f"factors.{index}.{part}"
                        expected.add(name)
                        value = archive[name]
                        assert value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all()
                        count += value.size
                assert set(archive.files) == expected and count == fit["trainable_parameters"]
        for name in ("NOTICE.txt", "ADAPTER_LICENSE.md", "GEMMA_TERMS.html", "GEMMA_PROHIBITED_USE_POLICY.html"):
            assert (folder / name).is_file()
    assert set(fits) == {"raw", "standardized", "raw_binary"}
    assert fits["raw"]["initial_parameters_sha256"] == fits["standardized"]["initial_parameters_sha256"] == fits["raw_binary"]["initial_parameters_sha256"]
    assert fits["raw"]["training_processed_tokens"] == fits["standardized"]["training_processed_tokens"] == fits["raw_binary"]["training_processed_tokens"]
    assert [r["tasks_sha256"] for r in histories["raw"]] == [r["tasks_sha256"] for r in histories["raw_binary"]]
    assert read(STUDY / "recipient/schedule.json") == read(STUDY / "binary/schedule.json")
    assert read(STUDY / "binary/request.json")["objective"] == "native_binary"


def test_binary_phase_replays_raw_sources_and_binds_the_read_only_audit():
    original, binary, audit = (STUDY / name for name in ("recipient", "binary", "numerical-audit"))
    old = {r["key"]: r for r in rows(original / "donors.jsonl")}
    new = {r["key"]: r for r in rows(binary / "donors.jsonl")}
    assert len(old) == len(new) == 4320 and old == new
    request = read(binary / "request.json")
    baseline, numerical = request["baseline_reference"], request["numerical_audit_reference"]
    assert baseline["terminal_sha256"] == file_sha(original / "terminal.json")
    assert baseline["schedule_sha256"] == digest(read(original / "schedule.json"))
    assert baseline["literal_base"]["functional_file_sha256"] == file_sha(original / "functional.jsonl")
    assert numerical["terminal_sha256"] == file_sha(audit / "terminal.json")
    assert numerical["summary_sha256"] == file_sha(audit / "summary.json")
    assert numerical["request_sha256"] == file_sha(audit / "request.json")
    assert request["binary_plan_sha256"] == file_sha(STUDY / "BINARY_PLAN.md")
    terminal = read(audit / "terminal.json")
    assert terminal["optimizer_updates"] == 0 and terminal["cases"] == 1152 and terminal["opposed_pairs"] == 576
    assert len(rows(audit / "cases.jsonl")) == 1152 and len(rows(audit / "opposed_hidden.jsonl")) == 576
    for artifact in rows(audit / "artifacts.jsonl"):
        assert artifact["parameter_sha256_before"] == artifact["parameter_sha256_after"]
    backbones = set()
    for folder in (original, binary, audit):
        t = read(folder / "terminal.json")
        assert not t["qualification_content_loaded"]
        assert t["backbone_sha256_before"] == t["backbone_sha256_after"]
        backbones.add(t["backbone_sha256_before"])
        assert read(folder / "native_parity.json")["exact"]
    assert len(backbones) == 1
    assert not read(original / "terminal.json")["G_trained"] and not read(binary / "terminal.json")["G_trained"]
