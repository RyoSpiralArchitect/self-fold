"""Public provenance bindings for the native-recipient and numerical-audit runs."""
import hashlib
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


def test_joint_summary_and_source_recipient_training_have_the_declared_budget():
    folder = STUDY / "joint"
    regenerated = subprocess.check_output([sys.executable, str(STUDY / "summarize.py"), str(folder)])
    assert regenerated == (STUDY / "joint_summary.json").read_bytes()
    request, fit, terminal = read(folder / "request.json"), read(folder / "fits.json"), read(folder / "terminal.json")
    schedule = read(folder / "schedule.json")
    assert schedule == read(STUDY / "recipient/schedule.json")
    history = rows(folder / "training.jsonl")
    assert [r["step"] for r in history] == list(range(1, 257))
    assert all({key: row[key] for key in entry} == entry for row, entry in zip(history, schedule, strict=True))
    for row in history:
        assert len(row["losses"]) == len(row["source_keys"]) == 4
        assert len(set(row["source_keys"])) == 4
        assert all(np.isfinite(row[k]) and row[k] > 0 for k in
                   ("encoder_gradient_norm", "recipient_gradient_norm", "gradient_norm"))
    original = [r for r in rows(STUDY / "recipient/training.jsonl") if r["candidate"] == "raw"]
    assert [r["tasks_sha256"] for r in history] == [r["tasks_sha256"] for r in original]
    assert fit["source_forwards"] == fit["task_forwards"] == 1024
    assert fit["source_prefix_layers"] == 13 and fit["task_decoder_layers"] == 26
    assert fit["source_processed_tokens"] == 1024 * request["source_forward_length"]
    assert fit["training_forwards"] == fit["source_forwards"] + fit["task_forwards"]
    assert fit["training_processed_tokens"] == fit["source_processed_tokens"] + fit["task_processed_tokens"]
    assert request["plan_sha256"] == file_sha(STUDY / "JOINT_PLAN.md")
    assert request["data_sha256"] == file_sha(DATA) and terminal["optimizer_updates"] == 256
    assert not terminal["G_trained"] and not terminal["qualification_content_loaded"]
    assert terminal["backbone_sha256_before"] == terminal["backbone_sha256_after"]
    assert terminal["backbone_sha256_before"] == read(STUDY / "recipient/terminal.json")["backbone_sha256_before"]
    assert read(folder / "native_parity.before.json")["exact"] and read(folder / "native_parity.after.json")["exact"]


def test_joint_artifacts_bind_both_roles_and_the_frozen_source_captures():
    folder = STUDY / "joint"
    source, host, fit, terminal = (read(folder / name) for name in ("source.json", "host.json", "fits.json", "terminal.json"))
    assert source["git_status"] == "" and digest(source["files"]) == source["source_sha256"]
    schedule_sha = digest(read(folder / "schedule.json"))
    binding = digest({"encoder_parameters_sha256": fit["final_encoder_sha256"],
                      "recipient_parameters_sha256": fit["final_recipient_sha256"],
                      "data_sha256": file_sha(DATA), "schedule_sha256": schedule_sha})
    assert binding == fit["joint_parameter_binding_sha256"] == terminal["joint_parameter_binding_sha256"]
    artifacts = (("encoder", "encoder_source.npz", "source_encoding", "scoped_source_encoder_qv_lora", range(host["cut"]-4, host["cut"])),
                 ("recipient", "recipient_joint_source.npz", "task_readout", "scoped_recipient_qv_lora", range(host["depth"]-4, host["depth"])))
    for role, name, scope, kind, layers in artifacts:
        assert file_sha(folder / name) == fit[role + "_artifact_sha256"] == terminal[role + "_artifact_sha256"]
        with np.load(folder / name, allow_pickle=False) as archive:
            meta = json.loads(archive["metadata_utf8"].tobytes())
            architecture = meta["architecture"]
            assert meta["kind"] == kind and meta["base_host_sha256"] == digest(host)
            assert meta["source_sha256"] == source["source_sha256"] and meta["dataset_sha256"] == file_sha(DATA)
            assert meta["training"]["role"] == scope and meta["training"]["joint_parameter_binding_sha256"] == binding
            assert meta["training"]["schedule_sha256"] == schedule_sha and architecture["rank"] == 16
            assert {(t["layer"], t["name"]) for t in architecture["targets"]} == {(layer, p) for layer in layers for p in ("q_proj", "v_proj")}
            expected, count, parameter_hash = {"metadata_utf8"}, 0, hashlib.sha256()
            for i, target in enumerate(architecture["targets"]):
                for part, shape in (("down", (target["inputs"], 16)), ("up", (16, target["outputs"]))):
                    key = f"factors.{i}.{part}"
                    value = archive[key]
                    assert value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all()
                    expected.add(key)
                    count += value.size
                    parameter_hash.update((key + str(shape) + "torch.float32").encode())
                    parameter_hash.update(value.tobytes(order="C"))
            assert set(archive.files) == expected and parameter_hash.hexdigest() == fit["final_" + role + "_sha256"]
            assert count == fit["source_trainable_parameters" if role == "encoder" else "recipient_trainable_parameters"]
    old = {r["key"]: r for r in rows(STUDY / "recipient/donors.jsonl")}
    parity = rows(folder / "initial_source_parity.jsonl")
    assert len(parity) == 6
    assert {(old[r["key"]]["split"], old[r["key"]]["stage"]) for r in parity} == {(s, t) for s in ("train", "dev") for t in ("source", "stage1", "stage2")}
    assert all(r["state_sha256"] == r["baseline_state_sha256"] == old[r["key"]]["state_sha256"]
               and r["positions"] == old[r["key"]]["positions"] for r in parity)
    captured = rows(folder / "donors.jsonl")
    assert len(captured) == len({r["key"] for r in captured}) == terminal["logical_donor_rows"] == 4320
    for row in captured:
        assert row["source_sha256"] == old[row["key"]]["source_sha256"]
        assert row["encoder_artifact_sha256"] == fit["encoder_artifact_sha256"]
    states = {r["key"]: r["state_sha256"] for r in captured}
    for row in rows(folder / "functional.jsonl"):
        key = row["stage"] + "/" + row["case"]
        if row["route"] == "matched":
            assert row["state_sha256"] == states[key]
        elif row["route"] == "opposed":
            other_pole = "minus" if row["pole"] == "plus" else "plus"
            other = key.replace("-" + row["pole"] + "-", "-" + other_pole + "-")
            assert row["state_sha256"] == states[other]
    for name in ("NOTICE.txt", "ADAPTER_LICENSE.md", "GEMMA_TERMS.html", "GEMMA_PROHIBITED_USE_POLICY.html"):
        assert (folder / name).is_file()


def test_paired_trainability_control_reproduces_its_training_only_endpoint():
    from collections import defaultdict
    from statistics import fmean

    folder = STUDY / "trainability"
    decisions, history, fits = (rows(folder / name) for name in ("decisions.jsonl", "training.jsonl", "fits.jsonl"))
    terminal, source, host = (read(folder / name) for name in ("terminal.json", "source.json", "host.json"))
    request, schedule, summary = (read(folder / name) for name in ("request.json", "schedule.json", "summary.json"))
    assert len(decisions) == terminal["decisions"] == 768 and len(history) == terminal["optimizer_updates"] == 128
    assert len(fits) == 2 and {f["condition"] for f in fits} == {"literal", "latent"} and len(schedule) == 64
    assert source["git_status"] == "" and digest(source["files"]) == source["source_sha256"]
    assert request["data_sha256"] == file_sha(DATA) and request["plan_sha256"] == file_sha(STUDY / "TRAINABILITY_PLAN.md")
    previous = {r["key"]: r for r in rows(STUDY / "recipient/donors.jsonl")}
    donors = rows(folder / "donors.jsonl")
    assert len(donors) == len({r["key"] for r in donors}) == terminal["logical_donors"] == 24
    for donor in donors:
        assert donor["state_sha256"] == previous[donor["key"]]["state_sha256"]
        assert donor["source_sha256"] == previous[donor["key"]]["source_sha256"]
    for fit in fits:
        entries = [r for r in history if r["condition"] == fit["condition"]]
        assert [r["step"] for r in entries] == list(range(1, 65)) and [r["group"] for r in entries] == schedule
        assert all(np.isfinite(r["gradient_norm"]) and r["gradient_norm"] >= 0 for r in entries)
        assert all(len(r["losses"]) == 4 and np.isfinite(r["losses"]).all() for r in entries)
        assert digest([r["tasks_sha256"] for r in entries]) == fit["task_order_sha256"]
        assert fit["training_forwards"] == 64 * 4
        assert file_sha(folder / fit["artifact"]) == fit["sha256"]
        with np.load(folder / fit["artifact"], allow_pickle=False) as archive:
            meta = json.loads(archive["metadata_utf8"].tobytes())
            assert meta["kind"] == "scoped_recipient_qv_lora" and meta["base_host_sha256"] == digest(host)
            assert meta["dataset_sha256"] == file_sha(DATA) and meta["source_sha256"] == source["source_sha256"]
            assert meta["training"]["input_condition"] == fit["condition"] and meta["training"]["schedule_sha256"] == digest(schedule)
            expected, parameter_hash = {"metadata_utf8"}, hashlib.sha256()
            for i, target in enumerate(meta["architecture"]["targets"]):
                for part, shape in (("down", (target["inputs"], 16)), ("up", (16, target["outputs"]))):
                    name = f"factors.{i}.{part}"
                    value = archive[name]
                    assert value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all()
                    expected.add(name)
                    parameter_hash.update((name + str(shape) + "torch.float32").encode())
                    parameter_hash.update(value.tobytes(order="C"))
            assert set(archive.files) == expected and parameter_hash.hexdigest() == fit["final_parameter_sha256"]
    assert fits[0]["initial_parameter_sha256"] == fits[1]["initial_parameter_sha256"]
    assert fits[0]["task_order_sha256"] == fits[1]["task_order_sha256"]
    assert len({(r["condition"], r["phase"], r["case"], r["world"], r["label_order"]) for r in decisions}) == 768
    for condition in ("literal", "latent"):
        for phase in ("initial", "final"):
            for split, count in (("train", 128), ("dev", 64)):
                panel = [r for r in decisions if (r["condition"], r["phase"], r["split"]) == (condition, phase, split)]
                assert len(panel) == count
                groups, pairs = defaultdict(list), defaultdict(list)
                for row in panel:
                    assert row["stage"] == "stage2" and row["frame"] == "self" and row["wording"] == 0
                    assert row["greedy_label"] in (None, "A", "B")
                    assert all(np.isfinite(row[label]) for label in ("A", "B"))
                    correct = row["greedy_label"] == row["gold"]
                    groups[row["group"]].append(correct)
                    pairs[row["group"], row["world"], row["label_order"]].append(correct)
                assert all(len(g) == 32 for g in groups.values()) and all(len(pair) == 4 for pair in pairs.values())
                forced = ["A" if r["A"] > r["B"] else ("B" if r["B"] > r["A"] else None) for r in panel]
                native_correct, compound_correct = sum(sum(g) for g in groups.values()), sum(all(p) for p in pairs.values())
                expected = {"n": count, "native_correct": native_correct, "native_accuracy": fmean(fmean(g) for g in groups.values()),
                    "forced_correct": sum(pred == row["gold"] for pred, row in zip(forced, panel, strict=True)),
                    "forced_ties": sum(pred is None for pred in forced),
                    "native_label_coverage": sum(r["greedy_label"] in ("A", "B") for r in panel),
                    "compound_n": len(pairs), "native_compound_correct": compound_correct,
                    "native_compound_accuracy": compound_correct / len(pairs)}
                assert summary[condition][phase][split] == expected
        trained = summary[condition]["final"]["train"]
        positive = trained["native_accuracy"] >= .95 and trained["native_compound_accuracy"] >= .90
        assert summary[condition]["engineering_trainability_positive"] == terminal["engineering_trainability_positive"][condition] == positive
    assert not terminal["G_trained"] and not terminal["qualification_content_loaded"]
    assert terminal["backbone_sha256_before"] == terminal["backbone_sha256_after"] == read(STUDY / "recipient/terminal.json")["backbone_sha256_before"]
    assert read(folder / "native_parity.json")["exact"]
    for name in ("NOTICE.txt", "ADAPTER_LICENSE.md", "GEMMA_TERMS.html", "GEMMA_PROHIBITED_USE_POLICY.html"):
        assert (folder / name).is_file()
