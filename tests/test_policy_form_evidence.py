"""Validate the archived paired comparison without loading a model."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from selffold.records import digest, file_sha

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/policy_form_20261103"
PRIOR = ROOT / "provenance/continuity_20261103"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_policy_form_archive_and_summary_replay():
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", str(STUDY.relative_to(ROOT))], cwd=ROOT
    )
    paths = {
        Path(x.decode()).relative_to(STUDY.relative_to(ROOT)).as_posix()
        for x in tracked.split(b"\0")
        if x
    }
    manifest = (STUDY / "SHA256SUMS").read_text().splitlines()
    expected = {name: sha for sha, name in (line.split("  ", 1) for line in manifest)}
    assert len(expected) == len(manifest)
    assert set(expected) == paths - {"SHA256SUMS"}
    for name, sha in expected.items():
        assert hashlib.sha256((STUDY / name).read_bytes()).hexdigest() == sha
    assert (
        subprocess.check_output([sys.executable, str(STUDY / "summarize.py")])
        == (STUDY / "summary.json").read_bytes()
    )


def test_case_list_fits_match_reference_schedule_and_replay_the_rule_control():
    run = STUDY / "run"
    prior = PRIOR / "trainability"
    seal, request = read(run / "source.json"), read(run / "request.json")
    assert seal["git_status"] == "" and digest(seal["files"]) == seal["source_sha256"]
    assert request["plan_sha256"] == file_sha(STUDY / "PLAN.md")
    assert request["prior_decisions_sha256"] == file_sha(prior / "decisions.jsonl")
    assert request["schedule_sha256"] == digest(read(prior / "schedule.json"))
    original = {r["condition"]: r for r in rows(prior / "fits.jsonl")}
    training = rows(run / "training.jsonl")
    for fit in rows(run / "fits.jsonl"):
        condition = fit["condition"]
        old = original[condition]
        assert fit["initial_parameter_sha256"] == old["initial_parameter_sha256"]
        assert fit["task_order_sha256"] == old["task_order_sha256"]
        history = [x for x in training if x["condition"] == condition]
        assert [x["step"] for x in history] == list(range(1, 65))
        assert digest([x["tasks_sha256"] for x in history]) == old["task_order_sha256"]
        assert [x["group"] for x in history] == read(prior / "schedule.json")
        assert fit["training_forwards"] == old["training_forwards"] == 256
        path = run / fit["artifact"]
        assert file_sha(path) == fit["sha256"]
        with np.load(path, allow_pickle=False) as artifact:
            meta = json.loads(artifact["metadata_utf8"].tobytes())
            assert meta["source_sha256"] == seal["source_sha256"]
            assert meta["dataset_sha256"] == request["data_sha256"]
            assert meta["base_host_sha256"] == digest(read(run / "host.json"))
            assert meta["training"]["form"] == "forward"
    old_decisions = {
        (r["condition"], r["phase"], r["case"], r["world"], r["label_order"]): r
        for r in rows(prior / "decisions.jsonl")
    }
    captured = {(r["form"], r["key"]): r for r in rows(run / "donors.jsonl")}
    seen = set()
    for r in rows(run / "decisions.jsonl"):
        ref = old_decisions[
            r["condition"], "initial", r["case"], r["world"], r["label_order"]
        ]
        assert (
            r["gold"] == ref["gold"] and r["question_sha256"] == ref["question_sha256"]
        )
        assert (
            r["source_sha256"]
            == captured[r["form"], "stage2/" + r["case"]]["source_sha256"]
        )
        if r["form"] == "rule" and r["endpoint"] in ("initial", "rule_fit"):
            key = (
                r["condition"],
                ("initial" if r["endpoint"] == "initial" else "final"),
                r["case"],
                r["world"],
                r["label_order"],
            )
            assert all(
                r[k] == old_decisions[key][k]
                for k in (
                    "source_sha256",
                    "question_sha256",
                    "gold",
                    "A",
                    "B",
                    "greedy_token",
                )
            )
            assert key not in seen
            seen.add(key)
    assert seen == old_decisions.keys()
    donors = {r["key"]: r for r in rows(PRIOR / "recipient/donors.jsonl")}
    captures = rows(run / "donors.jsonl")
    assert len(captures) == 72
    for r in captures:
        assert all(
            r[k] == donors[r["reference"]][k] for k in ("source_sha256", "state_sha256")
        )
    end = read(run / "terminal.json")
    assert end["backbone_sha256_before"] == end["backbone_sha256_after"]
    assert (
        end["updates"] == 128
        and not end["G_trained"]
        and not end["qualification_content_loaded"]
    )
