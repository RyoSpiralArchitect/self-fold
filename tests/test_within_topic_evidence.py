"""Bind both fitting arms, the stopped initialization and the prospective reducer."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from selffold.records import digest, file_sha

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/within_topic_20261103"
RUN = STUDY / "run"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def assert_same_result(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same_result(actual[key], expected[key])
    elif isinstance(expected, float):
        # Tolerate final-bit summation differences in descriptive means across runtimes.
        assert math.isclose(actual, expected, rel_tol=1e-13, abs_tol=1e-13)
    else:
        assert actual == expected


def test_within_topic_archive_and_raw_decision_reduction():
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
    regenerated = json.loads(
        subprocess.check_output([sys.executable, str(STUDY / "summarize.py")])
    )
    assert_same_result(regenerated, read(STUDY / "summary.json"))


def test_within_topic_arms_use_the_sealed_schedule_and_no_fresh_sources():
    request = read(RUN / "request.json")
    assert request["fixture_sha256"] == file_sha(
        ROOT / "data/within_topic_wording_20261103.json"
    )
    assert request["plan_sha256"] == file_sha(STUDY / "PLAN.md") and request[
        "scorer_sha256"
    ] == file_sha(STUDY / "summarize.py")
    end = read(RUN / "terminal.json")
    assert end["status"] == "COMPLETED_WITHIN_TOPIC_WORDING" and end["updates"] == 128
    assert (
        end["backbone_sha256_before"] == end["backbone_sha256_after"]
        and end["lexical_sha256_before"] == end["lexical_sha256_after"]
    )
    assert not end["G_trained"] and not end["qualification_content_loaded"]
    schedule = read(RUN / "schedule.json")
    units = [tuple(u) for step in schedule for u in step]
    assert len(schedule) == 64 and len(set(units)) == len(units) == 256
    history = rows(RUN / "training.jsonl")
    sources = rows(RUN / "sources.jsonl")
    for arm in ("single", "many"):
        selected = [r for r in history if r["arm"] == arm]
        fit_sources = {
            r["key"]: r for r in sources if r["arm"] == arm and r["part"] == "fit"
        }
        assert [r["step"] for r in selected] == list(range(1, 65))
        seen = Counter()
        for record, step in zip(selected, schedule, strict=True):
            keys = record["source_keys"]
            assert len(keys) == 8 and set(keys) <= fit_sources.keys()
            assert len(record["native_losses"]) == 8 and np.isfinite(
                record["gradient_norm"]
            )
            for key, (topic, a, b, order) in zip(
                keys, [u for u in step for _ in range(2)], strict=True
            ):
                assert key.startswith(f"{topic}/fit/{a}-{b}/") and key.endswith(
                    "/" + str(order)
                )
            seen.update(keys)
        assert set(seen) == fit_sources.keys() and set(seen.values()) == {1}
    compute = rows(RUN / "training_compute.jsonl")
    assert len(compute) == 2 and all(r["forwards"] == 512 for r in compute)
    assert compute[0]["processed_tokens"] == compute[1]["processed_tokens"]


def test_stopped_layout_replay_and_compact_artifacts_are_bound():
    first = STUDY / "attempt_01"
    assert (
        read(first / "source.json")["git_commit"]
        == "3a7d4fc98752023423b4be75a32f415b77f5525d"
    )
    assert read(first / "terminal.json")["error"] == "Mixture serialization mismatch"
    assert (
        not (first / "training.jsonl").exists()
        and not (first / "decisions.jsonl").exists()
    )
    repaired = read(RUN / "repair_replay.json")
    assert repaired["updates_before_replay"] == 0
    for name in repaired["files_exact"]:
        assert file_sha(first / name) == file_sha(RUN / name)
    for source in (first, RUN):
        seal = read(source / "source.json")
        assert (
            seal["git_status"] == "" and digest(seal["files"]) == seal["source_sha256"]
        )
    assert (
        read(RUN / "source.json")["git_commit"]
        == "a813911f6c4dd725b7ad3e8d03bdf3057af272a4"
    )
    diagnosis = read(STUDY / "layout_diagnosis.json")
    assert (
        diagnosis["identical_tensor_values"]
        and not diagnosis["C_copy_rows_exact"]
        and diagnosis["C_copy_logits_exact"]
    )
    assert diagnosis["K_copy_rows_exact"] and diagnosis["K_copy_logits_exact"]
    snapshots = {}
    for fit in rows(RUN / "fits.jsonl"):
        path = RUN / fit["artifact"]
        assert (
            fit["rows_exact"]
            and fit["logits_exact"]
            and file_sha(path) == fit["sha256"]
        )
        with np.load(path, allow_pickle=False) as artifact:
            meta = json.loads(artifact["metadata_utf8"].tobytes())
            assert meta["source_sha256"] == read(RUN / "source.json")["source_sha256"]
            assert meta["base_host_sha256"] == digest(read(RUN / "host.json"))
            assert (
                meta["lexical_sha256"] == read(RUN / "request.json")["lexical_sha256"]
            )
            assert (
                meta["fit_stats"]["sources_per_policy"] == [32] * 16
                and meta["fit_stats"]["trainable_parameters"] == 18688
            )
            assert set(artifact.files) == {
                "metadata_utf8",
                "calibration",
                "mean",
                "scale",
                "projection",
                "bias",
                "teachers",
                "renderer.down",
                "renderer.up",
            }
            assert artifact["projection"].shape == (18432, 16) and artifact[
                "teachers"
            ].shape == (16, 8, 2304)
            snapshots[fit["arm"], fit["endpoint"]] = {
                k: artifact[k]
                for k in ("mean", "scale", "projection", "bias", "teachers")
            }
    for arm in ("single", "many"):
        for k, v in snapshots[arm, "initial"].items():
            assert np.array_equal(v, snapshots[arm, "final"][k])
    prev = ROOT / "provenance/source_lexical_20261103/run"
    prior = {
        (r["group"], r["pole"]): r["state_sha256"]
        for r in rows(prev / "teachers.jsonl")
    }
    assert all(
        prior[r["group"], r["pole"]] == r["state_sha256"]
        for r in rows(RUN / "teachers.jsonl")
    )
    assert len(rows(RUN / "lexical_replay.jsonl")) == 24
