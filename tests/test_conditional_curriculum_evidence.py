"""Bind curriculum scores and updates to the declared split, scope and saved adapters."""

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from selffold.conditional_curriculum import STAGES, make_cases, tasks_for
from selffold.records import State, digest, file_sha

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/conditional_curriculum_20261103"
RUN = STUDY / "run"
FIXTURE = ROOT / "data/conditional_curriculum_20261103.json"
ACTIONS = ROOT / "data/within_topic_wording_20261103.json"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_curriculum_public_archive_and_registered_branch_reduction():
    prefix = STUDY.relative_to(ROOT).as_posix()
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z", "--", prefix], cwd=ROOT)
        .decode()
        .split("\0")
    )
    names = {n.removeprefix(prefix + "/") for n in tracked if n}
    lines = (STUDY / "SHA256SUMS").read_text().splitlines()
    manifest = {n: h for h, n in (line.split("  ", 1) for line in lines)}
    assert len(lines) == len(manifest) and set(manifest) == names - {"SHA256SUMS"}
    assert all(file_sha(STUDY / n) == h for n, h in manifest.items())
    reduced = json.loads(
        subprocess.check_output([sys.executable, "-S", str(STUDY / "summarize.py")])
    )
    assert reduced == read(STUDY / "summary.json")
    request, source, end = (
        read(RUN / n) for n in ("request.json", "source.json", "terminal.json")
    )
    assert (
        source["git_commit"] == "94473d4c775a01c9e60cf83ccccdf615bf073121"
        and source["git_status"] == ""
    )
    assert request["fixture_sha256"] == file_sha(FIXTURE)
    assert request["plan_sha256"] == file_sha(STUDY / "PLAN.md")
    assert request["reducer_sha256"] == file_sha(STUDY / "summarize.py")
    assert request["schedule_sha256"] == digest(read(RUN / "schedule.json"))
    assert end["status"] == "COMPLETED_CONDITIONAL_CURRICULUM"
    assert end["backbone_before"] == end["backbone_after"]
    assert not end["G_trained"] and not end["qualification_content_loaded"]
    assert read(RUN / "warm_gate.json") == reduced["gates"]["warm"]
    assert end["literal_gate"] == reduced["gates"]["warm"]["passed"]
    if end["literal_gate"]:
        assert read(RUN / "branches_closed.json") == {
            "fresh_decisions": 0,
            "both_fixed_endpoints_saved": True,
        }
        assert read(RUN / "source_parity.json")["before_source_updates"] == 0
    else:
        assert (
            not (RUN / "branches_closed.json").exists()
            and not (RUN / "captures.jsonl").exists()
        )


def test_curriculum_steps_use_only_fitting_sources_and_the_sealed_objective_scope():
    fixture = read(FIXTURE)
    groups = {g["id"]: g for g in read(ACTIONS)["groups"]}
    cases = {c["key"]: c for c in make_cases(list(groups.values()), fixture, "fit")}
    schedule = read(RUN / "schedule.json")
    history = rows(RUN / "training.jsonl")
    warm = read(RUN / "warm_gate.json")["passed"]
    expected = {"warm": 384} | ({"frozen": 256, "joint": 256} if warm else {})
    assert Counter(r["phase"] for r in history) == expected
    assert len(schedule["warm"]) == 384 and len(schedule["adaptation"]) == 256
    warm_stages = ["cases"] * 64 + ["rule"] * 64 + [STAGES[i % 3] for i in range(256)]
    for phase, count in expected.items():
        selected = [r for r in history if r["phase"] == phase]
        plan = schedule["warm" if phase == "warm" else "adaptation"]
        stages = warm_stages if phase == "warm" else [STAGES[i % 3] for i in range(256)]
        assert [r["step"] for r in selected] == list(range(1, count + 1))
        for record, units, stage in zip(selected, plan, stages, strict=True):
            assert record["units"] == units and len(units) == 2
            chosen = [cases[k] for k in record["source_keys"]]
            assert len(chosen) == len(set(record["source_keys"])) == 4
            assert [
                (
                    c["group"],
                    c["stage"],
                    c["flag_pair"],
                    c["style"],
                    c["source_order"],
                    c["pole"],
                )
                for c in chosen
            ] == [(*unit, pole) for unit in units for pole in (0, 1)]
            assert all(c["stage"] == stage and c["split"] == "fit" for c in chosen)
            tasks = [
                t
                for c in chosen
                for world in range(4)
                for t in tasks_for(c, groups[c["group"]], fixture, world)
            ]
            bindings = [{k: v for k, v in t.items() if k != "source"} for t in tasks]
            assert len(tasks) == 32 and record["tasks_sha256"] == digest(bindings)
            assert (
                len(record["micro_losses"]) == 8
                and np.isfinite(record["micro_losses"]).all()
            )
            assert np.isclose(
                record["weighted_loss"], sum(record["micro_losses"]), rtol=0, atol=1e-12
            )
            assert np.isfinite(record["task_gradient_norm"]) and np.isfinite(
                record["gradient_norm"]
            )
            assert (
                record["forwards"] == (16 if phase == "joint" else 8)
                and record["processed_tokens"] > 0
            )
            if phase == "joint":
                assert (
                    np.isfinite(record["source_gradient_norm"])
                    and len(record["source_layouts"]) == 8
                )
                assert [r["key"] for r in record["source_layouts"]] == [
                    c["key"] for c in chosen for _ in range(2)
                ]
                assert all(
                    len(r["positions"]) == 8
                    and r["positions"][-1] < r["forward_length"]
                    for r in record["source_layouts"]
                )
            else:
                assert (
                    record["source_gradient_norm"] is None
                    and not record["source_layouts"]
                )
    if warm:
        starts = rows(RUN / "branch_initializations.jsonl")
        initial = read(RUN / "warm_parameters.json")
        assert {r["arm"] for r in starts} == {"frozen", "joint"} and len(starts) == 2
        assert all(
            r["task_parameter_sha256"] == initial["sha256"]
            and r["warm_artifact_sha256"] == initial["artifact_sha256"]
            for r in starts
        )


def test_curriculum_native_answers_and_safe_arrays_bind_to_each_single_source():
    fixture = read(FIXTURE)
    groups = {g["id"]: g for g in read(ACTIONS)["groups"]}
    cases = {
        c["key"]: c
        for split in ("fit", "fresh")
        for c in make_cases(list(groups.values()), fixture, split)
    }
    assert len(cases) == 640
    captures = rows(RUN / "captures.jsonl") if (RUN / "captures.jsonl").exists() else []
    cap = {(r["bank"], r["key"]): r for r in captures}
    assert len(cap) == len(captures)
    assert all(
        r["source_sha256"] == digest(cases[r["key"]]["source"]) for r in captures
    )
    host = read(RUN / "host.json")
    zero = State(
        slots=torch.zeros(host["slots"], host["width"], dtype=torch.bfloat16)
    ).sha()
    mean_hashes = {}
    for r in rows(RUN / "decisions.jsonl"):
        c = cases[r["key"]]
        task = tasks_for(c, groups[c["group"]], fixture, r["world"])[r["label_order"]]
        assert r["gold"] == task["gold"] and r["question_sha256"] == digest(
            task["question"]
        )
        assert r["source_sha256"] == digest(c["source"]) and r["split"] == c["split"]
        if r["route"] == "literal":
            assert r["injected_sha256"] == digest(c["source"])
        elif r["route"] == "zero":
            assert r["injected_sha256"] == zero
        elif r["route"] == "mean":
            mean_hashes.setdefault((r["arm"], r["stage"]), set()).add(
                r["injected_sha256"]
            )
        else:
            donor = (
                c
                if r["route"] == "matched"
                else cases[
                    c["key"].rsplit("/", 2)[0] + f"/{1 - c['pole']}/{c['source_order']}"
                ]
            )
            bank = (
                ("frozen_fit" if r["split"] == "fit" else "frozen_fresh")
                if r["arm"] == "frozen"
                else "joint_" + r["split"]
            )
            assert r["injected_sha256"] == cap[bank, donor["key"]]["state_sha256"]
    assert all(len(v) == 1 for v in mean_hashes.values())
    warm = read(RUN / "warm_gate.json")["passed"]
    fits = rows(RUN / "fits.jsonl")
    assert {r["phase"] for r in fits} == (
        {"warm", "frozen", "joint"} if warm else {"warm"}
    )
    for record in fits:
        assert record["rows_exact"] and record["native_logits_exact"]
        expected = {record["phase"] + "_task.npz"} | (
            {"joint_source.npz"} if record["phase"] == "joint" else set()
        )
        assert set(record["artifacts"]) == expected
        for name, sha in record["artifacts"].items():
            path = RUN / name
            assert file_sha(path) == sha
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(archive["metadata_utf8"].tobytes())
                assert meta["base_host_sha256"] == digest(host) and meta[
                    "dataset_sha256"
                ] == file_sha(FIXTURE)
                assert (
                    meta["source_sha256"] == read(RUN / "source.json")["source_sha256"]
                )
                architecture = meta["architecture"]
                assert architecture["rank"] == 16 and architecture["last_layers"] == 4
                assert meta["kind"] == (
                    "scoped_source_encoder_qv_lora"
                    if name == "joint_source.npz"
                    else "scoped_recipient_qv_lora"
                )
                assert (
                    meta["training"]["phase"] == record["phase"]
                    and meta["training"]["fixed_endpoint"]
                )
                assert meta["training"]["schedule_file_sha256"] == file_sha(
                    RUN / "schedule.json"
                )
                arrays = [archive[k] for k in archive.files if k != "metadata_utf8"]
                assert len(arrays) == 16 and all(
                    a.ndim == 2 and a.dtype == np.float32 and np.isfinite(a).all()
                    for a in arrays
                )
