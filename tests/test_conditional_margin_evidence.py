"""Bind the margin continuation to its closed parent and its newly sealed fresh panel."""

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
STUDY = ROOT / "provenance/conditional_margin_20261103"
RUN = STUDY / "run"
PARENT = ROOT / "provenance/conditional_curriculum_20261103/run"
FIXTURE = ROOT / "data/conditional_margin_20261103.json"
OLD_FIXTURE = ROOT / "data/conditional_curriculum_20261103.json"
ACTIONS = ROOT / "data/within_topic_wording_20261103.json"
PARENT_SHA = "4fb28b1c00ccebf89387f3dfb1793bef3c9ea8bedf782cb0d359e23943cd7d5c"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_margin_archive_reduction_and_exact_parent_endpoint_continuity():
    prefix = STUDY.relative_to(ROOT).as_posix()
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z", "--", prefix], cwd=ROOT)
        .decode()
        .split("\0")
    )
    names = {n.removeprefix(prefix + "/") for n in tracked if n}
    lines = (STUDY / "SHA256SUMS").read_text().splitlines()
    manifest = {name: sha for sha, name in (line.split("  ", 1) for line in lines)}
    assert len(manifest) == len(lines) and set(manifest) == names - {"SHA256SUMS"}
    assert all(file_sha(STUDY / n) == sha for n, sha in manifest.items())
    reduced = json.loads(
        subprocess.check_output([sys.executable, "-S", str(STUDY / "summarize.py")])
    )
    assert reduced == read(STUDY / "summary.json")
    request, seal, end = (
        read(RUN / n) for n in ("request.json", "source.json", "terminal.json")
    )
    assert (
        seal["git_commit"] == "d5d152f4dfae21f0fccf60fe864a5e0e90cf84f7"
        and not seal["git_status"]
    )
    assert request["fixture_sha256"] == file_sha(FIXTURE)
    assert request["plan_sha256"] == file_sha(STUDY / "PLAN.md") and request[
        "reducer_sha256"
    ] == file_sha(STUDY / "summarize.py")
    assert request["margin"] == 2.0 and request["optimizer_reset"] is True
    assert (
        request["warm_start_sha256"] == PARENT_SHA == file_sha(PARENT / "warm_task.npz")
    )
    init = read(RUN / "parent_initialization.json")
    assert (
        init["artifact_sha256"] == PARENT_SHA
        and init["parameter_sha256"] == read(PARENT / "warm_parameters.json")["sha256"]
    )
    assert (
        init["parent_literal_gate"] is False
        and read(PARENT / "warm_gate.json")["passed"] is False
    )
    groups = read(ACTIONS)["groups"]
    old_fit, new_fit = (
        list(make_cases(groups, read(f), "fit")) for f in (OLD_FIXTURE, FIXTURE)
    )
    assert old_fit == new_fit and len(new_fit) == 480
    key = lambda r: (r["key"], r["world"], r["label_order"])
    initial = {
        key(r): r
        for r in rows(RUN / "decisions.jsonl")
        if (r["arm"], r["endpoint"], r["split"], r["route"])
        == ("warm", "initial", "fit", "literal")
    }
    previous = {
        key(r): r
        for r in rows(PARENT / "decisions.jsonl")
        if (r["arm"], r["endpoint"], r["split"], r["route"])
        == ("warm", "final", "fit", "literal")
    }
    assert initial.keys() == previous.keys() and len(initial) == 3840
    fields = (
        "A",
        "B",
        "greedy_token",
        "greedy_label",
        "gold",
        "source_sha256",
        "question_sha256",
        "injected_sha256",
    )
    assert all(all(initial[k][f] == previous[k][f] for f in fields) for k in initial)
    assert (
        end["status"] == "COMPLETED_CONDITIONAL_MARGIN"
        and end["backbone_before"] == end["backbone_after"]
    )
    assert not end["G_trained"] and not end["qualification_content_loaded"]
    assert (
        end["literal_gate"]
        == reduced["gates"]["warm"]["passed"]
        == read(RUN / "warm_gate.json")["passed"]
    )
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


def test_margin_steps_keep_all_worlds_and_fixed_adapter_scopes():
    fixture = read(FIXTURE)
    groups = {g["id"]: g for g in read(ACTIONS)["groups"]}
    fit = {c["key"]: c for c in make_cases(list(groups.values()), fixture, "fit")}
    schedule = read(RUN / "schedule.json")
    assert read(RUN / "request.json")["schedule_sha256"] == digest(schedule)
    assert len(schedule["warm"]) == 384 and len(schedule["adaptation"]) == 256
    warm = read(RUN / "warm_gate.json")["passed"]
    expected = {"warm": 384} | ({"frozen": 256, "joint": 256} if warm else {})
    history = rows(RUN / "training.jsonl")
    assert Counter(r["phase"] for r in history) == expected
    for phase, count in expected.items():
        selected = [r for r in history if r["phase"] == phase]
        stage_schedule = schedule["warm" if phase == "warm" else "adaptation"]
        assert [r["step"] for r in selected] == list(range(1, count + 1))
        for i, (r, units) in enumerate(zip(selected, stage_schedule, strict=True)):
            assert (
                r["units"] == units
                and len(units) == 2
                and all(u[1] == STAGES[i % 3] for u in units)
            )
            chosen = [fit[k] for k in r["source_keys"]]
            assert len(chosen) == len(set(r["source_keys"])) == 4
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
            ] == [(*u, p) for u in units for p in (0, 1)]
            tasks = [
                t
                for c in chosen
                for w in range(4)
                for t in tasks_for(c, groups[c["group"]], fixture, w)
            ]
            assert len(tasks) == 32 and r["tasks_sha256"] == digest(
                [{k: v for k, v in t.items() if k != "source"} for t in tasks]
            )
            assert r["margin"] == 2.0 and r["margin_weighting"] == "uniform_worlds"
            assert len(r["micro_losses"]) == 8 and np.isfinite(r["micro_losses"]).all()
            assert np.isclose(
                r["weighted_loss"], sum(r["micro_losses"]), rtol=0, atol=1e-12
            )
            assert np.isfinite(r["task_gradient_norm"]) and np.isfinite(
                r["gradient_norm"]
            )
            assert (
                r["forwards"] == (16 if phase == "joint" else 8)
                and r["processed_tokens"] > 0
            )
            if phase == "joint":
                assert (
                    np.isfinite(r["source_gradient_norm"])
                    and len(r["source_layouts"]) == 8
                )
                assert [v["key"] for v in r["source_layouts"]] == [
                    c["key"] for c in chosen for _ in range(2)
                ]
                assert all(
                    len(v["positions"]) == 8
                    and v["positions"][-1] < v["forward_length"]
                    for v in r["source_layouts"]
                )
            else:
                assert r["source_gradient_norm"] is None and not r["source_layouts"]
    if warm:
        initial = read(RUN / "warm_parameters.json")
        branches = rows(RUN / "branch_initializations.jsonl")
        assert len(branches) == 2 and {r["arm"] for r in branches} == {
            "frozen",
            "joint",
        }
        assert all(
            r["task_parameter_sha256"] == initial["sha256"]
            and r["warm_artifact_sha256"] == initial["artifact_sha256"]
            for r in branches
        )


def test_margin_decisions_and_safe_artifacts_bind_fresh_names_and_each_donor():
    fixture = read(FIXTURE)
    groups = {g["id"]: g for g in read(ACTIONS)["groups"]}
    cases = {
        c["key"]: c
        for split in ("fit", "fresh")
        for c in make_cases(list(groups.values()), fixture, split)
    }
    assert len(cases) == 640
    captures = rows(RUN / "captures.jsonl") if (RUN / "captures.jsonl").exists() else []
    indexed = {(r["bank"], r["key"]): r for r in captures}
    assert len(indexed) == len(captures) and all(
        r["source_sha256"] == digest(cases[r["key"]]["source"]) for r in captures
    )
    host = read(RUN / "host.json")
    zero = State(
        slots=torch.zeros(host["slots"], host["width"], dtype=torch.bfloat16)
    ).sha()
    means = {}
    for r in rows(RUN / "decisions.jsonl"):
        c = cases[r["key"]]
        t = tasks_for(c, groups[c["group"]], fixture, r["world"])[r["label_order"]]
        assert r["gold"] == t["gold"] and r["question_sha256"] == digest(t["question"])
        assert r["source_sha256"] == digest(c["source"]) and r["split"] == c["split"]
        if r["route"] == "literal":
            assert r["injected_sha256"] == digest(c["source"])
        elif r["route"] == "zero":
            assert r["injected_sha256"] == zero
        elif r["route"] == "mean":
            means.setdefault((r["arm"], r["stage"]), set()).add(r["injected_sha256"])
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
            assert r["injected_sha256"] == indexed[bank, donor["key"]]["state_sha256"]
    assert all(len(v) == 1 for v in means.values())
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
            with np.load(path, allow_pickle=False) as a:
                m = json.loads(a["metadata_utf8"].tobytes())
                assert m["base_host_sha256"] == digest(host) and m[
                    "dataset_sha256"
                ] == file_sha(FIXTURE)
                assert m["source_sha256"] == read(RUN / "source.json")["source_sha256"]
                assert (
                    m["architecture"]["rank"] == 16
                    and m["architecture"]["last_layers"] == 4
                )
                assert m["kind"] == (
                    "scoped_source_encoder_qv_lora"
                    if name == "joint_source.npz"
                    else "scoped_recipient_qv_lora"
                )
                assert (
                    m["training"]["phase"] == record["phase"]
                    and m["training"]["fixed_endpoint"]
                )
                assert m["training"]["schedule_file_sha256"] == file_sha(
                    RUN / "schedule.json"
                )
                values = [a[k] for k in a.files if k != "metadata_utf8"]
                assert len(values) == 16 and all(
                    v.ndim == 2 and v.dtype == np.float32 and np.isfinite(v).all()
                    for v in values
                )
