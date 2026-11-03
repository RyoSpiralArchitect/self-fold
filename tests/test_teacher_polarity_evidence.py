"""Verify the public study can be reconstructed from its sealed fixtures and native rows."""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from selffold.polarity_readout import predicted_class, source_cases
from selffold.records import digest, file_sha

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/teacher_polarity_20261103"
RUN = STUDY / "run"


def read(file):
    return json.loads(file.read_text())


def rows(file):
    return [json.loads(line) for line in file.read_text().splitlines()]


def test_teacher_polarity_public_manifest_and_full_reduction():
    prefix = str(STUDY.relative_to(ROOT))
    names = (
        subprocess.check_output(["git", "ls-files", "-z", "--", prefix], cwd=ROOT)
        .decode()
        .split("\0")
    )
    paths = {n.removeprefix(prefix + "/") for n in names if n}
    actual = {n: file_sha(STUDY / n) for n in paths - {"SHA256SUMS"}}
    listed = [
        line.split("  ", 1) for line in (STUDY / "SHA256SUMS").read_text().splitlines()
    ]
    assert len(listed) == len(actual) and actual == {n: h for h, n in listed}
    reduced = json.loads(
        subprocess.check_output([sys.executable, "-S", str(STUDY / "summarize.py")])
    )
    assert reduced == read(STUDY / "summary.json")
    paired = json.loads(
        subprocess.check_output(
            [sys.executable, "-S", str(STUDY / "paired_diagnostics.py")]
        )
    )
    assert paired == read(STUDY / "paired_diagnostics.json")
    end = read(RUN / "terminal.json")
    assert end["status"] == "COMPLETED_TEACHER_POLARITY"
    assert end["frozen_parameter_sha256_before"] == end["frozen_parameter_sha256_after"]
    assert end["teacher_gate"] == reduced["gates"]["teacher"]
    assert end["source_gate"] == reduced["gates"]["source"]
    assert end["joint_executed"] == (end["teacher_gate"] and end["source_gate"])
    assert not end["qualification_content_loaded"] and not end["G_trained"]


def test_teacher_screen_uses_exposed_questions_only_and_native_pairs_replay():
    fixture = read(ROOT / "data/teacher_polarity_20261103.json")
    prior = ROOT / fixture["prior_fixture"]
    groups = read(prior)["groups"]
    request = read(RUN / "request.json")
    assert request["fixture_sha256"] == file_sha(
        ROOT / "data/teacher_polarity_20261103.json"
    )
    assert request["prior_fixture_sha256"] == file_sha(prior)
    assert request["plan_sha256"] == file_sha(STUDY / "PLAN.md")
    assert request["reducer_sha256"] == file_sha(STUDY / "summarize.py")
    seal = read(RUN / "source.json")
    assert seal["git_commit"] == "270fe5b64fea86639a53b643bc48103b170fa072"
    assert not seal["git_status"] and digest(seal["files"]) == seal["source_sha256"]
    history = rows(RUN / "training.jsonl")
    schedule = read(RUN / "schedule.json")
    assert len(history) == len(schedule) == 64
    for step, (r, blocks) in enumerate(zip(history, schedule, strict=True), 1):
        assert r["step"] == step and r["blocks"] == blocks and len(blocks) == 4
        assert np.isfinite(r["losses"]).all() and np.isfinite(r["gradient_norm"])
    for start in range(0, 64, 8):
        assert sorted(
            tuple(b) for blocks in schedule[start : start + 8] for b in blocks
        ) == [(t, q) for t in range(8) for q in range(4)]
    pairs = rows(RUN / "native_pairs.jsonl")
    idx = {r["pair_sha256"]: r for r in pairs}
    assert len(idx) == len(pairs)
    for pair in pairs:
        assert pair["pair_sha256"] == digest(
            [pair["injected_sha256"], pair["question_sha256"]]
        )
    panel = rows(RUN / "teachers.jsonl")
    assert len(panel) == 1024
    for r in panel + (
        rows(RUN / "joint.jsonl") if (RUN / "joint.jsonl").exists() else []
    ):
        g = groups[r["topic"]]
        wordings = (
            g["questions"]
            if r["split"] == "fit"
            else fixture["generic_questions"] + fixture["specific_questions"][g["id"]]
        )
        choices = g["options"] if r["label_order"] == 0 else g["options"][::-1]
        question = f"{wordings[r['wording']]}\nA = {choices[0]}. B = {choices[1]}. Use the supplied description."
        pair = idx[r["pair_sha256"]]
        assert (
            r["question_sha256"]
            == digest(question)
            == pair["question_sha256"][r["label_order"]]
        )
        assert r["injected_sha256"] == pair["injected_sha256"]
        assert all(r[k] == v for k, v in pair["outputs"][r["label_order"]].items())
        assert r["gold"] == ("A" if r["pole"] == r["label_order"] else "B")
    renderer = read(RUN / "renderer.json")
    assert renderer["artifact_sha256"] == file_sha(RUN / "renderer.npz")
    assert renderer["parameters"] == 18432 and renderer["updates"] == 64
    assert (
        renderer["rows_exact_all_teachers"] and renderer["fitting_native_logits_exact"]
    )
    assert len(rows(RUN / "lexical_replay.jsonl")) == 24


def test_source_labels_and_axes_are_bound_without_source_anchors():
    records = rows(RUN / "sources.jsonl")
    fixture = read(ROOT / "data/teacher_polarity_20261103.json")
    prior = read(ROOT / fixture["prior_fixture"])
    forms = {f: prior["styles"][f] for f in prior["fit_styles"]}
    expected = {
        (split, c["key"]): digest(c["source"])
        for split, vocab in (("fit", forms), ("fresh", fixture["fresh_frames"]))
        for c in source_cases(prior["groups"], vocab, fixture["familiar_pairs"])
    }
    assert expected == {(r["split"], r["key"]): r["source_sha256"] for r in records}
    archive = {
        r["source_sha256"]: r["state_sha256"]
        for r in rows(ROOT / "provenance/within_topic_20261103/run/sources.jsonl")
    }
    repeated = [r for r in records if r["source_sha256"] in archive]
    assert len(repeated) == 224 and all(
        r["split"] == "fit" and r["state_sha256"] == archive[r["source_sha256"]]
        for r in repeated
    )

    assert Counter(r["split"] for r in records) == {"fit": 512, "fresh": 256}
    assert len({r["source_sha256"] for r in records}) == 768
    for r in records:
        scores = r["scores"]

        for method in ("direct", "mention", "factor"):
            assert r[method] == predicted_class(scores, method)
        assert r["oracle_factor"] == predicted_class(scores, oracle_topic=r["topic"])
    axes = read(RUN / "axes.json")
    assert axes["artifact_sha256"] == file_sha(RUN / "axes.npz")
    assert axes["fit_state_sha256"] == digest(
        [r["state_sha256"] for r in records if r["split"] == "fit"]
    )
    assert axes["score_replay"] == 512
    with np.load(RUN / "axes.npz", allow_pickle=False) as f:
        assert set(f.files) == {"mean", "scale", "shared", "offset", "slopes", "biases"}
        assert f["slopes"].shape == (8, 18432, 2) and f["shared"].shape == (18432, 9)
        assert all(np.isfinite(f[k]).all() for k in f.files)
