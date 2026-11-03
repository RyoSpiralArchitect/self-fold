"""Bind the source fit, operation comparison and actual conditional teachers to their seals."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from selffold.action_swap import action_pairs
from selffold.records import canonical, digest, file_sha
from selffold.role_exchange import crossed_records

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/action_operation_20261103"
RUN = STUDY / "run"
sys.path.insert(0, str(ROOT / "scripts"))
from conditional_teacher_audit import audit_cases, tasks_for
from teacher_operation_panel import question_groups


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_action_operation_archive_and_registered_reduction_match_source_seal():
    prefix = STUDY.relative_to(ROOT).as_posix()
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z", "--", prefix], cwd=ROOT)
        .decode()
        .split("\0")
    )
    names = {n.removeprefix(prefix + "/") for n in tracked if n}
    lines = (STUDY / "SHA256SUMS").read_text().splitlines()
    manifest = {n: h for h, n in (line.split("  ", 1) for line in lines)}
    assert len(manifest) == len(lines) and set(manifest) == names - {"SHA256SUMS"}
    assert all(file_sha(STUDY / n) == h for n, h in manifest.items())
    reduced = json.loads(
        subprocess.check_output([sys.executable, "-S", str(STUDY / "summarize.py")])
    )
    assert reduced == read(STUDY / "summary.json")
    paired = json.loads(
        subprocess.check_output(
            [sys.executable, "-S", str(STUDY / "paired_action_changes.py")]
        )
    )
    assert paired == read(STUDY / "paired_action_changes.json")
    request = read(RUN / "request.json")
    fixture_fields = {
        "fresh_fixture_sha256": "data/action_swap_20261103.json",
        "action_fixture_sha256": "data/within_topic_wording_20261103.json",
        "card_fixture_sha256": "data/matched_cues_20261103.json",
        "question_fixture_sha256": "data/question_operation_20261103.json",
        "conditional_fixture_sha256": "data/conditional_audit_20261103.json",
        "frozen_sha256": "provenance/role_exchange_20261103/run/weighted.npz",
        "plan_sha256": "provenance/action_operation_20261103/PLAN.md",
        "reducer_sha256": "provenance/action_operation_20261103/summarize.py",
    }
    assert all(request[k] == file_sha(ROOT / p) for k, p in fixture_fields.items())
    seal = read(RUN / "source.json")
    assert (
        seal["git_commit"] == "72ee1383fca6ef042923a97b05b7a7c4a0394867"
        and not seal["git_status"]
    )
    assert digest(seal["files"]) == seal["source_sha256"]
    end = read(RUN / "terminal.json")
    assert end["status"] == "COMPLETED_ACTION_OPERATION"
    assert (
        end["backbone_before"] == end["backbone_after"]
        and end["fixed_before"] == end["fixed_after"]
    )
    assert end["backbone_updates"] == request["backbone_updates"] == 0
    assert not end["G_trained"] and not end["qualification_content_loaded"]
    assert read(RUN / "fit_closed.json")["fresh_sources_encoded"] == 0
    assert read(RUN / "replay.json") == {
        "fit_states": 4608,
        "frozen_scores_exact": 4608,
        "fresh_sources_encoded": 0,
    }


def test_action_swap_sources_pairs_and_anchor_free_arrays_reconstruct_all_fitting_and_fresh_keys():
    groups = read(ROOT / "data/within_topic_wording_20261103.json")["groups"]
    cards = read(ROOT / "data/matched_cues_20261103.json")
    fresh = read(ROOT / "data/action_swap_20261103.json")["fresh_actions"]
    records = rows(RUN / "source_scores.jsonl")
    paired = rows(RUN / "action_pairs.jsonl")
    seen = set()
    for split, vocab, total in (("fit", None, 4608), ("fresh", fresh, 2048)):
        cases = list(crossed_records(groups, cards, vocab))
        expected = {c["key"]: digest(c["source"]) for c in cases}
        observed = [r for r in records if r["split"] == split]
        assert (
            len(observed) == total
            and {r["key"]: r["source_sha256"] for r in observed} == expected
        )
        assert seen.isdisjoint(expected.values())
        seen.update(expected.values())
        keys, ids = action_pairs(cases)
        pr = [r for r in paired if r["split"] == split]
        assert len(pr) == len(keys) == total // 2
        for key, (a, b), r in zip(keys, ids, pr, strict=True):
            assert tuple(r["stratum"]) == key
            assert (
                r["plus_first"] == cases[a]["key"]
                and r["minus_first"] == cases[b]["key"]
            )
            assert len(r["slots"]) == 8 and r["slots"][-1] < r["num_tokens"]
        assert all(len(r[arm]) == 8 for r in observed for arm in ("swap", "invariant"))
    old = {
        r["source_sha256"]: r
        for r in rows(ROOT / "provenance/role_exchange_20261103/run/scores.jsonl")
        if r["split"] == "fit"
    }
    fitted = [r for r in records if r["split"] == "fit"]
    assert all(
        r["state_sha256"] == old[r["source_sha256"]]["state_sha256"]
        and r["frozen"] == old[r["source_sha256"]]["weighted"]
        for r in fitted
    )
    capture_sha = digest([(r["source_sha256"], r["state_sha256"]) for r in fitted])
    closure = read(RUN / "fit_closed.json")
    fits = rows(RUN / "fits.jsonl")
    assert {r["arm"] for r in fits} == {"swap", "invariant"}
    for fit in fits:
        path = RUN / fit["artifact"]
        assert (
            fit["fit_capture_sha256"] == capture_sha
            and fit["fit_singleton_replays"] == 4608
        )
        assert file_sha(path) == fit["sha256"] == closure["artifact_sha256"][fit["arm"]]
        with np.load(path, allow_pickle=False) as a:
            assert set(a.files) == {"mean", "scale", "axes", "bias", "geometry"}
            assert a["geometry"].tolist() == [8, 2304] and a["axes"].shape == (18432, 8)
            assert a["mean"].shape == a["scale"].shape == (18432,) and a[
                "bias"
            ].shape == (8,)
            assert all(np.isfinite(a[k]).all() for k in a.files)


def test_teacher_native_pairs_and_conditional_world_labels_keep_all_topics_and_stages():
    groups = read(ROOT / "data/within_topic_wording_20261103.json")["groups"]
    query = question_groups(
        groups, read(ROOT / "data/question_operation_20261103.json")
    )
    pairs = rows(RUN / "native_pairs.jsonl")
    idx = {r["pair_sha256"]: r for r in pairs}
    assert len(idx) == len(pairs) and all(
        r["pair_sha256"] == digest([r["injected_sha256"], r["question_sha256"]])
        for r in pairs
    )
    teacher = rows(RUN / "teachers.jsonl")
    layout = {
        (r["topic"], r["family"], r["label_order"]): r
        for r in rows(RUN / "teacher_question_layout.jsonl")
    }
    assert (
        len(teacher)
        == len(
            {
                (
                    r["topic"],
                    r["family"],
                    r["mode"],
                    r["pole"],
                    r["label_order"],
                    r["route"],
                )
                for r in teacher
            }
        )
        == 512
    )
    assert len(layout) == 64
    donors = {}
    for r in teacher:
        t, p, o, f = r["topic"], r["pole"], r["label_order"], r["family"]
        mode = ("generic", "concrete").index(r["mode"])
        options = groups[t]["options"][::-1] if o else groups[t]["options"]
        question = (
            query[t]["fresh_questions"][2 * f + mode]
            + f"\nA = {options[0]}. B = {options[1]}. Use the supplied description."
        )
        pair = idx[r["pair_sha256"]]
        assert (
            r["question_sha256"]
            == digest(question)
            == pair["question_sha256"][o]
            == layout[t, f, o]["question_sha256"][mode]
        )
        assert r["injected_sha256"] == pair["injected_sha256"]
        assert all(r[k] == v for k, v in pair["outputs"][o].items()) and r["gold"] == (
            "A" if p == o else "B"
        )
        donors.setdefault((r["route"], t, p), set()).add(r["injected_sha256"])
    assert all(len(v) == 1 for v in donors.values())
    fixture = read(ROOT / "data/conditional_audit_20261103.json")
    cases = {c["key"]: c for c in audit_cases(groups, fixture)}
    sources = {r["key"]: r for r in rows(RUN / "audit_sources.jsonl")}
    assert len(cases) == len(sources) == 80
    assert {k: digest(c["source"]) for k, c in cases.items()} == {
        k: r["source_sha256"] for k, r in sources.items()
    }
    audit = rows(RUN / "audit_decisions.jsonl")
    assert (
        len(audit)
        == len({(r["key"], r["world"], r["label_order"], r["route"]) for r in audit})
        == 1920
    )
    zero = hashlib.sha256(
        canonical({"shape": [8, 2304], "dtype": "torch.bfloat16"}).encode()
    )
    zero.update(bytes(8 * 2304 * 2))
    zero_sha = zero.hexdigest()
    for r in audit:
        case = cases[r["key"]]
        task = tasks_for(case, groups[r["topic"]], fixture, r["world"])[
            r["label_order"]
        ]
        assert r["gold"] == task["gold"] and r["question_sha256"] == digest(
            task["question"]
        )
        assert r["source_sha256"] == sources[r["key"]]["source_sha256"]
        expected_donor = (
            r["source_sha256"]
            if r["route"] == "literal"
            else sources[r["key"]]["state_sha256"]
            if r["route"] == "raw_E"
            else zero_sha
        )
        assert r["injected_sha256"] == expected_donor
        answer = (
            case["pole"] if all(fixture["worlds"][r["world"]]) else 1 - case["pole"]
        )
        assert r["gold"] == ("A" if answer == r["label_order"] else "B")
    assert len({r["injected_sha256"] for r in audit if r["route"] == "zero"}) == 1
