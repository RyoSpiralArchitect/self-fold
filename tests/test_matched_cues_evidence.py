"""Check each public matched-cue comparison against its input and its native output."""

import json
from pathlib import Path
import subprocess
import sys

from selffold.matched_cues import HEADERS, preference_records, question_groups
from selffold.polarity_readout import predicted_class
from selffold.records import digest, file_sha

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/matched_cues_20261103"
RUN = STUDY / "run"
FIXTURE = ROOT / "data/matched_cues_20261103.json"
ACTIONS = ROOT / "data/within_topic_wording_20261103.json"


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def test_matched_cue_public_archive_and_registered_reduction():
    prefix = STUDY.relative_to(ROOT).as_posix()
    files = (
        subprocess.check_output(["git", "ls-files", "-z", "--", prefix], cwd=ROOT)
        .decode()
        .split("\0")
    )
    names = {s.removeprefix(prefix + "/") for s in files if s}
    lines = (STUDY / "SHA256SUMS").read_text().splitlines()
    expected = {n: h for h, n in [s.split("  ", 1) for s in lines]}
    assert len(expected) == len(lines) and set(expected) == names - {"SHA256SUMS"}
    assert all(file_sha(STUDY / n) == h for n, h in expected.items())
    reduced = json.loads(
        subprocess.check_output([sys.executable, "-S", str(STUDY / "summarize.py")])
    )
    assert reduced == read(STUDY / "summary.json")
    descriptive = json.loads(
        subprocess.check_output(
            [sys.executable, "-S", str(STUDY / "exchange_counts.py")]
        )
    )
    assert descriptive == read(STUDY / "exchange_counts.json")
    request = read(RUN / "request.json")
    assert request["fixture_sha256"] == file_sha(FIXTURE)
    assert request["plan_sha256"] == file_sha(STUDY / "PLAN.md") and request[
        "reducer_sha256"
    ] == file_sha(STUDY / "summarize.py")
    prior = ROOT / "provenance/teacher_polarity_20261103/run"
    for field, file in (
        ("axes", "axes.npz"),
        ("renderer", "renderer.npz"),
        ("prior_sources", "sources.jsonl"),
        ("prior_teacher_scores", "teachers.jsonl"),
    ):
        assert request[field + "_sha256"] == file_sha(prior / file)
    end = read(RUN / "terminal.json")
    assert (
        end["status"] == "COMPLETED_MATCHED_CUES"
        and end["updates"] == request["updates"] == 0
    )
    assert end["frozen_parameter_sha256_before"] == end["frozen_parameter_sha256_after"]
    assert not end["G_trained"] and not end["qualification_content_loaded"]
    seal = read(RUN / "source.json")
    assert (
        not seal["git_status"]
        and seal["git_commit"] == "95db5846d7b5f78a0a36dd9b1e850051b3907e81"
    )
    assert read(RUN / "replay.json") == {
        "canonical_teachers": 16,
        "native_scores_exact": 128,
        "source_hashes_and_decisions_exact": 256,
    }


def test_native_cue_pairs_bind_same_teacher_and_gold_to_all_four_questions():
    groups = question_groups(read(ACTIONS)["groups"], read(FIXTURE))
    pairs = {r["pair_sha256"]: r for r in rows(RUN / "native_pairs.jsonl")}
    layout = {
        (r["topic"], r["stem"], r["label_order"]): r
        for r in rows(RUN / "question_layout.jsonl")
    }
    assert len(layout) == 64 and all(
        len(set(r["header_token_ids"])) == 3
        and r["named_tokens"] == r["bare_tokens"] + 2
        for r in layout.values()
    )
    for r in pairs.values():
        assert r["pair_sha256"] == digest([r["injected_sha256"], r["question_sha256"]])
    conditioned = {}
    records = rows(RUN / "teachers.jsonl")
    assert len(records) == 1024
    for r in records:
        t, s, p, o = r["topic"], r["stem"], r["pole"], r["label_order"]
        i = HEADERS.index(r["header"])
        g = groups[t]
        opts = g["options"][::-1] if o else g["options"]
        q = (
            g["fresh_questions"][4 * s + i]
            + f"\nA = {opts[0]}. B = {opts[1]}. Use the supplied description."
        )
        pair = pairs[r["pair_sha256"]]
        assert (
            r["question_sha256"]
            == digest(q)
            == pair["question_sha256"][o]
            == layout[t, s, o]["question_sha256"][i]
        )
        assert r["injected_sha256"] == pair["injected_sha256"]
        assert all(r[k] == v for k, v in pair["outputs"][o].items()) and r["gold"] == (
            "A" if p == o else "B"
        )
        conditioned.setdefault((r["route"], t, p), set()).add(r["injected_sha256"])
    assert all(len(v) == 1 for v in conditioned.values())
    assert (
        len(
            {
                next(iter(v))
                for (route, t, p), v in conditioned.items()
                if route == "mean"
            }
        )
        == 1
    )


def test_source_and_literal_controls_share_all_512_sources_and_paired_geometry():
    expected = {
        c["key"]: c for c in preference_records(read(ACTIONS)["groups"], read(FIXTURE))
    }
    raw = rows(RUN / "sources.jsonl")
    literal = rows(RUN / "literal.jsonl")
    bare_question_hash = {
        (r["topic"], r["label_order"]): r["question_sha256"][0]
        for r in rows(RUN / "question_layout.jsonl")
        if r["stem"] == 0
    }
    assert all(
        r["question_sha256"] == bare_question_hash[r["topic"], r["label_order"]]
        for r in literal
    )
    hashed = {k: digest(c["source"]) for k, c in expected.items()}
    assert {r["key"]: r["source_sha256"] for r in raw} == hashed and len(raw) == 512
    for r in raw:
        assert all(
            r[m] == predicted_class(r["scores"], m)
            for m in ("factor", "direct", "mention")
        )
        assert r["oracle_factor"] == predicted_class(
            r["scores"], oracle_topic=r["topic"]
        )
    assert len(literal) == len({(r["key"], r["label_order"]) for r in literal}) == 1024
    assert all(
        r["source_sha256"] == hashed[r["key"]]
        and r["gold"] == ("A" if r["pole"] == r["label_order"] else "B")
        for r in literal
    )
    layouts = rows(RUN / "source_layout.jsonl")
    assert len(layouts) == 256 and {
        r[p] for r in layouts for p in ("before", "after")
    } == set(expected)
    for r in layouts:
        assert (
            r["before_sha256"] == hashed[r["before"]]
            and r["after_sha256"] == hashed[r["after"]]
        )
        assert len(r["slots"]) == 8 and r["slots"][-1] < r["tokens"]
