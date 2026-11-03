"""Bind both readouts, the untouched fresh split, paired controls and the public reduction."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from selffold.records import digest, file_sha
from selffold.role_exchange import crossed_records, role_pairs, null_orientations

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/role_exchange_20261103"
RUN = STUDY / "run"
FRESH = ROOT / "data/role_exchange_20261103.json"
ACTIONS = ROOT / "data/within_topic_wording_20261103.json"
CARDS = ROOT / "data/matched_cues_20261103.json"
PREV = ROOT / "provenance/matched_cues_20261103/run"


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(l) for l in p.read_text().splitlines()]


def test_role_exchange_public_reduction_and_sealed_fit_boundary():
    names = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--", str(STUDY.relative_to(ROOT))], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    names = {
        Path(n).relative_to(STUDY.relative_to(ROOT)).as_posix() for n in names if n
    }
    lines = (STUDY / "SHA256SUMS").read_text().splitlines()
    manifest = {n: h for h, n in (l.split("  ", 1) for l in lines)}
    assert (
        len(lines) == len(manifest)
        and set(manifest) == names - {"SHA256SUMS"}
        and all(file_sha(STUDY / n) == h for n, h in manifest.items())
    )
    assert json.loads(
        subprocess.check_output([sys.executable, "-S", str(STUDY / "summarize.py")])
    ) == read(STUDY / "summary.json")
    detail = json.loads(
        subprocess.check_output(
            [sys.executable, "-S", str(STUDY / "error_partition.py")]
        )
    )
    assert detail == read(STUDY / "error_partition.json")
    req = read(RUN / "request.json")
    src = read(RUN / "source.json")
    end = read(RUN / "terminal.json")
    assert (
        src["git_commit"] == "d93b140fb743de8ffbf814ba7af1be4e4bd12b3c"
        and not src["git_status"]
    )
    assert req["plan_sha256"] == file_sha(STUDY / "PLAN.md") and req[
        "reducer_sha256"
    ] == file_sha(STUDY / "summarize.py")
    assert [
        req[n]
        for n in (
            "fitting_fixture_sha256",
            "fresh_fixture_sha256",
            "card_fixture_sha256",
        )
    ] == [file_sha(p) for p in (ACTIONS, FRESH, CARDS)]
    assert (
        end["status"] == "COMPLETED_ROLE_EXCHANGE_READOUT"
        and end["backbone_before"] == end["backbone_after"]
    )
    assert (
        not end["G_trained"]
        and not end["qualification_content_loaded"]
        and req["optimizer_steps"] == end["optimizer_steps"] == 0
    )
    assert read(RUN / "fit_closed.json")["fresh_sources_encoded"] == 0
    assert read(RUN / "replay.json")["prior_reader_decisions_exact"] == 512


def test_role_exchange_pairs_reconstruct_the_training_null_and_all_new_records():
    groups = read(ACTIONS)["groups"]
    fmt = read(CARDS)
    new = read(FRESH)["fresh_actions"]
    scores = rows(RUN / "scores.jsonl")
    paired = rows(RUN / "pairs.jsonl")
    seen = set()
    for split, vocab, total in (("fit", None, 4608), ("fresh", new, 2048)):
        cases = list(crossed_records(groups, fmt, vocab))
        hashed = {c["key"]: digest(c["source"]) for c in cases}
        rr = [r for r in scores if r["split"] == split]
        assert (
            len(rr) == len(hashed) == total
            and {r["key"]: r["source_sha256"] for r in rr} == hashed
        )
        assert seen.isdisjoint(hashed.values())
        seen.update(hashed.values())
        keys, indices = role_pairs(cases)
        ps = [r for r in paired if r["split"] == split]
        assert len(keys) == len(ps) == total // 2
        flips = null_orientations(keys) if split == "fit" else None
        for i, (k, (a, b), r) in enumerate(zip(keys, indices, ps, strict=True)):
            assert (
                r["topic"],
                r["family"],
                r["position"],
                r["pair"],
                r["first_action"],
            ) == k
            assert (
                r["accept_first"] == cases[a]["key"]
                and r["reject_first"] == cases[b]["key"]
            )
            assert len(r["slots"]) == 8 and r["slots"][-1] < r["num_tokens"]
            if flips is None:
                assert "shuffled_orientation" not in r
            else:
                assert int(flips[i]) == r["shuffled_orientation"]
    fresh = {r["key"]: r for r in scores if r["split"] == "fresh"}
    literal = rows(RUN / "literal.jsonl")
    assert len(literal) == len({(r["key"], r["label_order"]) for r in literal}) == 4096
    for r in literal:
        c = fresh[r["key"]]
        opts = groups[c["topic"]]["options"]
        if r["label_order"]:
            opts = opts[::-1]
        q = (
            fmt["questions"][0]
            + f"\nA = {opts[0]}. B = {opts[1]}. Use the supplied description."
        )
        assert (
            r["question_sha256"] == digest(q)
            and r["source_sha256"] == c["source_sha256"]
        )
        assert r["gold"] == ("A" if c["pole"] == r["label_order"] else "B")


def test_role_exchange_weights_keep_no_training_examples_and_bind_to_frozen_router():
    fits = rows(RUN / "fits.jsonl")
    closes = read(RUN / "fit_closed.json")
    assert {r["arm"]: r["contrast_weight"] for r in fits} == {
        "weighted": 4,
        "ordinary": 1,
    }
    rr = [r for r in rows(RUN / "scores.jsonl") if r["split"] == "fit"]
    captures = digest([(r["source_sha256"], r["state_sha256"]) for r in rr])
    old = {r["source_sha256"]: r for r in rows(PREV / "sources.jsonl")}
    earlier = [r for r in rr if r["source_sha256"] in old]
    assert len(earlier) == 512 and all(
        (r["state_sha256"], r["old"])
        == (old[r["source_sha256"]]["state_sha256"], old[r["source_sha256"]]["factor"])
        for r in earlier
    )
    previous = ROOT / "provenance/teacher_polarity_20261103/run/axes.npz"
    assert read(RUN / "request.json")["router_sha256"] == file_sha(previous)
    for f in fits:
        name = f["arm"]
        path = RUN / (name + ".npz")
        assert f["sha256"] == closes["artifact_sha256"][name] == file_sha(path)
        assert (
            f["fit_singleton_replays"] == 4608 and f["fit_capture_sha256"] == captures
        )
        with (
            np.load(path, allow_pickle=False) as a,
            np.load(previous, allow_pickle=False) as old_axes,
        ):
            assert set(a.files) == {
                "geometry",
                "mean",
                "scale",
                "role_axes",
                "role_bias",
                "local_axes",
                "local_bias",
                "router_mean",
                "router_scale",
                "router_shared",
                "router_offset",
            }
            assert a["geometry"].tolist() == [8, 2304] and a["local_axes"].shape == (
                8,
                18432,
                2,
            )
            assert a["role_axes"].shape == (18432, 2 if name == "weighted" else 1)
            assert all(
                np.array_equal(a["router_" + k], old_axes[k])
                for k in ("mean", "scale", "shared", "offset")
            )
            assert all(np.isfinite(a[k]).all() for k in a.files)
