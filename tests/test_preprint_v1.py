import gzip
import json
from pathlib import Path
import subprocess

import numpy as np

from selffold.records import file_sha
from selffold.three_role_capacity import (
    ASSIGNMENTS,
    PAIRS,
    evaluate,
    exact_code,
    native_rewards,
    partitions,
    triplet,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/preprint_v1_20261104"


def read(path):
    return json.loads(path.read_text())


def test_capacity_joint_success_and_supplied_ceiling_on_all_three_action_queries():
    labels = [101, 103]
    # A correct target beats either rival. A row absent from a pair emits a third token.
    winners = np.full((1, 3, 3, 2), 999)
    for action in range(3):
        for i, pair in enumerate(PAIRS):
            if action in pair:
                winners[0, action, i] = [
                    labels[pair.index(action)],
                    labels[1 - pair.index(action)],
                ]
    rewards = native_rewards(winners, labels)
    assert evaluate(np.arange(6), ASSIGNMENTS, rewards)["native_histories"] == [6, 6]
    assert evaluate(np.arange(6) % 4, ASSIGNMENTS[:4], rewards)["native_histories"] == [
        4,
        6,
    ]
    assert len(list(partitions(6, 8))) == 203
    for w in partitions(6, 4):
        assert len(set(w)) <= 4
    assert all(len(set(triplet(t))) == 3 for t in range(8))


def test_frozen_writers_match_native_rewards_and_all_complete_histories():
    p = STUDY / "capacity"
    manifest = read(p / "fit_closed.json")
    published = read(p / "results.json")
    assert manifest["test_tables_compiled"] == 0 and manifest["updates"] == 512
    assert file_sha(p / "parameters.npz") == manifest["parameters"]
    for split, ntables, topics in (("fit", 4, [0, 2, 4, 6]), ("test", 8, [1, 3, 5, 7])):
        inputs = read(p / f"{split}_inputs.json")
        assert inputs["topics"] == topics
        with np.load(p / f"{split}_native.npz", allow_pickle=False) as f:
            assert f["top"].shape == (ntables, 3, 3, 2)
            assert f["tables"].shape == (ntables, 3, 16)
            assert np.allclose(f["tables"].sum(-1), 1.0)
            assert np.isfinite(f["ab"]).all()
            r = native_rewards(f["top"], inputs["labels"])
        for tag, (w, d) in manifest["models"].items():
            words, decoder = np.array(w), np.array(d)
            assert evaluate(words, decoder, r) == published[tag][split]
            if split == "test":
                metrics = evaluate(words, decoder, r, erase=True)
                assert metrics == published[tag]["test_erased"]
                assert metrics["native_decisions"] == [288, 576]
            if tag.startswith("gradient"):
                with np.load(p / "parameters.npz", allow_pickle=False) as fits:
                    np.testing.assert_array_equal(
                        fits[tag + "_writer"].argmax(-1), words
                    )
                    np.testing.assert_array_equal(
                        fits[tag + "_reader"].argmax(-1), decoder
                    )
            if tag.startswith("exact") and split == "fit":
                ref = exact_code(r, int(tag.split("_")[1][0]))
                np.testing.assert_array_equal(ref[0], words)
                np.testing.assert_array_equal(ref[1], decoder)
    assert [
        published[f"gradient_3bit_seed{s}"]["test"]["native_histories"][0]
        for s in (17, 41, 73)
    ] == [48, 48, 40]
    assert read(p / "terminal.json")["learned_pass"] is False
    assert read(p / "terminal.json")["supplied_pass"] is True


def test_all_raw_readouts_recompute_with_blind_opposed_questions():
    p = STUDY / "raw"
    cap = list(map(json.loads, (p / "captures.jsonl").read_text().splitlines()))
    assert sum(r["split"] == "test" for r in cap) == 160
    endpoint = read(p / "fit_closed.json")
    assert (
        endpoint["labels"] == 3072
        and endpoint["updates"] == 192
        and endpoint["fresh_captures"] == 0
    )
    assert file_sha(p / "reader.npz") == endpoint["weights"]
    with gzip.open(p / "decisions.jsonl.gz", "rt") as stream:
        rows = {
            (r["arm"], r["key"], r["world"], r["order"], r["route"]): r
            for r in map(json.loads, stream)
        }
    assert len(rows) == 10240
    summary = read(p / "summary.json")
    for arm in ("parent", "trained"):
        for stage in ("source", "cases", "rule"):
            blocks, counts = {}, dict(matched=0, zero=0, literal=0, opposite_reversal=0)
            for key, r in rows.items():
                if key[0] != arm or key[-1] != "matched" or r["stage"] != stage:
                    continue
                other = rows[(*key[:-1], "opposed")]
                assert (other["question"], other["gold"]) == (r["question"], r["gold"])
                ok = r["greedy_label"] == r["gold"]
                rev = other["greedy_label"] == ("B" if r["gold"] == "A" else "A")
                counts["opposite_reversal"] += int(
                    ok and rev and (r["A"] - r["B"]) * (other["A"] - other["B"]) < 0
                )
                for route in ("matched", "zero", "literal"):
                    counts[route] += (
                        rows[(*key[:-1], route)]["greedy_label"] == r["gold"]
                    )
                blocks.setdefault((r["topic"], r["pair"]), []).append(ok)
            s = summary[arm]["stages"][stage]
            assert all(s[k] == v for k, v in counts.items())
            assert (len(blocks), {len(x) for x in blocks.values()}) == (
                16,
                {16 if stage == "rule" else 32},
            )
            assert s["strict"] == [sum(all(v) for v in blocks.values()), 16] == [0, 16]
            assert not s["passed"]
    assert [
        summary["trained"]["stages"][k]["matched"] for k in ("source", "cases", "rule")
    ] == [254, 263, 123]


def test_versioned_release_evidence_manifest_and_replay():
    raw, cap = read(STUDY / "raw/terminal.json"), read(STUDY / "capacity/terminal.json")
    replay = read(STUDY / "replay/receipt.json")
    assert raw["backbone"] == cap["backbone"]
    for v in (raw, cap):
        assert (
            v["compute"]["forwards"] < 25000
            and v["compute"]["processed_tokens"] < 20000000
        )
        assert v["elapsed_seconds"] < 2400
    assert (
        replay["raw_decisions"],
        replay["capacity_decisions"],
        replay["raw_state_hashes"],
    ) == (10240, 216, 160)
    assert replay["raw_scores"] == file_sha(STUDY / "raw/decisions.jsonl.gz")
    assert replay["cpu_code_comparison"]["native_histories"] == [32, 48]
    files = {
        str((ROOT / n.decode()).relative_to(STUDY))
        for n in subprocess.check_output(
            ["git", "ls-files", "-z", "--", str(STUDY)], cwd=ROOT
        ).split(b"\0")
        if n
    }
    checks = {
        name: sha
        for sha, name in (
            line.split("  ", 1)
            for line in (STUDY / "SHA256SUMS").read_text().splitlines()
        )
    }
    assert files == set(checks) | {"SHA256SUMS"}
    for name, sha in checks.items():
        assert file_sha(STUDY / name) == sha
