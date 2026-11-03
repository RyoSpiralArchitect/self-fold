"""Reconstruct the scientific counts, blind joins and donor replacements from exact public bytes."""

from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess

import numpy as np

from selffold.records import file_sha
from selffold.state_only import ROUTES

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/state_only_20261104"
RUN = STUDY / "run"


def read(path):
    return json.loads(path.read_text())


def module(name):
    spec = importlib.util.spec_from_file_location(name, STUDY / (name + ".py"))
    item = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(item)
    return item


def test_study_bytes_and_blind_gates_recompute():
    tracked = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--", str(STUDY.relative_to(ROOT))], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    names = {str((ROOT / p).relative_to(STUDY)) for p in tracked if p}
    hashes = {
        name: sha
        for sha, name in (
            row.split("  ", 1)
            for row in (STUDY / "SHA256SUMS").read_text().splitlines()
        )
    }
    assert set(hashes) == names - {"SHA256SUMS"}
    assert all(file_sha(STUDY / k) == v for k, v in hashes.items())
    assert module("summarize").summarize(RUN) == read(STUDY / "summary.json")
    assert module("compose").grade(RUN, STUDY / "chain") == read(
        STUDY / "chain-summary.json"
    )
    fit = read(RUN / "produce/fit_closed.json")
    assert (
        fit["fresh_captures"] == 0
        and fit["native_logits_replayed"]
        and fit["lexical_one_hot_parity"]
    )
    assert (
        read(RUN / "produce/source.json")["git_commit"]
        == "effaaef64af873f8e32342049470558bfeb8c0a9"
    )
    for path in [RUN / "produce", RUN / "decide", STUDY / "chain/decide"]:
        assert (
            read(path / "terminal.json")["backbone_sha256"]
            == "2690ac5991d97f7caf539b457d3de686949626739480b7d228bcdad8609b9e92"
        )
    for p in [RUN / "produce/packet", STUDY / "chain/packet"]:
        binding = read(p / "binding.json")
        assert set(binding["files"]) == {
            "inputs.npz",
            "update.npz",
            "reader.npz",
            "menus.json",
        }
        assert all(file_sha(p / k) == v for k, v in binding["files"].items())


def test_source_free_exchanges_have_identical_questions_and_the_intended_state():
    meta = read(RUN / "produce/cases.json")
    lookup = {
        (c["split"], c["topic"], c["flag"], c["style"], c["pole"], c["order"]): i
        for i, c in enumerate(meta)
    }
    with np.load(RUN / "produce/grading_order.npz", allow_pickle=False) as f:
        restore = f["restore"]
    with np.load(RUN / "produce/packet/inputs.npz", allow_pickle=False) as f:
        req = f["requests"][restore].reshape(192, len(ROUTES), 3, 4, 2, 6)
        states = f["states"]
    assert (req[:, :, :, :, :, 1:5] == req[:, :1, :, :, :, 1:5]).all()
    ids = req[:, :, 0, 0, 0, 0]
    assert (req[:, :, :, :, :, 0] == ids[:, :, None, None, None]).all()

    def get(i, route):
        return states[ids[i, ROUTES.index(route)]]

    challenge = []
    for key, i in lookup.items():
        s, t, f, y, p, o = key
        z = get(i, "matched")
        opp = get(lookup[(s, t, f, y, 1 - p, o)], "matched")
        eq = get(lookup[(s, t, 1 - f, y, p, 1 - o)], "matched")
        assert hashlib.sha256(z.tobytes()).hexdigest() == meta[i]["mid_sha256"]
        for route, expected in [
            ("opposed", opp),
            ("equivalent", eq),
            ("joint_row", np.vstack((z[:3], opp[3:]))),
            ("default_rows", np.vstack((opp[:3], z[3:]))),
            ("omit", z),
            ("uniform", np.full_like(z, 1 / 16)),
        ]:
            assert np.array_equal(get(i, route), expected)
        if s == "challenge":
            a, b = z[[0, 3]].argmax(axis=1)
            challenge.append(
                (
                    a == 2 * t + p and b == 2 * t + 1 - p,
                    a == b,
                    {a, b} <= {2 * t, 2 * t + 1},
                )
            )
    assert sum(c for c, _, _ in challenge) == 32
    assert sum(same and inside for _, same, inside in challenge) == 20
    assert sum(not inside for _, _, inside in challenge) == 12


def test_chain_consumes_actual_first_outputs_and_directional_reversals():
    run = RUN / "produce"
    chain = STUDY / "chain"
    with np.load(run / "grading_order.npz", allow_pickle=False) as f:
        first_order = f["restore"]
    with np.load(run / "packet/inputs.npz", allow_pickle=False) as f:
        original = f["states"]
        first_requests = f["requests"][first_order].reshape(192, 8, 3, 4, 2, 6)
    with np.load(chain / "grading_order.npz", allow_pickle=False) as f:
        order = f["restore"]
    with np.load(chain / "packet/inputs.npz", allow_pickle=False) as f:
        actual = f["states"]
        requests = f["requests"][order].reshape(128, 3, 3, 3, 4, 2, 6)
    with np.load(run / "packet/update.npz", allow_pickle=False) as f:
        weights = np.exp(f["routing"] - f["routing"].max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
    assert np.all(weights[..., 0] == weights[..., 1]) and np.all(
        weights[..., 0] == weights[..., 2]
    )
    for op in range(3):
        for route, parent in enumerate(("matched", "opposed", "matched")):
            expected = (
                weights[op]
                @ original[first_requests[:128, ROUTES.index(parent), 0, 0, 0, 0]]
            )
            ids = requests[:, op, route, 0, 0, 0, 0]
            assert np.all(
                requests[:, op, route, :, :, :, 0] == ids[:, None, None, None]
            )
            assert np.all(requests[:, op, route, :, :, :, 5] == (route == 2))
            np.testing.assert_allclose(actual[ids], expected, rtol=1e-5, atol=1e-7)
    with np.load(chain / "decide/answers.npz", allow_pickle=False) as f:
        native = f["top_token"][f["dispatch"][order]].reshape(128, 3, 3, 3, 4, 2)
        assert set(np.unique(native)) == set(
            read(chain / "decide/binding.json")["labels"]
        )
        assert np.all(f["ab_logits"][:, 0] != f["ab_logits"][:, 1])
    assert np.all(native[:, 2, 0] != native[:, 2, 1])
    cells = read(STUDY / "chain-summary.json")["cells"]
    assert all(
        cells["exactly_one_on/" + op]["paired"] == [0, 1024]
        for op in ("both_on", "either_on", "exactly_one_on")
    )


def test_fixed_role_recovery_preserves_soft_values_and_blind_omissions():
    import sys

    chain = STUDY / "chain"
    recovery = STUDY / "recovery"
    assert read(recovery / "preparation.json")["training_updates"] == 0
    with np.load(chain / "packet/inputs.npz", allow_pickle=False) as f:
        old, req = f["states"], f["requests"]
    # Independent calculation using per-row vector norms. No action IDs enter this join.
    partner = np.array(
        [
            1 + np.argmax([np.linalg.norm(row.astype(float) - z[0]) for row in z[1:]])
            for z in old
        ]
    )
    other = old[np.arange(len(old)), partner]
    n = len(old)
    for mode, default, switched in (
        ("restore_roles", old[:, 0], other),
        ("exchange_roles", other, old[:, 0]),
    ):
        p = recovery / mode / "packet"
        for name in ("reader.npz", "update.npz", "menus.json"):
            assert file_sha(p / name) == file_sha(chain / "packet" / name)
        with np.load(p / "inputs.npz", allow_pickle=False) as f:
            states, addr = f["states"], f["requests"]
        assert np.array_equal(states[n:], old)
        assert np.array_equal(
            states[:n], np.stack((default, default, default, switched), axis=1)
        )
        assert np.array_equal(addr[:, 1:], req[:, 1:])
        assert np.array_equal(addr[:, 0], req[:, 0] + n * req[:, 5])
    score = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(STUDY / "role_recovery.py"),
                "grade",
                str(RUN),
                str(chain),
                str(recovery),
            ]
        )
    )
    assert score == read(STUDY / "recovery-summary.json") and score["pass_all"]
    for mode, total in (("restore_roles", 9216), ("exchange_roles", 0)):
        assert score["modes"][mode]["native"] == [total, 9216]
