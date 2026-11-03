"""Verify released policy learning, blind native joins, exact collisions and parity strata."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from selffold.records import file_sha
from selffold.role_memory import RoleMemory, edit_paths
from selffold.role_trials import rollout, twins

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/role_memory_20261104"
RUN = STUDY / "run"
CFG = json.loads((ROOT / "data/role_memory_20261104.json").read_text())


def read(p):
    return json.loads(p.read_text())


def test_sources_schedules_and_complete_primary_reduction():
    names = {
        str((ROOT / str(x.decode())).relative_to(STUDY))
        for x in subprocess.check_output(
            ["git", "ls-files", "-z", "--", str(STUDY.relative_to(ROOT))], cwd=ROOT
        ).split(b"\0")
        if x
    }
    files = {
        name: sha
        for sha, name in (
            line.split("  ", 1)
            for line in (STUDY / "SHA256SUMS").read_text().splitlines()
        )
    }
    assert set(files) == names - {"SHA256SUMS"} and all(
        file_sha(STUDY / k) == v for k, v in files.items()
    )
    closed = read(RUN / "produce/fit_closed.json")
    assert closed["fresh_captures"] == 0 and closed["weights"] == file_sha(
        RUN / "produce/fits.npz"
    )
    for seed in CFG["seeds"]:
        matches = [r for r in closed["fits"] if r["seed"] == seed]
        assert {r["arm"] for r in matches} == set(CFG["arms"])
        assert (
            len({r["schedule"] for r in matches})
            == len({r["start"] for r in matches})
            == 1
        )
    fitkeys = read(RUN / "produce/fit_keys.json")
    testkeys = read(RUN / "produce/test_keys.json")
    assert len(fitkeys["keys"]) == 144 and len(testkeys["keys"]) == 64
    assert not set(fitkeys["source_sha256"]) & set(testkeys["source_sha256"])
    assert {k[0] for k in fitkeys["keys"]} == set(CFG["fit_topics"])
    assert (
        read(RUN / "produce/source.json")["git_commit"]
        == "00104f747316d77f91e4aa143eed0231272e63b9"
    )
    assert (
        read(RUN / "produce/terminal.json")["backbone"]
        == read(RUN / "decide/terminal.json")["backbone"]
        == "2690ac5991d97f7caf539b457d3de686949626739480b7d228bcdad8609b9e92"
    )
    assert json.loads(
        subprocess.check_output([sys.executable, str(STUDY / "summarize.py"), str(RUN)])
    ) == read(STUDY / "summary.json")
    summary = read(STUDY / "summary.json")
    assert not any(summary["arm_pass"].values()) and not summary["reward_advantage"]


def test_saved_model_reconstructs_primary_trajectories_and_the_learned_writer_code():
    prod = RUN / "produce"
    with np.load(prod / "join.npz", allow_pickle=False) as f:
        order = f["restore"]
    with np.load(prod / "packet/rows.npz", allow_pickle=False) as f:
        rows = torch.from_numpy(f["rows"][order])
    keys = read(prod / "test_keys.json")["keys"]
    # All nine controller states are tiny enough to replay on CPU.
    masks = torch.arange(1, 15)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with (
            np.load(prod / "fits.npz", allow_pickle=False) as fits,
            np.load(prod / "trajectories.npz", allow_pickle=False) as tapes,
        ):
            for seed in CFG["seeds"]:
                for arm in CFG["arms"]:
                    actor = RoleMemory(seed).requires_grad_(False)
                    prefix = f"{arm}_{seed}_"
                    actor.load_state_dict(
                        {
                            k: torch.from_numpy(fits[prefix + k])
                            for k in actor.state_dict()
                        }
                    )
                    if arm != "table_only":
                        assert torch.equal((actor.write(masks) > 0).long(), masks % 2)
                    computed = rollout(actor, arm, rows, keys, CFG["test_masks"])
                    computed["twins"] = twins(actor, arm, rows, CFG["test_masks"])
                    assert all(
                        np.array_equal(tapes[prefix + k], v)
                        for k, v in computed.items()
                    )
    finally:
        torch.set_num_threads(threads)


def test_exhaustive_orders_and_native_truth_from_raw_endpoints():
    from itertools import product

    prod = RUN / "produce"
    full = STUDY / "full"
    packet = prod / "packet"
    native = RUN / "decide"
    b = read(full / "binding.json")
    assert b["training_updates"] == 0 and all(
        file_sha(RUN / k) == v for k, v in b["parent"].items()
    )
    assert b["plan"] == file_sha(STUDY / "ALL_ORDERS_PLAN.md") and b[
        "scorer"
    ] == file_sha(STUDY / "all_orders.py")
    assert all(file_sha(native / k) == v for k, v in b["replayed"].items()) and b[
        "replay"
    ]["packet"] == file_sha(packet / "binding.json")
    report = read(full / "summary.json")
    assert report["addresses_sha256"] == file_sha(full / "addresses.npz")
    paths = np.asarray(list(product(CFG["test_masks"], repeat=4)))
    parity = (paths[:, :3] % 2).sum(-1) % 2
    schedule = np.array(edit_paths(CFG["test_masks"]))
    assert (
        len(paths) == 1296
        and parity.sum() == 648
        and not ((schedule[:, :3] % 2).sum(-1) % 2).any()
    )
    keys = read(prod / "test_keys.json")["keys"]
    with np.load(prod / "join.npz", allow_pickle=False) as f:
        order = f["restore"]
    with np.load(native / "scores.npz", allow_pickle=False) as f:
        top = f["top"][f["dispatch"][order]]
    labels = np.array(read(native / "native.json")["labels"])
    pole = np.array([k[3] for k in keys])[:, None, None, None]
    with np.load(full / "addresses.npz", allow_pickle=False) as f:
        for seed in CFG["seeds"]:
            for arm in CFG["arms"]:
                name = f"{arm}_{seed}"
                rows = f[name]
                results = report["scores"][name]
                joint = True
                for t in range(4):
                    gold = labels[
                        pole
                        ^ ((paths[:, t, None] >> np.arange(4)) % 2)[..., None]
                        ^ np.arange(2)
                    ]
                    hit = (
                        top[np.arange(len(keys))[:, None, None], rows[:, :, t]] == gold
                    )
                    joint = joint & hit
                assert [int(joint.sum()), joint.size] == results["all_steps"]
                assert [int(hit.sum()), hit.size] == results["final"]
                for value, n in [(0, "even"), (1, "odd")]:
                    selected = hit[:, parity == value]
                    assert [int(selected.sum()), selected.size] == results[n]["final"]
                assert results["all_steps"] == (
                    [82944, 663552] if arm == "table_only" else [663552, 663552]
                )
