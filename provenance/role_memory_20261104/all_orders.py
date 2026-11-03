"""Frozen exhaustive sequence audit, with a separate native-cache replay requirement."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import torch

from selffold.records import RunStore, file_sha
from selffold.role_memory import RoleMemory, edit_paths, final_edit, native_row_indices
from selffold.role_trials import donor_indices
from summarize import CFG, count, expected, rate, read, select_scores

BOUND = {
    "produce/fits.npz": "438e385dcc6121ab69aa1d47f1f205fc5f6d1baa996e7476741ea09d197042a6",
    "produce/packet/rows.npz": "c29b23e186875aae743ef8a25aa04c06026016b02e78eb78735642d8b54eaaaa",
    "produce/trajectories.npz": "b8bb66e6900eff0c6299b4beae0ea9a1fc314f3c2b9ad9333a7b8c0d049d718e",
    "produce/join.npz": "f7193be8452a3678d817df5530858d1cc14b5c80cfe436af65152eb56e57caeb",
    "produce/test_keys.json": "35d409fd7300cd002932ecedb805c864dbe31ee21acad5ca7203e8ded492f192",
    "decide/scores.npz": "6f435e2cf573130cc5f01cf2382e62aa824d294f906bc1cbe3a0369e40768871",
}


@torch.no_grad()
def simulate(actor, rows, paths, arm):
    parts = []
    for chunk in torch.tensor(paths).split(36):
        n = len(chunk)
        ref = rows.repeat_interleave(n, 0)
        table = ref[:, [0, 0, 0, 1]]
        bit = torch.zeros(len(table), dtype=torch.long)
        steps = []
        for mask in chunk.repeat(len(rows), 1).T:
            table, bit = final_edit(actor, table, bit, mask, arm == "table_only")
            steps.append(native_row_indices(table, ref).reshape(len(rows), n, 4))
        parts.append(torch.stack(steps, -2))
    return torch.cat(parts, 1).numpy()


def compare(rows, keys, paths, top, margin, labels, old):
    n = len(keys)
    ids = np.arange(n)
    opp = donor_indices(keys)[1]
    parity = np.bitwise_xor.reduce(paths[:, :3] & 1, axis=1)
    if np.bincount(parity).tolist() != [648, 648] or parity[old].any():
        raise ValueError("Parity strata differ from the sealed universe")
    previous = []
    for k in range(4):
        target = expected(keys, paths[:, k], labels)
        predictions = select_scores(top, rows[:, :, k], ids)
        previous.append(predictions == target)
    hit = previous[-1]
    all_steps = np.logical_and.reduce(previous)
    opposing = select_scores(top, rows[opp, :, 3], opp)
    both = (
        hit
        & (opposing == target[..., ::-1])
        & (
            select_scores(margin, rows[:, :, 3], ids)
            * select_scores(margin, rows[opp, :, 3], opp)
            < 0
        )
    )
    grouped = [(k[0], k[1]) for k in keys]
    result = dict(
        final=count(hit),
        paired=count(both),
        all_steps=count(all_steps),
        original36_all_steps=count(all_steps[:, old]),
        partial_accuracy=[count(v) for v in previous],
    )
    for tag, stratum in enumerate(("even", "odd")):
        chosen = parity == tag
        strict = [
            bool(hit[np.array([u == key for u in grouped])][:, chosen].all())
            for key in sorted(set(grouped))
        ]
        result[stratum] = dict(
            final=count(hit[:, chosen]),
            paired=count(both[:, chosen]),
            strict=count(np.asarray(strict)),
        )
    return result


def run(source, replay, out):
    if any(file_sha(source / k) != v for k, v in BOUND.items()):
        raise ValueError("Original fixed run differs")
    expected_files = ("native.json", "calls.json", "scores.npz")
    replayed = {k: file_sha(replay / k) for k in expected_files}
    if (
        any(v != file_sha(source / "decide" / k) for k, v in replayed.items())
        or read(replay / "terminal.json")["packet"]
        != read(source / "decide/terminal.json")["packet"]
    ):
        raise ValueError("Fresh Gemma replay failed")
    store = RunStore(out)
    store.write(
        "binding.json",
        dict(
            parent=BOUND,
            replayed=replayed,
            replay=read(replay / "terminal.json"),
            plan=file_sha(Path(__file__).with_name("ALL_ORDERS_PLAN.md")),
            scorer=file_sha(Path(__file__)),
            training_updates=0,
        ),
    )
    keys = read(source / "produce/test_keys.json")["keys"]
    with np.load(source / "produce/join.npz", allow_pickle=False) as f:
        order = f["restore"]
    with np.load(source / "produce/packet/rows.npz", allow_pickle=False) as f:
        rows = torch.from_numpy(f["rows"][order])
    with np.load(replay / "scores.npz", allow_pickle=False) as f:
        dispatch = f["dispatch"][order]
        top = f["top"][dispatch]
        margin = np.subtract(*f["ab"].T)[dispatch]
    labels = np.array(read(replay / "native.json")["labels"])
    paths = np.array(list(product(CFG["test_masks"], repeat=4)))
    lookup = {tuple(path): i for i, path in enumerate(paths)}
    old = np.array([lookup[tuple(p)] for p in edit_paths(CFG["test_masks"])])
    opp = donor_indices(keys)[1]
    outputs = {}
    summary = {}
    with (
        np.load(source / "produce/fits.npz", allow_pickle=False) as fit,
        np.load(source / "produce/trajectories.npz", allow_pickle=False) as prior,
    ):
        for seed in CFG["seeds"]:
            for arm in CFG["arms"]:
                key = f"{arm}_{seed}"
                actor = RoleMemory(seed).requires_grad_(False)
                actor.load_state_dict(
                    {
                        k: torch.from_numpy(fit[key + "_" + k])
                        for k in actor.state_dict()
                    }
                )
                z = simulate(actor, rows, paths, arm)
                if not np.array_equal(
                    z[:, old, 3], prior[key + "_fourth"][:, 0]
                ) or not np.array_equal(
                    z[opp][:, old, 3], prior[key + "_fourth"][:, 1]
                ):
                    raise ValueError(
                        "Previous explicit native trajectories do not replay"
                    )
                outputs[key] = z
                summary[key] = compare(z, keys, paths, top, margin, labels, old)
    passed = {}
    for arm in CFG["arms"]:
        checks = []
        for s in CFG["seeds"]:
            item = summary[f"{arm}_{s}"]
            checks.append(
                all(
                    rate(item[p][k]) >= 0.9
                    for p in ("even", "odd")
                    for k in ("final", "paired")
                )
                and rate(item["all_steps"]) >= 0.9
                and rate(item["final"]) - rate(summary[f"table_only_{s}"]["final"])
                >= 0.25
            )
        passed[arm] = all(checks)
    np.savez_compressed(out / "addresses.npz", **outputs)
    store.write(
        "summary.json",
        dict(
            scores=summary,
            pass_all_seeds=passed,
            paths=len(paths),
            per_parity=648,
            addresses_sha256=file_sha(out / "addresses.npz"),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("source", "replay", "out"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    run(args.source, args.replay, args.out)
