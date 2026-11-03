"""Restore the two branch roles using only an incoming table, and grade the native replay."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from compose import grade, read
from selffold.records import RunStore, file_sha

CHAIN_PACKET = "fd28be504eff40d5df4e72ef09d0ceffdb13bef2ee286277028b96d10c020a30"
CHAIN_ORDER = "5ca529462d75bfd356a9cf8c52c6c3f6283f81ad745fcbf4dabdd2621a3abc8b"
MODES = ("restore_roles", "exchange_roles")


def restore_roles(tables, exchange):
    """No keys, operation labels, displayed options, questions or supervision enter here."""
    default = tables[:, 0]
    distance = np.square(
        tables[:, 1:].astype(np.float64) - default[:, None].astype(np.float64)
    ).sum(-1)
    counterpart = tables[np.arange(len(tables)), 1 + distance.argmax(1)]
    if exchange:
        default, counterpart = counterpart, default
    return np.stack((default, default, default, counterpart), axis=1)


def prepare(chain, output):
    parent = chain / "packet"
    if (
        file_sha(parent / "binding.json") != CHAIN_PACKET
        or file_sha(chain / "grading_order.npz") != CHAIN_ORDER
    ):
        raise ValueError("Unexpected parent composition run")
    binding = read(parent / "binding.json")
    if any(file_sha(parent / k) != v for k, v in binding["files"].items()):
        raise ValueError("Changed packet contents")
    study = RunStore(output)
    with np.load(parent / "inputs.npz", allow_pickle=False) as f:
        old, req = f["states"], f["requests"].copy()
    # The bypass always observes the original table before either intervention.
    req[req[:, 5] == 1, 0] += len(old)
    for mode in MODES:
        dest = RunStore(output / mode)
        packet = RunStore(dest.path / "packet")
        for name in ("update.npz", "reader.npz", "menus.json"):
            shutil.copyfile(parent / name, packet.path / name)
        shutil.copyfile(chain / "grading_order.npz", dest.path / "grading_order.npz")
        corrected = restore_roles(old, mode == "exchange_roles")
        np.savez_compressed(
            packet.path / "inputs.npz",
            states=np.concatenate((corrected, old)),
            requests=req,
        )
        packet.write(
            "binding.json",
            binding
            | dict(files={p.name: file_sha(p) for p in sorted(packet.path.iterdir())}),
        )
    study.write(
        "preparation.json",
        dict(
            parent_packet=CHAIN_PACKET,
            parent_order=CHAIN_ORDER,
            training_updates=0,
            plan=file_sha(Path(__file__).with_name("ROLE_RECOVERY_PLAN.md")),
            producer=file_sha(Path(__file__)),
            logical_per_mode=len(req),
        ),
    )


def score(run, chain, output):
    original = grade(run, chain)
    modes = {m: grade(run, output / m) for m in MODES}
    restored, exchanged = (modes[m]["cells"] for m in MODES)
    atypical = [k for k in restored if not k.startswith("both_on/")]

    def rate(d, k):
        a, n = d[k]["native"]
        return a / n

    advantage = sum(
        rate(restored, k) - rate(original["cells"], k) for k in atypical
    ) / len(atypical)
    ordinary = [rate(restored, k) for k in restored if k not in atypical]
    checks = dict(
        all_previous_cells=modes[MODES[0]]["pass_all"],
        atypical_gain=advantage >= 0.4,
        ordinary_retained=min(ordinary) >= 0.99,
        wrong_binding_low=max(rate(exchanged, k) for k in exchanged) <= 0.1,
    )
    return dict(
        raw=original["native"],
        modes=modes,
        atypical_gain_points=advantage * 100,
        checks=checks,
        pass_all=all(checks.values()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["prepare", "grade"])
    p.add_argument("run", type=Path)
    p.add_argument("chain", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if args.phase == "prepare":
        prepare(args.chain, args.output)
    else:
        print(json.dumps(score(args.run, args.chain, args.output), indent=2))
