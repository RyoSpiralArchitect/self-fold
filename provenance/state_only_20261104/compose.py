"""Apply the fixed update once to complete states, then independently grade a second blind call."""

import argparse
import json
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import torch

from selffold.records import RunStore, file_sha
from selffold.state_only import OPERATIONS, ROUTES, PolicyUpdate, branch_for

PARENT = {
    "produce/cases.json": "8d81156efbbf804ef726d55111b42ed79a51a9aac126036b731f85f2e8afb36a",
    "produce/grading_order.npz": "194e89d7c8491b9d071165a35cc3098653be97d35d79b10d259fd01243b94710",
    "produce/packet/binding.json": "74c292dc2099175abbc6c2ea83171d4f6b87285fdfd2487ae482d67599efc734",
}
CHAIN_ROUTES = ("matched", "opposed", "omit_second")


def read(path):
    return json.loads(path.read_text())


def prepare(run, output):
    if any(file_sha(run / path) != sha for path, sha in PARENT.items()):
        raise ValueError("Unexpected parent")
    parent = run / "produce/packet"
    binding = read(parent / "binding.json")
    if any(file_sha(parent / k) != v for k, v in binding["files"].items()):
        raise ValueError("Parent packet differs")
    cases = read(run / "produce/cases.json")
    primary = [(i, c) for i, c in enumerate(cases) if c["split"] == "primary"]
    if len(primary) != 128:
        raise ValueError("Incomplete primary panel")
    store = RunStore(output)
    packet = RunStore(output / "packet")
    for name in ("menus.json", "reader.npz", "update.npz"):
        shutil.copyfile(parent / name, packet.path / name)
    update = PolicyUpdate().requires_grad_(False)
    with np.load(packet.path / "update.npz", allow_pickle=False) as f:
        update.load_state_dict({"routing": torch.from_numpy(f["routing"])})
    with np.load(run / "produce/grading_order.npz", allow_pickle=False) as f:
        restore = f["restore"]
    with np.load(parent / "inputs.npz", allow_pickle=False) as f:
        requests = f["requests"][restore].reshape(len(cases), len(ROUTES), 3, 4, 2, 6)
        original = torch.from_numpy(f["states"])
    mids, calls, addresses = [], [], {}
    for i, c in primary:
        for first, route in product(range(3), CHAIN_ROUTES):
            incoming = "opposed" if route == "opposed" else "matched"
            source = requests[i, ROUTES.index(incoming), 0, 0, 0, 0]
            key = int(source), first
            if key not in addresses:
                addresses[key] = len(mids)
                mids.append(update(original[source], first).numpy())
            for second, w, order in product(range(3), range(4), range(2)):
                calls.append(
                    (
                        addresses[key],
                        second,
                        w,
                        c["topic"],
                        order,
                        int(route == "omit_second"),
                    )
                )
    perm = np.random.default_rng(632).permutation(len(calls))
    np.savez_compressed(
        packet.path / "inputs.npz",
        states=np.stack(mids),
        requests=np.array(calls, dtype=np.int32)[perm],
    )
    np.savez_compressed(output / "grading_order.npz", restore=np.argsort(perm))
    packet.write(
        "binding.json",
        binding
        | dict(files={p.name: file_sha(p) for p in sorted(packet.path.iterdir())}),
    )
    store.write(
        "preparation.json",
        dict(
            parent=PARENT,
            cases=len(primary),
            paths=len(calls),
            first_tables=len(mids),
            plan=file_sha(Path(__file__).with_name("COMPOSITION_PLAN.md")),
            code=file_sha(Path(__file__)),
            training_updates=0,
        ),
    )


def count(a):
    return [int(a.sum()), int(a.size)]


def grade(run, output):
    meta = [c for c in read(run / "produce/cases.json") if c["split"] == "primary"]
    pred = output / "decide"
    binding = read(pred / "binding.json")
    if (
        read(pred / "terminal.json")["status"] != "BLIND_DECISION_COMPLETE"
        or binding["answers_sha256"] != file_sha(pred / "answers.npz")
        or binding["packet_sha256"] != file_sha(output / "packet/binding.json")
    ):
        raise ValueError("Consumer receipts do not match")
    shape = (len(meta), 3, 3, 3, 4, 2)
    with np.load(output / "grading_order.npz", allow_pickle=False) as f:
        restore = f["restore"]
    if not np.array_equal(np.sort(restore), np.arange(np.prod(shape))):
        raise ValueError("Missing chained decisions")
    with np.load(pred / "answers.npz", allow_pickle=False) as f:
        dispatch = f["dispatch"][restore].reshape(shape)
        token = f["top_token"][dispatch]
        margin = np.subtract(*f["ab_logits"].T)[dispatch]
    labels = np.array(binding["labels"])
    poles = np.array([c["pole"] for c in meta])[:, None, None]
    units = [(c["topic"], c["flag"], c["style"]) for c in meta]
    report = {}
    for a, first in enumerate(OPERATIONS):
        for b, second in enumerate(OPERATIONS):
            truth = np.array([branch_for(second, w) for w in range(4)])[None, :, None]
            gold = labels[poles ^ truth ^ np.arange(2)]
            hits = token[:, a, 0, b] == gold
            paired = (
                hits
                & (token[:, a, 1, b] == gold[:, :, ::-1])
                & (margin[:, a, 0, b] * margin[:, a, 1, b] < 0)
            )
            bypass = token[:, a, 2, b] == gold
            strict = [
                hits[[u == key for u in units]].all() for key in sorted(set(units))
            ]
            changed = [branch_for(first, w) != branch_for(second, w) for w in range(4)]
            pass_changed = not any(changed) or (
                hits[:, changed].mean() - bypass[:, changed].mean() >= 0.4
            )
            report[first + "/" + second] = dict(
                native=count(hits),
                paired=count(paired),
                strict=count(np.array(strict)),
                bypass=count(bypass),
                updated_changed=count(hits[:, changed]),
                bypass_changed=count(bypass[:, changed]),
                pass_all=bool(
                    hits.mean() >= 0.9
                    and paired.mean() >= 0.9
                    and np.mean(strict) >= 0.75
                    and pass_changed
                ),
            )
    return dict(
        cells=report,
        native=[sum(v["native"][0] for v in report.values()), len(meta) * 72],
        pass_all=all(v["pass_all"] for v in report.values()),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "grade"])
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.run, args.output)
    else:
        print(json.dumps(grade(args.run, args.output), indent=2))
