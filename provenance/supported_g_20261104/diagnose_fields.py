"""Post-result localization with fixed numerical rows, native outputs and one-bit weights."""

import argparse
import json
from pathlib import Path

import numpy as np

from summarize import reference


def diagnose(study, route):
    orig = study / "structured"
    folder = study / route
    root = Path(__file__).resolve().parents[2]
    actor_path = root / "provenance/role_memory_20261104/run/produce/fits.npz"
    with np.load(actor_path, allow_pickle=False) as f:
        pointer = f["reward_17_address"]
        writer = f["reward_17_message"]
    with np.load(orig / "heads.npz", allow_pickle=False) as f:
        gates = f["gates"] > 0
    with np.load(folder / "native.npz", allow_pickle=False) as f:
        native = {
            (int(t), r.tobytes()): (a, p)
            for t, r, a, p, n in zip(
                f["topics"], f["rows"], f["top"], f["ab"], f["missing"], strict=True
            )
            if not n
        }
    with np.load(folder / "states.npz", allow_pickle=False) as f:
        keys, table, base, guard, bit, updated = (
            f[k] for k in ("keys", "table", "field_base", "guard", "bit", "updated")
        )
    labels = np.array(json.loads((folder / "native.json").read_text())["labels"])
    gold, _, _ = reference(keys)
    masks = np.array([3, 12, 5, 10])
    worlds = np.arange(4)
    bit_rows = (masks[base, None] >> worlds) & 1
    branch_rows = np.stack(
        [
            table[np.arange(len(table)), np.argmax(bit_rows == b, axis=1)]
            for b in (0, 1)
        ],
        axis=1,
    )
    result = {}
    for tag, fix_base, fix_guard in (
        ("observed", False, False),
        ("true_base", True, False),
        ("true_veto", False, True),
        ("both", True, True),
    ):
        b = keys[:, 1] if fix_base else base
        g = keys[:, 2] + 1 if fix_guard else guard
        pred_mask = (masks[b, None] >> worlds) & 1
        incoming = branch_rows[np.arange(len(keys))[:, None], pred_mask]
        message = (
            ((2.0 * pred_mask - 1) * writer[:4]).sum(axis=1) + writer[4] > 0
        ).astype(int)
        diff = np.abs(incoming - incoming[:, :1]).sum(-1)
        relative = diff / np.maximum(diff.max(-1, keepdims=True), 1e-12)
        first = (pointer[message, 0, None] * relative).argmax(1)
        fixed = np.where(
            gates[g, :, None], incoming[np.arange(len(keys)), first, None, :], incoming
        )
        if tag == "observed" and (
            not np.array_equal(message, bit) or not np.array_equal(fixed, updated)
        ):
            raise ValueError("The saved learned G is not reconstructed")
        scores = []
        for i, k in enumerate(keys):
            hits = []
            for w in worlds:
                top, ab = native[int(k[0]), fixed[i, w].tobytes()]
                hits.extend(
                    ((top == labels[gold[i, w]]) & (ab[:, 0] != ab[:, 1])).tolist()
                )
            scores.append(hits)
        scores = np.array(scores)
        result[tag] = dict(
            correct=int(scores.sum()),
            n=int(scores.size),
            complete=int(scores.all(1).sum()),
            sources=len(keys),
        )
    result["scope"] = "post_result_formal_field_substitution_no_fit"
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("study", type=Path)
    p.add_argument("route", choices=("structured", "readout"))
    a = p.parse_args()
    print(json.dumps(diagnose(a.study, a.route), sort_keys=True, separators=(",", ":")))
