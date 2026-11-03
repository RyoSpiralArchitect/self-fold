"""Grade locked native responses. The consumer process has no access to this answer join."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from selffold.records import digest, file_sha
from selffold.state_only import (
    OPERATIONS,
    ROUTES,
    branch_for,
    selected_action,
    source_bank,
)

ROOT = Path(__file__).resolve().parents[2]


def read(p):
    return json.loads(p.read_text())


def accuracy(mask):
    return [int(mask.sum()), int(mask.size)]


def fraction(pair):
    return pair[0] / pair[1]


def panel_score(correct, flips, changed, operations, topics, groups):
    # All arrays have axes source, route, operation, world, answer-label mapping.
    pos = ROUTES.index
    hit = correct[:, pos("matched")]
    joint = hit & correct[:, pos("opposed")] & flips
    axes = np.array(
        [branch_for(op, w) for op, w in product(OPERATIONS, range(4))], dtype=bool
    ).reshape(1, 3, 4, 1)
    oracle = correct[:, pos("oracle")]
    same = hit & correct[:, pos("equivalent")]
    partial = {r: hit & correct[:, pos(r)] for r in ("joint_row", "default_rows")}
    per_op = {}
    for i, op in enumerate(operations):
        per_op[op] = {r: accuracy(correct[:, pos(r), i]) for r in ROUTES}
        per_op[op]["original_and_reversed_margin"] = accuracy(joint[:, i])
        per_op[op]["original_and_equivalent"] = accuracy(same[:, i])
    group_pass = []
    for unit in sorted(set(groups)):
        ix = np.array([v == unit for v in groups])
        if sum(ix) != 4:
            raise ValueError("Invalid strict unit")
        group_pass.append(bool(hit[ix].all()))
    result = dict(
        native=accuracy(hit),
        oracle=accuracy(oracle),
        opposed_pair=accuracy(joint),
        equivalent_pair=accuracy(same),
        uniform=accuracy(correct[:, pos("uniform")]),
        strict=accuracy(np.array(group_pass)),
        by_operation=per_op,
        by_topic={str(t): accuracy(hit[topics == t]) for t in sorted(set(topics))},
    )
    for r in partial:
        edited = np.broadcast_to(axes if r == "joint_row" else ~axes, hit.shape)
        result[r] = {
            n: accuracy(partial[r][m])
            for n, m in (("affected", edited), ("unaffected", ~edited))
        }
    changed = np.broadcast_to(changed, hit.shape)
    result["changed_worlds"] = dict(
        updated=accuracy(hit[changed]),
        omitted=accuracy(correct[:, pos("omit")][changed]),
    )
    result["unchanged_worlds"] = dict(
        updated=accuracy(hit[~changed]),
        omitted=accuracy(correct[:, pos("omit")][~changed]),
    )
    checks = {
        "native_each_operation": min(fraction(p["matched"]) for p in per_op.values())
        >= 0.9,
        "opposed_paired_each_operation": min(
            fraction(p["original_and_reversed_margin"]) for p in per_op.values()
        )
        >= 0.9,
        "equivalent_each_operation": min(
            fraction(p["original_and_equivalent"]) for p in per_op.values()
        )
        >= 0.9,
        "strict_units": fraction(result["strict"]) >= 0.75,
        "partial_donors": min(
            fraction(pair) for r in partial for pair in result[r].values()
        )
        >= 0.9,
        "over_uniform_each_operation": min(
            fraction(p["matched"]) - fraction(p["uniform"]) for p in per_op.values()
        )
        >= 0.25,
        "benefit_on_changed_worlds": fraction(result["changed_worlds"]["updated"])
        - fraction(result["changed_worlds"]["omitted"])
        >= 0.4,
    }
    return result | dict(criteria=checks, pass_all=all(checks.values()))


def summarize(directory):
    producer, consumer = directory / "produce", directory / "decide"
    meta = read(producer / "cases.json")
    if (
        read(producer / "terminal.json")["status"] != "STATE_PACKET_CLOSED"
        or read(consumer / "terminal.json")["status"] != "BLIND_DECISION_COMPLETE"
    ):
        raise ValueError("Incomplete execution")
    binding = read(consumer / "binding.json")
    if binding["answers_sha256"] != file_sha(consumer / "answers.npz") or binding[
        "packet_sha256"
    ] != file_sha(producer / "packet/binding.json"):
        raise ValueError("Raw receipts differ")
    design = read(ROOT / "data/state_only_20261104.json")
    groups = read(ROOT / "data/within_topic_wording_20261103.json")["groups"]
    keys = [
        (panel, k, digest(text))
        for panel in ("primary", "challenge")
        for k, text in source_bank(design, groups, panel)
    ]
    if keys != [
        (
            c["split"],
            tuple(c[x] for x in ("topic", "flag", "style", "pole", "order")),
            c["source_sha256"],
        )
        for c in meta
    ]:
        raise ValueError("Missing, extra or changed source")
    labels = np.asarray(binding["labels"])
    with np.load(producer / "grading_order.npz", allow_pickle=False) as f:
        restore = f["restore"]
    shape = (len(meta), len(ROUTES), len(OPERATIONS), 4, 2)
    if len(restore) != np.prod(shape) or not np.array_equal(
        np.sort(restore), np.arange(len(restore))
    ):
        raise ValueError("Invalid blinded denominator")
    with np.load(consumer / "answers.npz", allow_pickle=False) as f:
        dispatch = f["dispatch"][restore].reshape(shape)
        native = f["top_token"][dispatch]
        margin = np.subtract(*f["ab_logits"].T)[dispatch]
    with np.load(producer / "packet/inputs.npz", allow_pickle=False) as f:
        req = f["requests"][restore].reshape(*shape, 6)
        expected = np.indices(shape)
        cells = np.array([c["topic"] for c in meta])[:, None, None, None, None]
        if any(
            not np.all(req[..., col] == value)
            for col, value in (
                (1, expected[2]),
                (2, expected[3]),
                (3, cells),
                (4, expected[4]),
                (5, expected[1] == ROUTES.index("omit")),
            )
        ):
            raise ValueError("Blind request alignment failed")
        mid = f["states"][req[:, ROUTES.index("matched"), 0, 0, 0, 0]]
    correct = np.zeros(shape, dtype=bool)
    for i, c in enumerate(meta):
        for j, r in enumerate(ROUTES):
            for o, op in enumerate(OPERATIONS):
                for w in range(4):
                    bit = branch_for(op, w)
                    value = c["pole"] ^ bit
                    if (
                        r == "opposed"
                        or (r == "joint_row" and bit)
                        or (r == "default_rows" and not bit)
                    ):
                        value ^= 1
                    correct[i, j, o, w] = (
                        native[i, j, o, w] == labels[[value, 1 - value]]
                    )
    flips = margin[:, ROUTES.index("matched")] * margin[:, ROUTES.index("opposed")] < 0
    changed = np.array(
        [
            [branch_for(op, w) != branch_for(OPERATIONS[0], w) for w in range(4)]
            for op in OPERATIONS
        ]
    )[None, :, :, None]
    result = {}
    for panel in ("primary", "challenge"):
        ix = np.array([c["split"] == panel for c in meta])
        cm = [c for c, yes in zip(meta, ix) if yes]
        units = [(c["topic"], c["flag"], c["style"]) for c in cm]
        score = panel_score(
            correct[ix],
            flips[ix],
            changed,
            OPERATIONS,
            np.array([c["topic"] for c in cm]),
            units,
        )
        truth = np.array(
            [
                [
                    selected_action(c["topic"], c["pole"], OPERATIONS[0], w)
                    for w in range(4)
                ]
                for c in cm
            ]
        )
        score["middle_all_four_rows"] = accuracy((mid[ix].argmax(-1) == truth).all(1))
        score["cases"] = len(cm)
        result[panel] = score
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    print(json.dumps(summarize(parser.parse_args().run), indent=2))
