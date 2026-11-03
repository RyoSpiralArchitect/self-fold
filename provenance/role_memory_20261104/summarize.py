"""Join frozen learned trajectories to native scores and count the registered development endpoints."""

import argparse
import json
from pathlib import Path

import numpy as np

from selffold.records import file_sha
from selffold.role_memory import edit_paths
from selffold.role_trials import ROUTES, donor_indices

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "data/role_memory_20261104.json").read_text())


def read(path):
    return json.loads(path.read_text())


def count(a):
    return [int(a.sum()), int(a.size)]


def rate(pair):
    return pair[0] / pair[1]


def bits(mask):
    return (np.asarray(mask)[..., None] >> np.arange(4)) & 1


def select_scores(matrix, rows, case_ids):
    if rows.dtype != np.uint8 or np.any(rows > 1):
        raise ValueError("Invalid stored row address")
    indices = np.asarray(case_ids).reshape(-1, *([1] * (rows.ndim - 1)))
    return matrix[indices, rows]


def expected(keys, masks, labels):
    polarity = np.array([k[3] for k in keys]).reshape(
        (-1,) + (1,) * (np.ndim(masks) + 2)
    )
    return labels[polarity ^ bits(masks)[..., None] ^ np.arange(2)]


def depth(rows, keys, top, margin, labels, paths, level):
    previous, last = (0, 1) if level == 2 else (2, 3)
    lookup = donor_indices(keys)
    truth = expected(keys, paths[:, last], labels)
    wanted = truth[..., ::-1]
    by_route = {}
    for r, name in enumerate(ROUTES):
        donor = lookup[{"opposed": 1, "equivalent": 2}.get(name, 0)]
        by_route[name] = select_scores(top, rows[:, r], donor)
    hit = by_route["matched"] == truth
    raw_opp = by_route["opposed"]
    signed = (
        select_scores(margin, rows[:, 0], lookup[0])
        * select_scores(margin, rows[:, 1], lookup[1])
        < 0
    )
    both = hit & (raw_opp == wanted) & signed
    same = hit & (by_route["equivalent"] == truth)
    flipped = hit & (by_route["flip"] == wanted)
    changed = np.broadcast_to(
        (bits(paths[:, previous]) != bits(paths[:, last]))[None, :, :, None], hit.shape
    )
    omit = by_route["omit"] == truth
    units = [(k[0], k[1]) for k in keys]
    strict = []
    for unit in sorted(set(units)):
        ix = np.array([v == unit for v in units])
        if sum(ix) != 4:
            raise ValueError("Bad strict denominator")
        strict.append(hit[ix].all())
    cells = {
        str(i): dict(native=count(hit[:, i]), opposed=count(both[:, i]))
        for i in range(len(paths))
    }
    topics = np.array([k[0] for k in keys])
    report = dict(
        native=count(hit),
        opposed=count(both),
        equivalent=count(same),
        flipped_bit=count(flipped),
        erased=count(by_route["erase"] == truth),
        strict=count(np.asarray(strict)),
        cells=cells,
        changed=count(hit[changed]),
        bypass_changed=count(omit[changed]),
        preserved=count(hit[~changed]),
        bypass_preserved=count(omit[~changed]),
        per_topic={str(t): count(hit[topics == t]) for t in sorted(set(topics))},
        updater_unseen_topics=count(hit[~np.isin(topics, CFG["fit_topics"])]),
    )
    report["criteria"] = {
        "all_cell_native": all(rate(c["native"]) >= 0.9 for c in cells.values()),
        "all_cell_opposed": all(rate(c["opposed"]) >= 0.9 for c in cells.values()),
        "equivalent": rate(report["equivalent"]) >= 0.9,
        "strict": rate(report["strict"]) >= 0.75,
        "changed_vs_bypass": rate(report["changed"]) - rate(report["bypass_changed"])
        >= 0.5,
        "preservation": rate(report["preserved"]) >= 0.9,
    }
    return report


def collision(rows, keys, top, margin, labels, masks):
    if rows.shape != (len(keys), len(masks) // 2, len(masks), 2, 2, 4):
        raise ValueError("Invalid collision denominator")
    truths = expected(keys, np.tile(masks, (len(masks) // 2, 1)), labels)
    case = np.arange(len(keys))
    report = {}
    for j, name in enumerate(("intact", "erased")):
        left, right = [select_scores(top, rows[:, :, :, h, j], case) for h in range(2)]
        lmar, rmar = [
            select_scores(margin, rows[:, :, :, h, j], case) for h in range(2)
        ]
        a, b = left == truths, right == truths[..., ::-1]
        report[name] = dict(
            native=count(np.stack((a, b))),
            paired=count(a & b & (lmar * rmar < 0)),
            identical_row_addresses=count(rows[:, :, :, 0, j] == rows[:, :, :, 1, j]),
        )
    if not np.array_equal(rows[:, :, :, 0, 1], rows[:, :, :, 1, 1]):
        raise ValueError("Erased twins no longer receive identical rows")
    return report


def summarize(run):
    prod, dec = run / "produce", run / "decide"
    if (
        read(prod / "terminal.json")["status"] != "ROLE_CONTROLLERS_CLOSED"
        or read(dec / "terminal.json")["status"] != "NATIVE_PACKET_SCORED"
    ):
        raise ValueError("Incomplete run")
    if read(prod / "rollout_closed.json")["paths"] != file_sha(
        prod / "trajectories.npz"
    ) or read(dec / "native.json")["scores_sha256"] != file_sha(dec / "scores.npz"):
        raise ValueError("Result binding mismatch")
    if read(dec / "terminal.json")["packet"] != file_sha(prod / "packet/binding.json"):
        raise ValueError("Consumer packet mismatch")
    keys = read(prod / "test_keys.json")["keys"]
    expected_keys = [
        [g, f, s, p, o]
        for g in range(8)
        for f in range(2)
        for s in CFG["test_styles"]
        for p in (0, 1)
        for o in (0, 1)
    ]
    if keys != expected_keys:
        raise ValueError("Source key coverage mismatch")
    with np.load(prod / "join.npz", allow_pickle=False) as f:
        restore = f["restore"]
        if not np.array_equal(
            np.sort(restore), np.arange(len(keys))
        ) or not np.array_equal(f["donors"], donor_indices(keys)):
            raise ValueError("Invalid native join")
    with np.load(dec / "scores.npz", allow_pickle=False) as f:
        dispatch = f["dispatch"][restore]
        top = f["top"][dispatch]
        margin = np.subtract(*f["ab"].T)[dispatch]
    labels = np.array(read(dec / "native.json")["labels"])
    masks = np.array(CFG["test_masks"])
    paths = np.asarray(edit_paths(masks))
    gold = expected(keys, masks, labels)
    scores = {}
    with np.load(prod / "trajectories.npz", allow_pickle=False) as f:
        for seed in CFG["seeds"]:
            for arm in CFG["arms"]:
                key = f"{arm}_{seed}"
                first = f[key + "_first"]
                if first.shape != (len(keys), len(masks), 4):
                    raise ValueError("First-edit denominator")
                final = {}
                for name, level in (("second", 2), ("fourth", 4)):
                    rows = f[key + "_" + name]
                    if rows.shape != (len(keys), len(ROUTES), len(paths), 4):
                        raise ValueError("Incomplete paths")
                    final[name] = depth(rows, keys, top, margin, labels, paths, level)
                twin = collision(f[key + "_twins"], keys, top, margin, labels, masks)
                scores[key] = dict(
                    first=count(
                        select_scores(top, first, np.arange(len(keys))) == gold
                    ),
                    twins=twin,
                    **final,
                )
    for seed in CFG["seeds"]:
        for arm in CFG["arms"]:
            s = scores[f"{arm}_{seed}"]
            s["qualified_development"] = bool(
                all(
                    all(s[d]["criteria"].values())
                    and rate(s[d]["native"])
                    - rate(scores[f"table_only_{seed}"][d]["native"])
                    >= 0.25
                    for d in ("second", "fourth")
                )
                and rate(s["twins"]["intact"]["paired"]) >= 0.9
                and s["twins"]["erased"]["paired"][0] == 0
                and rate(s["twins"]["erased"]["native"]) <= 0.5
            )
    deltas = [
        rate(scores[f"reward_{seed}"]["fourth"]["native"])
        - rate(scores[f"supervised_{seed}"]["fourth"]["native"])
        for seed in CFG["seeds"]
    ]
    return dict(
        scores=scores,
        paths=paths.tolist(),
        arm_pass={
            a: all(scores[f"{a}_{s}"]["qualified_development"] for s in CFG["seeds"])
            for a in CFG["arms"]
        },
        reward_minus_supervised_points=[round(d * 100, 6) for d in deltas],
        reward_advantage=bool(
            min(deltas) >= 0
            and np.mean(deltas) >= 0.05
            and sum(d > 0 for d in deltas) >= 2
            and all(
                scores[f"reward_{s}"]["fourth"]["strict"][0]
                >= scores[f"supervised_{s}"]["fourth"]["strict"][0]
                for s in CFG["seeds"]
            )
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    print(json.dumps(summarize(parser.parse_args().run), indent=2))
