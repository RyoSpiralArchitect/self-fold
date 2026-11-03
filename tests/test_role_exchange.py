import json
from pathlib import Path

import numpy as np
import pytest

from selffold.role_exchange import (
    _solve,
    crossed_records,
    role_pairs,
    null_orientations,
    fit_exchange,
    score_one,
    load_weights,
)

ROOT = Path(__file__).resolve().parents[1]


def test_exchange_design_preserves_mention_order_and_holds_out_all_action_phrases():
    groups = json.loads((ROOT / "data/within_topic_wording_20261103.json").read_text())[
        "groups"
    ]
    form = json.loads((ROOT / "data/matched_cues_20261103.json").read_text())
    fresh = json.loads((ROOT / "data/role_exchange_20261103.json").read_text())[
        "fresh_actions"
    ]
    old_strings = {w for g in groups for p in ("plus", "minus") for w in g[p]}
    unseen = {w for g in fresh.values() for p in ("plus", "minus") for w in g[p]}
    assert len(unseen) == 64 and unseen.isdisjoint(old_strings)
    for bank, n in ((None, 6), (fresh, 4)):
        cases = list(crossed_records(groups, form, bank))
        assert len(set(c["source"] for c in cases)) == len(cases) == 8 * n * n * 16
        keys, idx = role_pairs(cases)
        assert len(keys) == len(cases) // 2
        for k, (a, b) in zip(keys, idx, strict=True):
            l, r = cases[a], cases[b]
            assert l["order"] == 0 and r["order"] == 1 and r["pole"] == 1 - l["pole"]
            assert l["pole"] == k[-1] and (l["plus_index"], l["minus_index"]) == divmod(
                k[3], n
            )
            g = groups[k[0]] if bank is None else bank[groups[k[0]]["id"]]
            pos, neg = g["plus"][l["plus_index"]], g["minus"][l["minus_index"]]
            assert all(
                (c["source"].index(pos) < c["source"].index(neg)) == (k[-1] == 0)
                for c in (l, r)
            )
        if bank is None:
            signs = null_orientations(keys)
            for t in range(8):
                for f in form["verdicts"]:
                    for p in ("before", "after"):
                        for a in (0, 1):
                            assert (
                                sum(
                                    s
                                    for k, s in zip(keys, signs, strict=True)
                                    if (k[0], k[1], k[2], k[4]) == (t, f, p, a)
                                )
                                == 0
                            )


def test_unit_weight_equals_singleton_ridge_and_four_changes_the_objective():
    rng = np.random.default_rng(47)
    x = rng.normal(size=(32, 7))
    y = rng.normal(size=(32, 2))
    a, b = x[::2], x[1::2]
    ya, yb = y[::2], y[1::2]
    fit = _solve((a + b) / 2, (a - b) / 2, (ya + yb) / 2, (ya - yb) / 2)
    z = x - x.mean(0)
    lam = 0.01 * (z * z).sum(1).mean()
    expected = np.linalg.solve(z.T @ z + lam * np.eye(7), z.T @ (y - y.mean(0)))
    assert np.allclose(fit["ordinary"][0], expected, atol=1e-9)
    assert np.allclose(fit["ordinary"][1], y.mean(0) - x.mean(0) @ expected)
    assert not np.allclose(fit["weighted"][0], expected)


def test_single_state_reader_preserves_inputs_restores_and_needs_no_mate(tmp_path):
    rows = []
    xs = []
    for topic in range(2):
        for pair in range(4):
            for pole in (0, 1):
                for order in (0, 1):
                    rows.append(
                        dict(
                            topic=topic,
                            family="a",
                            position="before",
                            pair=pair,
                            pole=pole,
                            order=order,
                        )
                    )
                    xs.append(
                        [
                            [
                                1 - 2 * topic,
                                1 - 2 * order,
                                (1 - 2 * (pole ^ order)) * (1 - 2 * topic),
                                pair / 5,
                            ]
                        ]
                    )
    x = np.array(xs, dtype=np.float64)
    orig = x.copy()
    router = dict(
        mean=np.zeros(4, dtype="f"),
        scale=np.ones(4, dtype="f"),
        shared=np.zeros((4, 3), dtype="f"),
        offset=np.zeros(3, dtype="f"),
    )
    router["shared"][0, :] = [1, -1, 0]
    keys, ind = role_pairs(rows)
    maps, _ = fit_exchange(x, keys, ind, router)
    assert np.array_equal(x, orig)
    for name, fit in maps.items():
        path = tmp_path / (name + ".npz")
        np.savez_compressed(path, **fit)
        restored = load_weights(path)
        with pytest.raises(ValueError, match="single"):
            score_one(x[:2], fit)
        for z, c in zip(x, rows, strict=True):
            r = score_one(z, fit)
            assert r == score_one(z, restored) and r["topic"] == c["topic"]
            assert (r["role"] * r["actions"][r["topic"]] < 0) == bool(c["pole"])
