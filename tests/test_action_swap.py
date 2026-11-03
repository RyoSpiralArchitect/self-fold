import json
from pathlib import Path

import numpy as np
import pytest

from selffold.action_swap import (
    _fit_directions,
    action_pairs,
    fit_action_heads,
    score_action,
)
from selffold.role_exchange import crossed_records

ROOT = Path(__file__).resolve().parents[1]


def test_action_swaps_hold_verdict_sequence_fixed_and_reverse_action_order():
    groups = json.loads((ROOT / "data/within_topic_wording_20261103.json").read_text())[
        "groups"
    ]
    fixture = json.loads((ROOT / "data/matched_cues_20261103.json").read_text())
    cases = list(crossed_records(groups, fixture))
    keys, indices = action_pairs(cases)
    assert len(keys) == len(indices) == 2304
    assert np.array_equal(np.sort(indices.ravel()), np.arange(len(cases)))
    for key, (a, b) in zip(keys, indices, strict=True):
        first, second = cases[a], cases[b]
        assert first["order"] == second["order"] == key[-1]
        assert first["pole"] == first["order"]
        assert second["pole"] != second["order"]
        group = groups[first["topic"]]
        plus = group["plus"][first["plus_index"]]
        minus = group["minus"][first["minus_index"]]
        assert first["source"].index(plus) < first["source"].index(minus)
        assert second["source"].index(plus) > second["source"].index(minus)
        normalized = [
            case["source"].replace(plus, "ACTION").replace(minus, "ACTION")
            for case in (first, second)
        ]
        assert normalized[0] == normalized[1]
    with pytest.raises(ValueError, match="Incomplete"):
        action_pairs(cases[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        action_pairs(cases + [cases[0]])


def test_augmented_dual_matches_the_registered_primal_objective():
    rng = np.random.default_rng(47)
    midpoint = rng.normal(size=(8, 5))
    difference = rng.normal(size=(8, 5))
    residual = rng.normal(size=(16, 5))
    fits = _fit_directions(midpoint, difference, residual)
    centered = midpoint - midpoint.mean(0)
    base = np.concatenate((centered, difference))
    ridge = 0.01 * np.square(base).sum(1).mean()
    covariance = centered.T @ centered + 4.0 * difference.T @ difference
    response = 4.0 * difference.T @ np.ones(len(difference))
    for name, extra in (
        ("swap", np.zeros((5, 5))),
        ("invariant", residual.T @ residual / 2.0),
    ):
        expected = np.linalg.solve(covariance + extra + ridge * np.eye(5), response)
        axes, bias = fits[name]
        assert np.allclose(axes, expected, atol=1e-10)
        assert np.isclose(bias, -midpoint.mean(0) @ expected)


def test_paraphrase_penalty_suppresses_its_designated_nuisance_direction():
    midpoint = np.zeros((4, 2))
    difference = np.ones((4, 2))
    residual = np.asarray([[0.0, 10.0], [0.0, -10.0]] * 4)
    fits = _fit_directions(midpoint, difference, residual)
    assert abs(fits["invariant"][0][1]) < abs(fits["swap"][0][1]) / 10
    assert fits["invariant"][0][0] > fits["swap"][0][0]


def test_fitted_heads_score_one_state_restore_and_preserve_training_inputs(tmp_path):
    cases, values = [], []
    for topic in (0, 1):
        for pair in range(4):
            for pole in (0, 1):
                for order in (0, 1):
                    cases.append(
                        dict(
                            topic=topic,
                            family="accept",
                            position="before",
                            pair=pair,
                            pole=pole,
                            order=order,
                        )
                    )
                    action = 1 - 2 * (pole ^ order)
                    values.append(
                        [
                            [
                                1 - 2 * topic,
                                1 - 2 * order,
                                action * (1 - 2 * topic),
                                pair / 4,
                            ]
                        ]
                    )
    states = np.asarray(values, dtype=np.float64)
    original = states.copy()
    heads = fit_action_heads(states, cases)
    assert np.array_equal(states, original)
    for name, weights in heads.items():
        assert set(weights) == {"mean", "scale", "axes", "bias", "geometry"}
        path = tmp_path / (name + ".npz")
        np.savez_compressed(path, **weights)
        with np.load(path, allow_pickle=False) as archive:
            restored = {k: archive[k].copy(order="K") for k in archive.files}
        for state, case in zip(states, cases, strict=True):
            scores = score_action(state, weights)
            assert scores == score_action(state, restored)
            assert scores[case["topic"]] * (1 - 2 * (case["pole"] ^ case["order"])) > 0
        with pytest.raises(ValueError, match="single"):
            score_action(states[:2], weights)
        with pytest.raises(ValueError, match="Nonfinite"):
            score_action(np.full_like(states[0], np.nan), weights)
