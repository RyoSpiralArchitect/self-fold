"""Action-order readouts fitted on paired swaps with an optional paraphrase penalty."""

from collections import defaultdict

import numpy as np


def action_pairs(cases):
    """Return plus-first/minus-first indices with the verdict sequence held fixed."""
    groups = defaultdict(dict)
    for index, case in enumerate(cases):
        key = (
            case["topic"],
            case["family"],
            case["position"],
            case["pair"],
            case["order"],
        )
        first_action = case["pole"] ^ case["order"]
        if first_action in groups[key]:
            raise ValueError("Duplicate member in action-order pair")
        groups[key][first_action] = index
    keys = sorted(groups)
    if not keys or any(set(groups[key]) != {0, 1} for key in keys):
        raise ValueError("Incomplete action-order exchange")
    return keys, np.asarray([[groups[key][0], groups[key][1]] for key in keys])


def _paraphrase_residuals(values, cases):
    strata = defaultdict(list)
    for index, case in enumerate(cases):
        strata[
            case["family"],
            case["position"],
            case["order"],
            case["pole"] ^ case["order"],
        ].append(index)
    residuals = np.empty_like(values)
    for indices in strata.values():
        if len(indices) < 2:
            raise ValueError("Paraphrase strata require repeated realizations")
        residuals[indices] = values[indices] - values[indices].mean(0)
    return residuals


def _fit_directions(midpoints, differences, residuals):
    center = midpoints.mean(0)
    base = np.concatenate((midpoints - center, differences))
    ridge = max(0.01 * np.square(base).sum(1).mean(), 1e-8)
    targets = np.concatenate((np.zeros(len(midpoints)), np.full(len(differences), 2.0)))
    paired = np.concatenate((midpoints - center, 2.0 * differences))
    fits = {}
    for name, nuisance in (("swap", None), ("invariant", residuals)):
        design = paired
        response = targets
        if nuisance is not None:
            # There are twice as many singleton residuals as paired midpoints.
            design = np.concatenate((paired, nuisance / np.sqrt(2.0)))
            response = np.concatenate((targets, np.zeros(len(nuisance))))
        gram = design @ design.T
        gram.flat[:: len(gram) + 1] += ridge
        axes = design.T @ np.linalg.solve(gram, response)
        fits[name] = (axes, -center @ axes)
    return fits


def fit_action_heads(states, cases):
    """Fit eight independent action contrasts using only fitting-bank statistics."""
    original = np.asarray(states)
    if (
        original.ndim != 3
        or len(original) != len(cases)
        or not np.isfinite(original).all()
    ):
        raise ValueError("Expected finite source states paired with fitting cases")
    keys, indices = action_pairs(cases)
    if not np.array_equal(np.sort(indices.ravel()), np.arange(len(cases))):
        raise ValueError("Every state must belong to one action pair")
    topics = np.asarray([case["topic"] for case in cases])
    n_topics = len(set(topics))
    if set(topics) != set(range(n_topics)):
        raise ValueError("Topic identifiers must be contiguous from zero")
    flat = np.array(original, dtype=np.float64, copy=True).reshape(len(cases), -1)
    mean, scale = flat.mean(0), flat.std(0)
    scale = scale.clip(min=max(0.01 * scale.mean(), 1e-8))
    flat -= mean
    flat /= scale
    pair_topics = np.asarray([key[0] for key in keys])
    fits = {"swap": [], "invariant": []}
    for topic in range(n_topics):
        chosen = topics == topic
        group_cases = [case for case in cases if case["topic"] == topic]
        plus, minus = indices[pair_topics == topic].T
        midpoints = (flat[plus] + flat[minus]) * 0.5
        differences = (flat[plus] - flat[minus]) * 0.5
        residuals = _paraphrase_residuals(flat[chosen], group_cases)
        for name, value in _fit_directions(midpoints, differences, residuals).items():
            fits[name].append(value)
    artifacts = {}
    for name, topic_fits in fits.items():
        axes, bias = zip(*topic_fits, strict=True)
        artifacts[name] = {
            "mean": np.ascontiguousarray(mean, dtype=np.float32),
            "scale": np.ascontiguousarray(scale, dtype=np.float32),
            "axes": np.ascontiguousarray(np.stack(axes, axis=1), dtype=np.float32),
            "bias": np.ascontiguousarray(bias, dtype=np.float32),
            "geometry": np.asarray(original.shape[1:], dtype=np.int64),
        }
    return artifacts


def score_action(state, weights):
    """Score every topic from one E state without a topic or its opposing mate."""
    if tuple(np.shape(state)) != tuple(weights["geometry"]):
        raise ValueError("Expected a single source state")
    value = np.asarray(state, dtype=np.float32).reshape(-1)
    normalized = (value - weights["mean"]) / weights["scale"]
    scores = normalized @ weights["axes"] + weights["bias"]
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite action-order score")
    return scores.tolist()
