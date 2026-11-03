"""Paired training on a factorial bank, followed by independent singleton inference."""

from collections import defaultdict

import numpy as np

from .matched_cues import preference_records


def crossed_records(groups, fixture, vocabulary=None):
    expanded = []
    for g in groups:
        words = g if vocabulary is None else vocabulary[g["id"]]
        n = len(words["plus"])
        if n != len(words["minus"]) or not n:
            raise ValueError("Need equal nonempty pole vocabularies")
        expanded.append(
            g
            | {
                "plus": [a for a in words["plus"] for _ in range(n)],
                "minus": words["minus"] * n,
            }
        )
    width = len(expanded[0]["plus"])
    if any(len(g["plus"]) != width for g in expanded):
        raise ValueError("Unbalanced topic vocabulary")
    side = len(
        groups[0]["plus"] if vocabulary is None else vocabulary[groups[0]["id"]]["plus"]
    )
    for c in preference_records(expanded, fixture | {"familiar_pairs": range(width)}):
        a, b = divmod(c["pair"], side)
        yield c | {"plus_index": a, "minus_index": b}


def role_pairs(cases):
    """Indices of accept-first and reject-first states at fixed action mention order."""
    matched = defaultdict(dict)
    for i, c in enumerate(cases):
        key = (
            c["topic"],
            c["family"],
            c["position"],
            c["pair"],
            c["pole"] ^ c["order"],
        )
        if c["order"] in matched[key]:
            raise ValueError("Duplicate member in role pair")
        matched[key][c["order"]] = i
    keys = sorted(matched)
    if not keys or any(matched[k].keys() != {0, 1} for k in keys):
        raise ValueError("Incomplete role exchange")
    return keys, np.asarray([[matched[k][0], matched[k][1]] for k in keys])


def null_orientations(keys):
    rng = np.random.default_rng(47)
    blocks = defaultdict(list)
    for i, (topic, family, placement, pair, first_action) in enumerate(keys):
        blocks[topic, family, placement, first_action].append(i)
    signs = np.ones(len(keys))
    for _, indices in sorted(blocks.items()):
        if len(indices) % 2:
            raise ValueError("Need even strata for the shuffled-role control")
        signs[rng.permutation(indices)[: len(indices) // 2]] = -1
    return signs


def _solve(mid, diff, mid_targets, diff_targets, *, null_last=False):
    """Fit both weights with identical ridge. The intercept belongs only to the midpoint."""
    center, intercept = mid.mean(0), mid_targets.mean(0)
    design = np.concatenate((mid - center, diff))
    targets = np.concatenate((mid_targets - intercept, diff_targets))
    gram = design @ design.T
    ridge = max(0.01 * gram.diagonal().mean(), 1e-8)
    results = {}
    for name, w in (("ordinary", 1.0), ("weighted", 4.0)):
        tags = targets[:, :-1] if null_last and name == "ordinary" else targets
        scales = np.repeat([1.0, np.sqrt(w)], len(mid))
        matrix = (scales[:, None] * gram) * scales
        matrix.flat[:: len(matrix) + 1] += ridge
        alpha = np.linalg.solve(matrix, scales[:, None] * tags)
        axes = design.T @ (scales[:, None] * alpha)
        bias = intercept[: axes.shape[1]] - center @ axes
        results[name] = (axes, bias)
    return results


def fit_exchange(states, keys, indices, router):
    x = np.array(states, dtype=np.float64, copy=True).reshape(len(states), -1)
    if (
        len(keys) * 2 != len(x)
        or indices.shape != (len(keys), 2)
        or not np.array_equal(np.sort(indices.ravel()), np.arange(len(x)))
    ):
        raise ValueError("Every fitting state must enter exactly one opposed pair")
    mean, scale = x.mean(0), x.std(0)
    scale = scale.clip(min=max(0.01 * scale.mean(), 1e-8))
    x -= mean
    x /= scale
    left, right = indices.T
    mid = (x[left] + x[right]) * 0.5
    diff = (x[left] - x[right]) * 0.5
    del x
    topics = np.array([k[0] for k in keys])
    first = np.array([1 - 2 * k[-1] for k in keys])
    n_topics = router["offset"].size - 1
    if set(topics) != set(range(n_topics)):
        raise ValueError("Router and topic bank differ")
    signs = null_orientations(keys)
    role_targets = np.column_stack((np.ones(len(keys)), signs))
    shared = _solve(
        mid, diff, np.zeros_like(role_targets), role_targets, null_last=True
    )
    local = []
    for t in range(n_topics):
        chosen = topics == t
        action = first[chosen]
        zero = np.zeros_like(action)
        local.append(
            _solve(
                mid[chosen],
                diff[chosen],
                np.column_stack((action, zero)),
                np.column_stack((zero, action)),
            )
        )
    weights = {}
    for name, (roles, bias) in shared.items():
        directions, offsets = zip(*(entry[name] for entry in local), strict=True)
        learned = dict(
            mean=mean,
            scale=scale,
            role_axes=roles,
            role_bias=bias,
            local_axes=np.stack(directions),
            local_bias=np.stack(offsets),
        )
        weights[name] = {
            k: np.ascontiguousarray(v, dtype=np.float32) for k, v in learned.items()
        } | {
            "geometry": np.array(states.shape[1:], dtype=np.int64),
            **{"router_" + k: router[k] for k in ("mean", "scale", "shared", "offset")},
        }
    return weights, signs


def score_one(state, weights):
    """Only one source tensor. All centering, slopes and topic routing come from frozen artifacts."""
    if tuple(np.shape(state)) != tuple(weights["geometry"]):
        raise ValueError("Expected a single source state")
    x = np.asarray(state, dtype=np.float32).reshape(-1)
    routed = ((x - weights["router_mean"]) / weights["router_scale"]) @ weights[
        "router_shared"
    ] + weights["router_offset"]
    topic_scores = routed[:-1]
    winners = np.flatnonzero(topic_scores == topic_scores.max())
    topic = int(winners[0]) if len(winners) == 1 else None
    z = (x - weights["mean"]) / weights["scale"]
    roles = z @ weights["role_axes"] + weights["role_bias"]
    local = z @ weights["local_axes"] + weights["local_bias"]
    if not (
        np.isfinite(routed).all()
        and np.isfinite(roles).all()
        and np.isfinite(local).all()
    ):
        raise ValueError("Nonfinite readout")
    result = {
        "topic": topic,
        "role": float(roles[0]),
        "actions": local[:, 0].tolist(),
        "directs": local[:, 1].tolist(),
    }
    if len(roles) > 1:
        result["shuffled_role"] = float(roles[1])
    return result


def load_weights(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy(order="K") for k in f.files}
