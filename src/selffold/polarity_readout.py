"""Read role and action from one source state with training-only affine contrasts."""

import numpy as np


def source_cases(groups, forms, pairs):
    for topic, group in enumerate(groups):
        for frame, template in forms.items():
            for pair in pairs:
                actions = (group["plus"][pair], group["minus"][pair])
                for pole in (0, 1):
                    clauses = (
                        template["chosen"].format(actions[pole]),
                        template["rejected"].format(actions[1 - pole]),
                    )
                    for order in (0, 1):
                        yield {
                            "key": f"{group['id']}/{frame}/{pair}/{pole}/{order}",
                            "topic": topic,
                            "group": group["id"],
                            "frame": frame,
                            "pair": pair,
                            "pole": pole,
                            "order": order,
                            "source": " ".join(
                                clauses if order == 0 else clauses[::-1]
                            ),
                        }


def _axes(x, y):
    center, levels = x.mean(0), y.mean(0)
    z = x - center
    gram = z @ z.T
    penalty = max(0.01 * gram.diagonal().mean(), 1e-8)
    gram.flat[:: len(gram) + 1] += penalty
    axes = z.T @ np.linalg.solve(gram, y - levels)
    return axes, levels - center @ axes


def fit_axes(states, cases, n_topics):
    x = np.asarray(states, dtype=np.float64).reshape(len(cases), -1)
    topic, pole, order = (
        np.array([c[k] for c in cases]) for k in ("topic", "pole", "order")
    )
    if (
        set(topic) != set(range(n_topics))
        or set(pole) != {0, 1}
        or set(order) != {0, 1}
    ):
        raise ValueError("Incomplete topic, pole or order bank")
    mean, scale = x.mean(0), x.std(0)
    scale = scale.clip(min=max(0.01 * scale.mean(), 1e-8))
    x = (x - mean) / scale
    role = 1 - 2 * order
    accepted = 1 - 2 * pole
    shared, offset = _axes(x, np.column_stack((np.eye(n_topics)[topic], role)))
    by_topic = [
        _axes(x[topic == t], np.column_stack((role * accepted, accepted))[topic == t])
        for t in range(n_topics)
    ]
    slopes, biases = zip(*by_topic, strict=True)
    return {
        k: np.asarray(v, dtype=np.float32)
        for k, v in dict(
            mean=mean,
            scale=scale,
            shared=shared,
            offset=offset,
            slopes=np.stack(slopes),
            biases=np.stack(biases),
        ).items()
    }


def read_axes(state, axes):
    """Scalar inference, including all conditional scores for the oracle-topic diagnostic."""
    x = (np.asarray(state, dtype=np.float32).reshape(-1) - axes["mean"]) / axes["scale"]
    shared = x @ axes["shared"] + axes["offset"]
    conditional = x @ axes["slopes"] + axes["biases"]
    if not (np.isfinite(shared).all() and np.isfinite(conditional).all()):
        raise ValueError("Nonfinite source score")
    topics = shared[:-1]
    best = np.flatnonzero(topics == topics.max())
    return {
        "topic": int(best[0]) if len(best) == 1 else None,
        "topic_scores": topics.tolist(),
        "role": float(shared[-1]),
        "first_action": conditional[:, 0].tolist(),
        "direct": conditional[:, 1].tolist(),
    }


def predicted_class(scores, method="factor", *, oracle_topic=None):
    topic = scores["topic"] if oracle_topic is None else oracle_topic
    if topic is None:
        return None
    if method == "factor":
        value = scores["first_action"][topic] * scores["role"]
    elif method == "direct":
        value = scores["direct"][topic]
    elif method == "mention":
        value = scores["first_action"][topic]
    else:
        raise ValueError("Unknown polarity readout")
    return 2 * topic + int(value < 0) if value != 0 else None
