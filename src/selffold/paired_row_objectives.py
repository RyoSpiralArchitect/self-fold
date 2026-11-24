"""Teacher-forced branch objectives for opposed donors with identical four-row states."""

from math import log

import torch
import torch.nn.functional as F

# Unit coefficients, constant across seeds and updates. Both extras exclude the default.
ARMS = {"mean": (0, 0), "row_ce": (1, 0), "agreement": (0, 1), "combined": (1, 1)}


def branch_objective(scores, targets, arm, margin):
    """scores[pair, donor, field, action], ordered by aligned action ID within each pair."""
    if scores.ndim != 4 or scores.shape[1:] != (2, 5, 2) or scores.shape[0] == 0:
        raise ValueError("Expected a nonempty batch of [2 donors, 5 fields, 2 choices]")
    if targets.shape != scores.shape[:-1] or targets.dtype != torch.long:
        raise ValueError("Expected one integer target per field")
    if not torch.equal(targets[:, 0, :4], targets[:, 1, :4]) or not torch.equal(
        targets[:, 0, 4], 1 - targets[:, 1, 4]
    ):
        raise ValueError("Twin rows must share targets and defaults must be opposite")
    row_weight, agree_weight = ARMS[arm]
    lexical = F.cross_entropy(scores.reshape(-1, 2), targets.reshape(-1))
    default = scores[:, :, 4]
    signed = (default[..., 0] - default[..., 1]) * (1 - 2 * targets[:, :, 4])
    role = F.softplus(margin - signed).mean()
    rows = scores[:, :, :4]
    row_ce = F.cross_entropy(rows.reshape(-1, 2), targets[:, :, :4].reshape(-1))

    # JS in nats for each pair and coordinate. Both donors receive gradients.
    left, right = rows.log_softmax(-1).unbind(1)
    mixture = torch.logaddexp(left, right) - log(2)
    js = 0.5 * ((left.exp() * (left - mixture) + right.exp() * (right - mixture)).sum(-1)).mean()
    return {
        "lexical": lexical,
        "role": role,
        "row_ce": row_ce,
        "row_js": js,
        "auxiliary": row_weight * row_ce + agree_weight * js,
    }


def batch_objective(logits, ids, batch, arm, margin):
    """Gather each action's first divergent token using its own causal prefix."""
    active = torch.zeros_like(ids, dtype=torch.bool)
    for i, (seq, prompt_len, choices) in enumerate(batch):
        if len(choices) != 5:
            raise ValueError("Expected five canonical report fields")
        active[i, prompt_len - 1 : len(seq) - 1] = True
    token_ce = F.cross_entropy(logits[active].float(), ids.roll(-1, dims=1)[active])
    fields = [c for _, _, choices in batch for c in choices]
    pos = torch.tensor([c[0] for c in fields], device=ids.device)
    tokens = torch.tensor([c[1] for c in fields], device=ids.device).reshape(-1, 2, 5, 2)
    if not torch.equal(tokens[:, 0], tokens[:, 1]):
        raise ValueError("Twin action branches have different token order")
    targets = torch.tensor([c[2] for c in fields], device=ids.device).reshape(-1, 2, 5)
    which = torch.arange(len(batch), device=ids.device).repeat_interleave(5)
    scores = logits[which[:, None], pos[:, None], tokens.reshape(-1, 2)].float()
    parts = branch_objective(scores.reshape_as(tokens), targets, arm, margin)
    loss = token_ce + parts["lexical"] + parts["role"] + parts["auxiliary"]
    return loss, {"token_ce": token_ce, **parts}
