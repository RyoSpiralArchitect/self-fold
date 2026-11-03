"""Recurrent row addressing with a learned binary message."""

from __future__ import annotations

import torch
from torch import nn


def truth_bits(masks):
    return ((masks[..., None] >> torch.arange(4, device=masks.device)) & 1).bool()


def edit_paths(masks):
    """All test pairs. Two further edits form one balanced continuation per pair."""
    n = len(masks)
    return [
        [masks[i], masks[j], masks[(i + j) % n], masks[(i + 2 * j + 1) % n]]
        for i in range(n)
        for j in range(n)
    ]


class RoleMemory(nn.Module):
    def __init__(self, seed):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        self.address = nn.Parameter(torch.randn(2, 2, generator=rng) * 0.05)
        self.message = nn.Parameter(torch.randn(5, generator=rng) * 0.05)

    def read(self, table, bit):
        # Each pointer sees row contrast and the incoming bit, never action IDs or a prior trigger.
        difference = (table - table[:, :1]).abs().sum(-1)
        relative = difference / difference.amax(-1, keepdim=True).clamp_min(1e-12)
        return self.address[bit, :, None] * relative[:, None, :]

    def write(self, mask):
        return (truth_bits(mask).float() * 2 - 1) @ self.message[:4] + self.message[4]

    def step(self, table, bit, mask, uniform=None, floor=0.0):
        pointer = self.read(table, bit)
        tag = self.write(mask)
        row_prob = (1 - floor) * pointer.softmax(-1) + floor / 4
        bit_prob = (1 - floor) * tag.sigmoid() + floor / 2
        if uniform is None:
            choice = pointer.argmax(-1)
            output_bit = (tag > 0).long()
        else:
            choice = (uniform[:, :2, None] > row_prob.cumsum(-1)).sum(-1).clamp_max(3)
            output_bit = (uniform[:, 2] < bit_prob).long()
        picked = table.gather(1, choice[..., None].expand(-1, -1, table.shape[-1]))
        result = torch.where(
            truth_bits(mask)[..., None], picked[:, 1:2], picked[:, 0:1]
        )
        row_logp = row_prob.gather(-1, choice[..., None]).squeeze(-1).log().sum(-1)
        tag_logp = torch.where(output_bit.bool(), bit_prob, 1 - bit_prob).log()
        return result, output_bit, choice, row_logp, tag_logp, pointer, tag


def recode_indices(index, choice, mask):
    """Environment bookkeeping for exact native reward lookup. The learner never receives this field."""
    values = index.gather(1, choice)
    return torch.where(truth_bits(mask), values[:, 1:2], values[:, 0:1]), values


def advantage(value):
    # A baseline computed from the other episodes does not observe this episode's action.
    return ((value - value.mean()) * (len(value) / (len(value) - 1))).detach()


def train_batch(model, initial, gold, native_credit, masks, randoms, arm, floor):
    """Two on-policy edits; the reward arm gets two scalar native correctness scores."""
    table = initial
    batch = len(table)
    bit = torch.zeros(batch, dtype=torch.long)
    ancestry = torch.tensor([0, 0, 0, 1]).expand(batch, 4)
    use_reward = float(arm == "reward")
    # Fixed dummy targets in the inactive loss for reward learning keep the two graphs shape-matched.
    guide = torch.zeros_like(gold) if arm == "reward" else gold
    tag_target = (
        torch.zeros(batch) if arm == "reward" else truth_bits(masks[:, 0])[:, 0].float()
    )
    row_log, credits, guide_losses = [], [], []
    for t in range(2):
        result, written, choice, logrow, logtag, ptr, tag = model.step(
            table, bit, masks[:, t], randoms[:, t], floor
        )
        ancestry, picked = recode_indices(ancestry, choice, masks[:, t])
        # Environment averages the two label orders, with equal weight for the two true branches.
        rewards = native_credit[torch.arange(batch)[:, None], picked, torch.arange(2)]
        credits.append(rewards.mean(-1))
        row_log.append(logrow)
        mass = (ptr.softmax(-1) @ table).gather(-1, guide[..., None]).squeeze(-1)
        guide_losses.append(-mass.clamp_min(1e-10).log().mean())
        if t == 0:
            writer_log = logtag
            writer_ce = nn.functional.binary_cross_entropy_with_logits(tag, tag_target)
        table, bit = result, written * int(arm != "table_only")
    r1, r2 = credits
    rl = (
        -(
            row_log[0] * advantage(r1 + r2)
            + writer_log * advantage(r2)
            + row_log[1] * advantage(r2)
        ).mean()
        / 3
    )
    guided = (guide_losses[0] + guide_losses[1] + writer_ce) / 3
    return (1 - use_reward) * guided + use_reward * rl, (
        float(r1.mean()),
        float(r2.mean()),
    )


@torch.no_grad()
def final_edit(model, table, bit, mask, no_bit=False):
    table, written, *_ = model.step(table, bit, mask)
    return table, written * int(not no_bit)


def native_row_indices(table, rows):
    """Lossless compression for downstream replay, only after the real-valued update."""
    equal = (table[:, :, None, :] == rows[:, None, :, :]).all(-1)
    if not equal.any(-1).all():
        raise ValueError("A final soft row does not occur in its source pair")
    return equal.long().argmax(-1).to(torch.uint8)
