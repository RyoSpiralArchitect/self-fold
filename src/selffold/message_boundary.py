"""Multi-edit controllers whose intermediate state crosses a bounded channel.

Development arm of docs/G_MESSAGE_BOUNDARY_PLAN.md. The same updated table is
handed to the next edit either as soft rows (`soft`, the L analog) or as the
actions its rows name (`hard`, the verbalized analog: diagnostics showed a
language roundtrip preserves the named action and drops the mixture). A small
discrete message may cross beside the table; whether it is learned from reward,
learned from supervision, supplied, or dropped separates what words alone can
carry from what an extra learned symbol adds.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .role_memory import advantage, truth_bits

FORMS = ("soft", "hard")
CHANNELS = ("reward", "supervised", "none", "supplied", "traced")
TRACE_EPS = 0.25


def mix(rows, beta):
    """Blend each member row toward its partner. beta=1 keeps the pure pair."""
    if beta == 1.0:
        return rows
    partner = rows[:, ::-1]
    return beta * rows + (1 - beta) * partner


def mark(table, weights, on, eps=TRACE_EPS):
    """Write the row-0 role violation into the mixture, not into any symbol.

    When the off/off world selects the switched action, row 0 is blended toward
    the mean of the other rows. The trace is invisible to a hard argmax boundary
    but visible in soft weights.
    """
    other = table[:, 1:].mean(1, keepdim=True)
    marked = (1 - eps) * table[:, :1] + eps * other
    table = torch.where(on[:, None, None], torch.cat([marked, table[:, 1:]], 1), table)
    wother = weights[:, 1:].mean(1, keepdim=True)
    wmarked = (1 - eps) * weights[:, :1] + eps * wother
    weights = torch.where(
        on[:, None, None], torch.cat([wmarked, weights[:, 1:]], 1), weights
    )
    return table, weights


def bits_to_int(bits):
    return (bits.long() * (1 << torch.arange(bits.shape[-1], device=bits.device))).sum(-1)


def harden(table):
    """The symbolic residue of a row: which single action it names."""
    eye = torch.eye(table.shape[-1], device=table.device, dtype=table.dtype)
    return eye[table.argmax(-1)]


def hard_rows(rows):
    return harden(rows)


def convention_bits(mask, width):
    """The supplied code: whether the off/off world selects the switched row.

    Extra message positions carry a fixed zero so wider channels can be tested
    without silently adding information.
    """
    first = truth_bits(mask)[:, 0].long()
    return torch.stack(
        [first] + [torch.zeros_like(first)] * (width - 1), -1
    )


class BoundaryMemory(nn.Module):
    def __init__(self, seed, width=1):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        self.width = width
        self.address = nn.Parameter(torch.randn(2**width, 2, generator=rng) * 0.05)
        self.message = nn.Parameter(torch.randn(5 * width, generator=rng) * 0.05)

    def read(self, table, message):
        # Each pointer sees row contrast and the incoming message, never action IDs or a prior trigger.
        difference = (table - table[:, :1]).abs().sum(-1)
        relative = difference / difference.amax(-1, keepdim=True).clamp_min(1e-12)
        return self.address[message, :, None] * relative[:, None, :]

    def write_logits(self, mask):
        raw = (truth_bits(mask).float() * 2 - 1) @ self.message[: 4 * self.width].view(
            4, self.width
        )
        return raw + self.message[4 * self.width :]

    def step(self, table, message, mask, uniform=None, floor=0.0):
        pointer = self.read(table, message)
        logits = self.write_logits(mask)
        row_prob = (1 - floor) * pointer.softmax(-1) + floor / 4
        bit_prob = (1 - floor) * logits.sigmoid() + floor / 2
        if uniform is None:
            choice = pointer.argmax(-1)
            out_bits = (logits > 0).long()
        else:
            choice = (uniform[:, :2, None] > row_prob.cumsum(-1)).sum(-1).clamp_max(3)
            out_bits = (uniform[:, 2 : 2 + self.width] < bit_prob).long()
        picked = table.gather(1, choice[..., None].expand(-1, -1, table.shape[-1]))
        result = torch.where(
            truth_bits(mask)[..., None], picked[:, 1:2], picked[:, 0:1]
        )
        row_logp = row_prob.gather(-1, choice[..., None]).squeeze(-1).log().sum(-1)
        bit_logp = torch.where(out_bits.bool(), bit_prob, 1 - bit_prob).log().sum(-1)
        return result, out_bits, choice, row_logp, bit_logp, pointer, logits


def recode_indices(index, choice, mask):
    """Environment bookkeeping for exact native reward lookup. The learner never receives this field."""
    values = index.gather(1, choice)
    return torch.where(truth_bits(mask), values[:, 1:2], values[:, 0:1]), values


def boundary_bits(out_bits, mask, channel, width):
    """What actually crosses to the next edit under each channel."""
    if channel == "none":
        return torch.zeros_like(out_bits)
    if channel == "supplied":
        return convention_bits(mask, width)
    return out_bits


def boundary_table(result, form):
    return harden(result) if form == "hard" else result


def train_batch(model, initial, gold, native_credit, masks, randoms, form, channel, floor):
    """Two on-policy edits; the reward channel gets two scalar native correctness scores."""
    # All arms start from the same compiled table; the channel only governs
    # what each update hands to the next edit.
    table = initial
    batch = len(table)
    message = torch.zeros(batch, dtype=torch.long, device=table.device)
    ancestry = torch.tensor([0, 0, 0, 1]).expand(batch, 4).to(table.device)
    guide = torch.zeros_like(gold) if channel != "supervised" else gold
    tag_targets = [convention_bits(masks[:, t], model.width) for t in range(2)]
    row_log, credits, guide_losses, writer_log, writer_ce = [], [], [], [], []
    for t in range(2):
        result, out_bits, choice, logrow, logbit, ptr, logits = model.step(
            table, message, masks[:, t], randoms[:, t], floor
        )
        ancestry, picked = recode_indices(ancestry, choice, masks[:, t])
        # Environment averages the two label orders, with equal weight for the two true branches.
        rewards = native_credit[torch.arange(batch)[:, None], picked, torch.arange(2)]
        credits.append(rewards.mean(-1))
        row_log.append(logrow)
        mass = (ptr.softmax(-1) @ table).gather(-1, guide[..., None]).squeeze(-1)
        guide_losses.append(-mass.clamp_min(1e-10).log().mean())
        writer_log.append(logbit)
        writer_ce.append(
            nn.functional.binary_cross_entropy_with_logits(logits, tag_targets[t].float())
        )
        table = boundary_table(result, form)
        message = bits_to_int(boundary_bits(out_bits, masks[:, t], channel, model.width))
    r1, r2 = credits
    if channel == "reward":
        loss = -(
            row_log[0] * advantage(r1 + r2)
            + writer_log[0] * advantage(r2)
            + row_log[1] * advantage(r2)
            + writer_log[1] * advantage(r2)
        ).mean() / 4
    elif channel == "supervised":
        loss = (guide_losses[0] + guide_losses[1] + writer_ce[0] + writer_ce[1]) / 4
    else:
        # none and supplied leave the writer to the environment; only the row
        # pointer is learnable, still by outcome reward.
        loss = -(row_log[0] * advantage(r1 + r2) + row_log[1] * advantage(r2)).mean() / 2
    return loss, (float(r1.mean()), float(r2.mean()))


@torch.no_grad()
def boundary_step(model, table, message, mask, form, channel):
    result, out_bits, *_ = model.step(table, message, mask)
    carried = boundary_bits(out_bits, mask, channel, model.width)
    return boundary_table(result, form), bits_to_int(carried), carried


@torch.no_grad()
def rollout_boundary(model, form, channel, rows, keys, masks):
    """Four-edit strings over the given masks; indices compress only after the real update."""
    from itertools import product

    from .role_memory import native_row_indices
    from .role_trials import donor_indices

    n = len(rows)
    paths = torch.tensor(list(product(range(len(masks)), repeat=4)))
    num = len(paths)
    instructions = torch.tensor(masks)[paths].repeat(n, 1)
    donors = donor_indices(keys)
    # Boundary arms differ only in what an update hands onward; the first edit
    # always reads the same compiled table, while index compression compares
    # against the row form the channel actually carries.
    carried = harden(rows) if form == "hard" else rows
    ref = carried.repeat_interleave(num, 0)
    init = rows.repeat_interleave(num, 0)[:, [0, 0, 0, 1]]
    message = torch.zeros(len(init), dtype=torch.long)
    table1, message1, written1 = boundary_step(
        model, init, message, instructions[:, 0], form, channel
    )
    first = native_row_indices(table1, ref).reshape(n, num, 4)
    second, fourth, messages = [], [], [written1.reshape(n, num, -1)]
    depths = {0: first}
    for route in ("matched", "opposed", "equivalent", "erase"):
        ids = donors[{"opposed": 1, "equivalent": 2}.get(route, 0)]
        start = table1.reshape(n, num, 4, 16)[ids].reshape(-1, 4, 16)
        msg = message1.reshape(n, num)[ids].reshape(-1)
        if route == "erase":
            msg = torch.zeros_like(msg)
        base = carried[ids].repeat_interleave(num, 0)
        table2, msg2, w2 = boundary_step(
            model, start, msg, instructions[:, 1], form, channel
        )
        table3, msg3, w3 = boundary_step(
            model, table2, msg2, instructions[:, 2], form, channel
        )
        table4, _, _ = boundary_step(
            model, table3, msg3, instructions[:, 3], form, channel
        )
        if route == "matched":
            messages += [w2.reshape(n, num, -1), w3.reshape(n, num, -1)]
            depths[1] = native_row_indices(table2, base).reshape(n, num, 4)
            depths[2] = native_row_indices(table3, base).reshape(n, num, 4)
            depths[3] = native_row_indices(table4, base).reshape(n, num, 4)
        second.append(native_row_indices(table2, base).reshape(n, num, 4))
        fourth.append(native_row_indices(table4, base).reshape(n, num, 4))
    return dict(
        paths=paths.numpy(),
        first=first.numpy(),
        depths=np.stack([depths[d] for d in range(4)], 2),
        second=torch.stack(second, 1).numpy(),
        fourth=torch.stack(fourth, 1).numpy(),
        routes=np.asarray(["matched", "opposed", "equivalent", "erase"]),
        written=torch.stack(messages, 2).numpy().astype(np.uint8),
    )


@torch.no_grad()
def twins_boundary(model, form, channel, rows, masks):
    """Identical boundary state, opposite required answers; only the message may differ."""
    from itertools import product

    from .role_memory import native_row_indices

    representatives = masks[::2]
    if any(a ^ b != 15 for a, b in zip(representatives, masks[1::2])):
        raise ValueError("Need complement-paired masks")
    grid = torch.tensor(list(product(range(len(rows)), representatives, masks)))
    ix, prior, current = grid.T
    base = harden(rows)[ix] if form == "hard" else rows[ix]
    shared = torch.where(truth_bits(prior)[:, :, None], base[:, 1:2], base[:, :1])
    results, bits = [], []
    for history in (prior, 15 - prior):
        written = (
            convention_bits(history, model.width)
            if channel == "supplied"
            else (model.write_logits(history) > 0).long()
        )
        incoming = (
            torch.zeros(len(shared), dtype=torch.long)
            if channel == "none"
            else bits_to_int(written)
        )
        conditions = []
        for message in (incoming, torch.zeros_like(incoming)):
            revised, _, _ = boundary_step(
                model, shared, message, current, form, channel
            )
            conditions.append(
                native_row_indices(revised, base).reshape(
                    len(rows), len(representatives), len(masks), 4
                )
            )
        results.append(torch.stack(conditions, -2))
        bits.append(written.reshape(len(rows), len(representatives), len(masks), -1))
    return dict(
        twins=torch.stack(results, -3).numpy(),
        written=torch.stack(bits, -3).numpy(),
    )
