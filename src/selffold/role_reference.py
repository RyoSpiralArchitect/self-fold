"""A source-derived action/role reference carried beside a report.

The reference is supplied from the initial compiled hard rows, not learned.
It is copied through each boundary, never refreshed from source labels.
The text pointer has exactly the same parameters as BoundaryMemory.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from .message_boundary import BoundaryMemory
from .mixture_boundary import eval_step, init_weights, realize, step
from .role_memory import advantage
from .role_trials import donor_indices
from .roundtrip import pattern_of


def source_reference(members):
    """Option slot of member 0 in the fixed 8-menu, 16-action hard bank."""
    if members.ndim != 3 or tuple(members.shape[1:]) != (2, 16):
        raise ValueError("Expected two hard action rows from the fixed bank")
    if not bool((((members == 0) | (members == 1)).all()
                 & (members.sum(-1) == 1).all())):
        raise ValueError("The reference requires exact one-hot action rows")
    slots = members.argmax(-1)
    if not bool(((slots[:, 0] ^ slots[:, 1]) == 1).all()):
        raise ValueError("Source members must be the two actions of one menu")
    return slots[:, 0] % 2


def attach(payload, reference):
    if reference.ndim != 1 or len(reference) != len(payload):
        raise ValueError("Reference must have one entry per carried state")
    if not bool(((reference == 0) | (reference == 1)).all()):
        raise ValueError("Reference is one binary value")
    if payload.ndim == 1:
        payload = payload[:, None]
    return torch.cat([payload, reference.to(payload)[:, None]], -1)


def intervene(reference, mode):
    if mode == "intact":
        return reference.clone()
    if mode == "fixed_zero":
        return torch.zeros_like(reference)
    if mode == "opposed":
        return 1 - reference
    raise ValueError(f"Unknown reference intervention: {mode}")


class ReferenceReader(BoundaryMemory):
    def read(self, table, message):
        if message.ndim == 1:
            return super().read(table, message)
        if message.shape[-1] != self.feat_dim + 1:
            raise ValueError("Expected report features followed by one reference bit")
        reference = message[:, -1]
        if not bool(((reference == 0) | (reference == 1)).all()):
            raise ValueError("Nonbinary carried reference")
        pointer = super().read(table, message[:, :-1].contiguous())
        # Architectural role exchange, not an additional learned operation.
        return torch.where(reference[:, None, None].bool(), pointer.flip(1), pointer)


class SlotReader(BoundaryMemory):
    """Expose the decoded comparator's previously implicit source reference."""
    def read(self, table, message):
        if message.ndim == 1:
            return super().read(table, message)
        if message.shape[-1] != 2:
            raise ValueError("Expected an absolute action slot and reference bit")
        if not torch.equal(message, message.long().to(message)):
            raise ValueError("Decoded slot and reference must be integral")
        slot, reference = message.long().T
        if not bool((((slot >= -1) & (slot <= 1))
                     & ((reference == 0) | (reference == 1))).all()):
            raise ValueError("Invalid decoded slot/reference")
        # Keep write7's missing-report fallback: role 0, independent of pole.
        role = torch.where(slot < 0, 0, slot ^ reference)
        return super().read(table, role)


def lookup(bank, cases, prior, mask, out, reference, kind):
    payload = bank[cases, prior, mask, out]
    if kind == "text":
        if torch.isnan(payload).any():
            raise ValueError("Text lookup outside the sealed feasible corpus")
    elif kind == "slot":
        if (payload < -1).any():
            raise ValueError("Slot lookup outside the sealed feasible corpus")
    else:
        raise ValueError(kind)
    return attach(payload, reference)


def train_reference_batch(model, members, reward, masks, randoms, bank, cases,
                          reference, floor):
    weights = init_weights(len(members), 1.)
    table = realize(weights, members)
    incoming = model.initial_message(len(members), members.device)
    reference = reference.clone()
    row_logs, credits = [], []
    for t in range(2):
        prior = pattern_of(weights)
        table, weights, _, _, logrow, _, _, _ = step(
            model, table, weights, incoming, masks[:, t], "hard", "lmtext", members,
            uniform=randoms[:, t], floor=floor)
        incoming = lookup(bank, cases, prior, masks[:, t], pattern_of(weights),
                          reference, "text")
        # State carry: the next writer receives this bit, not a fresh source lookup.
        reference = incoming[:, -1]
        row_logs.append(logrow)
        credits.append(reward(weights.argmax(-1).float(), masks[:, t]).mean((-1, -2)))
    first, second = credits
    loss = -(row_logs[0] * advantage(first + second)
             + row_logs[1] * advantage(second)).mean() / 2
    return loss, (float(first.mean()), float(second.mean()))


@torch.no_grad()
def rollout_reference(model, members, keys, masks, bank, reference, kind,
                      mode="intact"):
    """Four real updates; intervene once after update 1 and then carry the bit."""
    n = len(members)
    paths = torch.tensor(list(itertools.product(range(len(masks)), repeat=4)))
    count = len(paths)
    instructions = torch.tensor(masks)[paths].repeat(n, 1)
    donors = donor_indices(keys)
    mem = members.repeat_interleave(count, 0)
    weights = init_weights(n * count, 1.)
    table = realize(weights, mem)
    incoming = model.initial_message(n * count, members.device)
    cases = torch.arange(n * count) // count
    initial = reference.repeat_interleave(count)

    def advance(table, weights, message, mask, cases, mem, first_reference=None):
        prior = pattern_of(weights)
        carried = first_reference if message.ndim == 1 else message[:, -1]
        if carried is None:
            raise ValueError("Initial reference was not supplied")
        table, result, _, _ = eval_step(model, table, weights, message, mask,
                                        "hard", "lmtext", mem)
        written = lookup(bank, cases, prior, mask, pattern_of(result), carried, kind)
        return table, result, written

    table, weights, first_message = advance(table, weights, incoming, instructions[:, 0],
                                             cases, mem, initial)
    first_message[:, -1] = intervene(first_message[:, -1], mode)
    depths = [weights[..., 1].reshape(n, count, 4)]
    written = [first_message.reshape(n, count, -1)]
    second, fourth, mismatch = [], [], []
    for route in ("matched", "opposed", "equivalent", "erase"):
        ids = donors[{"opposed": 1, "equivalent": 2}.get(route, 0)]
        cur = weights.reshape(n, count, 4, 2)[ids].reshape(-1, 4, 2)
        route_members = mem.reshape(n, count, 2, 16)[ids].reshape(-1, 2, 16)
        message = first_message.reshape(n, count, -1)[ids].reshape(-1, first_message.shape[-1])
        if route == "erase":
            message = torch.zeros_like(message)
        at_boundary = message[:, -1].clone()
        current_table = realize(cur, route_members)
        route_cases = torch.as_tensor(ids).repeat_interleave(count)
        outputs = []
        for t in range(1, 4):
            if route == "matched":
                mismatch.append((pattern_of(cur) != instructions[:, t-1]).reshape(n, count))
            current_table, cur, message = advance(current_table, cur, message,
                        instructions[:, t], route_cases, route_members)
            if not torch.equal(message[:, -1], at_boundary):
                raise ValueError("Reference was refreshed or changed after the boundary")
            outputs.append(cur)
            if route == "matched" and t < 3:
                written.append(message.reshape(n, count, -1))
        if route == "matched":
            depths += [w[..., 1].reshape(n, count, 4) for w in outputs]
        second.append(outputs[0][..., 1].reshape(n, count, 4))
        fourth.append(outputs[2][..., 1].reshape(n, count, 4))
    return dict(paths=paths.numpy(),
        depths=np.stack([w.numpy() for w in depths], 2).astype(np.float16),
        second=np.stack([w.numpy() for w in second], 1).astype(np.float16),
        fourth=np.stack([w.numpy() for w in fourth], 1).astype(np.float16),
        written=torch.stack(written, 2).numpy().astype(np.float16),
        written_semantics=np.asarray(f"{kind}_report_plus_carried_source_reference_v1"),
        prior_mismatch=np.stack([w.numpy() for w in mismatch], 1),
        initial_reference=reference.numpy(),
        boundary_reference=intervene(reference, mode).numpy(),
        reference_carried_unchanged=np.asarray(True))


@torch.no_grad()
def reference_twins(model, members, masks, bank, reference, kind, mode="intact"):
    """The same constructed-twin fixture, with an explicit carried reference."""
    from .role_memory import truth_bits
    reps = masks[::2]
    if any(a ^ b != 15 for a, b in zip(reps, masks[1::2])):
        raise ValueError("Need complement-paired masks")
    grid = torch.tensor(list(itertools.product(range(len(members)), reps, masks)))
    cases, prior, current = grid.T
    base = members[cases]
    shared = torch.nn.functional.one_hot(truth_bits(prior).long(), 2).float()
    ref = intervene(reference[cases], mode)
    outcomes = []
    for history in (prior, 15-prior):
        incoming = lookup(bank, cases, torch.full_like(prior, 8), history, history, ref, kind)
        for erase in (False, True):
            message = torch.zeros_like(incoming) if erase else incoming
            _, weights, _, _ = eval_step(model, realize(shared, base), shared, message,
                                         current, "hard", "lmtext", base)
            outcomes.append(weights[..., 1].reshape(len(members), len(reps), len(masks), 4))
    return torch.stack(outcomes, -2).reshape(len(members), len(reps), len(masks), 2, 2, 4).numpy().astype(np.float16)
