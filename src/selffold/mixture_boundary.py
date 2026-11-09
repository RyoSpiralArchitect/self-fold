"""Boundary-channel controllers over genuinely mixed intermediate states.

Extension of message_boundary.py (provenance/message_boundary_20261106). Rows
are tracked as member-space weights W (batch, rows, 2 members); the realized
table is w * member1 + (1-w) * member0. This keeps every boundary transform
exact: soft handoffs carry the mixture, hard handoffs carry only its argmax
member, and the `traced` channel can write role information into weights that a
symbolic boundary erases.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

from .message_boundary import BoundaryMemory, TRACE_EPS, bits_to_int, convention_bits
from .role_memory import advantage, truth_bits
from .roundtrip import pattern_of


def mix_members(rows, beta):
    """Blend each member row toward its partner; beta=1 keeps the pure pair."""
    return beta * rows + (1 - beta) * rows.flip(1)


def realize(weights, members):
    """W (b, 4, 2) over member rows (b, 2, 16) -> realized table (b, 4, 16)."""
    return torch.einsum("bwm,bmd->bwd", weights, members)


def init_weights(batch, beta, device=None):
    """Layout [m0, m0, m0, m1] under member mixing."""
    w = torch.tensor([[beta, 1 - beta]] * 3 + [[1 - beta, beta]], device=device)
    return w.expand(batch, 4, 2).clone()


def mark(weights, on, eps=TRACE_EPS):
    """Blend row 0 toward the mean of rows 1-3 when the off/off role flipped.

    Deterministic and table-local; eps < 0.5 keeps each marked row's dominant
    member unchanged, so a hard boundary cannot see the trace.
    """
    other = weights[:, 1:].mean(1, keepdim=True)
    marked = (1 - eps) * weights[:, :1] + eps * other
    return torch.where(
        on[:, None, None], torch.cat([marked, weights[:, 1:]], 1), weights
    )


def boundary_weights(out_bits, mask, channel, width, lm_bits=None):
    if channel in ("none", "traced", "lmtext"):
        # lmtext's carried payload is a feature vector, installed by the
        # caller — the integer slot here is never read.
        return torch.zeros_like(out_bits)
    if channel == "supplied":
        return convention_bits(mask, width)
    if channel == "lm":
        if (not isinstance(lm_bits, torch.Tensor)
                or lm_bits.shape != out_bits.shape
                or lm_bits.device != out_bits.device
                or not torch.all((lm_bits == 0) | (lm_bits == 1))):
            raise ValueError("LM messages must be explicit binary tensors matching out_bits")
        return lm_bits.long()
    if channel not in ("reward", "supervised"):
        raise ValueError(f"Unknown boundary channel: {channel}")
    return out_bits


INITIAL_PRIOR = 8  # actual initial member layout [m0,m0,m0,m1] = mask 0b1000


def _validate_lm_matrix(channel, model, members, masks, matrix):
    """Reject missing/partial LM messages instead of silently becoming no-message."""
    if channel != "lm":
        return
    if (model.width != 1 or not isinstance(matrix, torch.Tensor)
            or matrix.dim() not in (2, 3) or matrix.shape[0] != len(members)
            or matrix.shape[-1] != 16
            or matrix.device != members.device):
        raise ValueError("LM rollout requires width=1 and a [sources,16] or [sources,priors,16] bit matrix")
    if not masks or any(m < 0 or m > 15 for m in masks):
        raise ValueError("LM masks must be nonempty four-world masks")
    selected = matrix[..., masks]
    if not torch.all((selected == 0) | (selected == 1)):
        raise ValueError("LM matrix is missing a required binary message")


def step(model, table, weights, message, mask, form, channel, members,
         uniform=None, floor=0.0, eps=TRACE_EPS, lm_bits=None):
    """One update plus the boundary transform that hands state to the next edit.

    `table` is the realized input for this edit's pointer. The returned
    (result_table, result_w) is the carried state after `form`/`channel` act.
    """
    pointer = model.read(table, message)
    logits = model.write_logits(mask)
    row_prob = (1 - floor) * pointer.softmax(-1) + floor / 4
    bit_prob = (1 - floor) * logits.sigmoid() + floor / 2
    if uniform is None:
        choice = pointer.argmax(-1)
        out_bits = (logits > 0).long()
    else:
        choice = (uniform[:, :2, None] > row_prob.cumsum(-1)).sum(-1).clamp_max(3)
        out_bits = (uniform[:, 2 : 2 + model.width] < bit_prob).long()
    picked = table.gather(1, choice[..., None].expand(-1, -1, table.shape[-1]))
    wpicked = weights.gather(1, choice[..., None].expand(-1, -1, 2))
    on = truth_bits(mask)
    result_w = torch.where(on[..., None], wpicked[:, 1:2], wpicked[:, 0:1])
    if channel == "traced":
        result_w = mark(result_w, on[:, 0], eps)
    if form == "hard":
        # The symbolic boundary keeps only the dominant member; the trace
        # (eps < 0.5) cannot change it.
        result_w = torch.nn.functional.one_hot(result_w.argmax(-1), 2).to(result_w)
        result_table = torch.nn.functional.one_hot(
            realize(result_w, members).argmax(-1), table.shape[-1]
        ).to(table)
    else:
        result_table = realize(result_w, members)
    row_logp = row_prob.gather(-1, choice[..., None]).squeeze(-1).log().sum(-1)
    bit_logp = torch.where(out_bits.bool(), bit_prob, 1 - bit_prob).log().sum(-1)
    carried = bits_to_int(boundary_weights(out_bits, mask, channel, model.width, lm_bits))
    return result_table, result_w, out_bits, carried, row_logp, bit_logp, pointer, logits


def train_batch(model, members, reward, masks, randoms, form, channel,
                floor, beta=1.0, lm_bits=None, rt_bits=None, cases=None,
                rt_feats=None):
    """Two on-policy edits scored by realized-row native correctness.

    reward: callable (realized_descriptor, mask_at_t) -> (batch, 4 worlds, 2 orders)
    correctness indicator. When rt_bits is given, the carried message is the
    decoded report of THIS transition, keyed by its realized
    (source, prior, mask, out) — computed post-step, so coverage must hold
    under the exploration floor's arbitrary picks.
    """
    weights = init_weights(len(members), beta)
    table = realize(weights, members)
    batch = len(members)
    if rt_feats is not None:
        message = torch.zeros(batch, rt_feats.shape[-1])
    else:
        message = torch.zeros(batch, dtype=torch.long)
    row_log, bit_log, credits, supervised_losses = [], [], [], []
    for t in range(2):
        prior_p = pattern_of(weights) if (rt_bits is not None or rt_feats is not None) else None
        table, weights, out_bits, carried, logrow, logbit, ptr, wl = step(
            model, table, weights, message, masks[:, t], form, channel, members,
            uniform=randoms[:, t], floor=floor,
            lm_bits=(lm_bits[:, t : t + 1] if lm_bits is not None
                     else (torch.zeros(batch, 1, dtype=torch.long)
                           if rt_bits is not None else None)),
        )
        if rt_feats is not None:
            message = rt_feats[cases, prior_p, masks[:, t], pattern_of(weights)]
            if torch.isnan(message).any():
                raise ValueError("RoundTrip text lookup outside generated coverage")
        elif rt_bits is not None:
            carried = rt_bits[cases, prior_p, masks[:, t], pattern_of(weights)]
            if (carried < 0).any():
                raise ValueError("RoundTrip lookup outside generated coverage")
        descriptor = weights[..., 1] if form == "soft" else weights.argmax(-1).float()
        credits.append(reward(descriptor, masks[:, t]).mean((-1, -2)))
        row_log.append(logrow)
        bit_log.append(logbit)
        if rt_feats is None:
            message = carried
        if channel == "supervised":
            supervised_losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    wl, convention_bits(masks[:, t], model.width).float()
                )
            )
    r1, r2 = credits
    row_rl = -(row_log[0] * advantage(r1 + r2) + row_log[1] * advantage(r2)).mean() / 2
    if channel == "reward":
        loss = -(
            row_log[0] * advantage(r1 + r2)
            + bit_log[0] * advantage(r2)
            + row_log[1] * advantage(r2)
            + bit_log[1] * advantage(r2)
        ).mean() / 4
    elif channel == "supervised":
        # Convention-supervised writer; the row pointer still learns by outcome.
        loss = row_rl + sum(supervised_losses) / len(supervised_losses)
    else:
        loss = row_rl
    return loss, (float(r1.mean()), float(r2.mean()))


@torch.no_grad()
def eval_step(model, table, weights, message, mask, form, channel, members,
              lm_bits=None):
    result, w, out_bits, carried, *_ = step(
        model, table, weights, message, mask, form, channel, members,
        lm_bits=lm_bits,
    )
    # out_bits belongs to the controller's sampled writer. For LM/supplied/none
    # it is NOT the message crossing the boundary. Record the actual message.
    written = (carried[:, None] >> torch.arange(model.width, device=carried.device)) & 1
    return result, w, carried, written


def enumerate_w(beta, eps=TRACE_EPS, depth=4):
    """All member-weight values a traced channel can realize within `depth` edits.

    Conservative enumeration over pick/mark dynamics; used to pre-score every
    realizable vector.
    """
    states = {tuple(sorted([1 - beta] * 3 + [beta]))}
    # rows 0-2 hold w=1-beta (member0-dominant), row 3 holds w=beta (member1).
    values = set()
    for _ in range(depth):
        nxt = set()
        for st in states:
            for picks in itertools.product(st, repeat=4):
                nxt.add(tuple(sorted(picks)))
                m = ((1 - eps) * picks[0] + eps * sum(picks[1:]) / 3,) + picks[1:]
                nxt.add(tuple(sorted(m)))
        for st in nxt:
            values.update(st)
        states = nxt
    return sorted(values)


@torch.no_grad()
def rollout_w(model, form, channel, members, keys, masks, beta=1.0, lm_matrix=None):
    """Four-edit strings over `masks`; realized rows stay in member space.

    members: (n, 2, 16) already-mixed member rows for the arm's beta.
    Returns per-depth member-weight descriptors for every route.
    """
    from .role_trials import donor_indices

    _validate_lm_matrix(channel, model, members, masks, lm_matrix)
    n = len(members)
    paths = torch.tensor(list(itertools.product(range(len(masks)), repeat=4)))
    num = len(paths)
    instructions = torch.tensor(masks)[paths].repeat(n, 1)
    donors = donor_indices(keys)
    carried_members = members.repeat_interleave(num, 0)
    weights = init_weights(n * num, beta)
    table = realize(weights, carried_members)
    message = torch.zeros(n * num, dtype=torch.long)
    case_of = torch.arange(n * num) // num

    def bits_for(mask_vec, prior_vec, cases):
        if lm_matrix is None:
            return None
        if lm_matrix.dim() == 3:
            return lm_matrix[cases, prior_vec, mask_vec].unsqueeze(-1)
        return lm_matrix[cases, mask_vec].unsqueeze(-1)

    table, weights, message, w0 = eval_step(
        model, table, weights, message, instructions[:, 0], form, channel,
        carried_members,
        lm_bits=bits_for(instructions[:, 0], pattern_of(weights), case_of),
    )
    depths = [weights[..., 1].reshape(n, num, 4)]
    second, fourth, written, mismatch = [], [], [w0.reshape(n, num, -1)], []
    for route in ("matched", "opposed", "equivalent", "erase"):
        ids = donors[{"opposed": 1, "equivalent": 2}.get(route, 0)]
        start_w = weights.reshape(n, num, 4, 2)[ids].reshape(-1, 4, 2)
        start_members = carried_members.reshape(n, num, 2, 16)[ids].reshape(-1, 2, 16)
        msg = message.reshape(n, num)[ids].reshape(-1)
        if route == "erase":
            msg = torch.zeros_like(msg)
        start_table = realize(start_w, start_members)
        w2, msg2 = start_w, msg
        ws = []
        route_cases = torch.as_tensor(ids).repeat_interleave(num)
        for t in range(1, 4):
            realized_prior = pattern_of(w2)
            if route == "matched":
                # Measured divergence: realized carried pattern vs the mask
                # that was requested at t-1 (per path, before this edit).
                mismatch.append((realized_prior != instructions[:, t - 1]).reshape(n, num))
            table, w2, msg2, wt = eval_step(
                model, start_table, w2, msg2, instructions[:, t], form, channel,
                start_members,
                lm_bits=bits_for(instructions[:, t], realized_prior, route_cases),
            )
            start_table = table
            ws.append(w2)
            if route == "matched" and t < 3:
                written.append(wt.reshape(n, num, -1))
        if route == "matched":
            depths += [w[..., 1].reshape(n, num, 4) for w in ws]
        second.append(ws[0][..., 1].reshape(n, num, 4))
        fourth.append(ws[2][..., 1].reshape(n, num, 4))
    return dict(
        paths=paths.numpy(),
        depths=np.stack([d.numpy() for d in depths], 2).astype(np.float16),
        second=np.stack([s.numpy() for s in second], 1).astype(np.float16),
        fourth=np.stack([s.numpy() for s in fourth], 1).astype(np.float16),
        routes=np.asarray(["matched", "opposed", "equivalent", "erase"]),
        written=torch.stack(written, 2).numpy().astype(np.uint8),
        written_semantics=np.asarray("carried_message_bits_v2"),
        # (n, 3, paths): realized prior != requested previous mask, per edit.
        prior_mismatch=np.stack(
            [m.numpy() if isinstance(m, torch.Tensor) else m for m in mismatch], 1),
    )


@torch.no_grad()
def rollout_rt(model, form, members, keys, masks, rt_bits, beta=1.0, rt_feats=None):
    """Four-edit strings where each decoded report feeds the next edit.

    Unlike rollout_w's (prior, mask) lookup, the report crossing the boundary
    is keyed by the transition's REALIZED triple (prior, mask, out): the
    writer saw the actual post-edit table, including pointer errors. The
    decoded bit becomes the next message verbatim.
    """
    from .role_trials import donor_indices

    if rt_feats is not None:
        if (not isinstance(rt_feats, torch.Tensor) or rt_feats.dim() != 5
                or rt_feats.shape[0] != len(members)
                or tuple(rt_feats.shape[1:4]) != (16, 16, 16)):
            raise ValueError("RT text rollout requires a [sources,16,16,16,K] feature matrix")
    elif (not isinstance(rt_bits, torch.Tensor) or rt_bits.dim() != 4
            or rt_bits.shape[0] != len(members)
            or tuple(rt_bits.shape[1:]) != (16, 16, 16)):
        raise ValueError("RT rollout requires a [sources,16,16,16] bit matrix")
    if model.width != 1 or not masks:
        raise ValueError("RT rollout requires width=1 and nonempty masks")
    n = len(members)
    paths = torch.tensor(list(itertools.product(range(len(masks)), repeat=4)))
    num = len(paths)
    instructions = torch.tensor(masks)[paths].repeat(n, 1)
    donors = donor_indices(keys)
    carried_members = members.repeat_interleave(num, 0)
    weights = init_weights(n * num, beta)
    table = realize(weights, carried_members)
    message = (torch.zeros(n * num, rt_feats.shape[-1]) if rt_feats is not None
               else torch.zeros(n * num, dtype=torch.long))
    case_of = torch.arange(n * num) // num
    channel = "lmtext" if rt_feats is not None else "lm"

    def rt_step(table, weights, message, cur, cases, mem):
        prior_p = pattern_of(weights)
        table, w2, _, _ = eval_step(
            model, table, weights, message, cur, form, channel, mem,
            lm_bits=None if rt_feats is not None
            else torch.zeros(len(table), 1, dtype=torch.long),
        )
        if rt_feats is not None:
            written = rt_feats[cases, prior_p, cur, pattern_of(w2)]
            if torch.isnan(written).any():
                raise ValueError("RoundTrip text lookup outside generated coverage")
        else:
            written = rt_bits[cases, prior_p, cur, pattern_of(w2)]
            if (written < 0).any():
                raise ValueError("RoundTrip lookup outside generated coverage")
        return table, w2, written

    table, weights, w0 = rt_step(table, weights, message, instructions[:, 0],
                                 case_of, carried_members)
    depths = [weights[..., 1].reshape(n, num, 4)]
    second, fourth, written, mismatch = [], [], [w0.reshape(n, num, -1)], []
    for route in ("matched", "opposed", "equivalent", "erase"):
        ids = donors[{"opposed": 1, "equivalent": 2}.get(route, 0)]
        start_w = weights.reshape(n, num, 4, 2)[ids].reshape(-1, 4, 2)
        start_members = carried_members.reshape(n, num, 2, 16)[ids].reshape(-1, 2, 16)
        msg = (w0.reshape(n, num, -1)[ids].reshape(-1, w0.shape[-1])
               if w0.dim() == 2 else w0.reshape(n, num)[ids].reshape(-1))
        if route == "erase":
            msg = torch.zeros_like(msg)
        start_table = realize(start_w, start_members)
        w2 = start_w
        ws = []
        route_cases = torch.as_tensor(ids).repeat_interleave(num)
        for t in range(1, 4):
            if route == "matched":
                mismatch.append((pattern_of(w2) != instructions[:, t - 1]).reshape(n, num))
            start_table, w2, wt = rt_step(
                start_table, w2, msg, instructions[:, t], route_cases, start_members)
            msg = wt
            ws.append(w2)
            if route == "matched" and t < 3:
                written.append(wt.reshape(n, num, -1))
        if route == "matched":
            depths += [w[..., 1].reshape(n, num, 4) for w in ws]
        second.append(ws[0][..., 1].reshape(n, num, 4))
        fourth.append(ws[2][..., 1].reshape(n, num, 4))
    w_stack = torch.stack(written, 2)
    if rt_feats is not None:
        w_out, w_sem = w_stack.numpy().astype(np.float16), "rt_report_feats_v1"
    else:
        w_out, w_sem = w_stack.numpy().astype(np.uint8), "rt_report_bits_v1"
    return dict(
        paths=paths.numpy(),
        depths=np.stack([d.numpy() for d in depths], 2).astype(np.float16),
        second=np.stack([s.numpy() for s in second], 1).astype(np.float16),
        fourth=np.stack([s.numpy() for s in fourth], 1).astype(np.float16),
        routes=np.asarray(["matched", "opposed", "equivalent", "erase"]),
        written=w_out,
        written_semantics=np.asarray(w_sem),
        prior_mismatch=np.stack([m.numpy() for m in mismatch], 1),
    )


@torch.no_grad()
def twins_w(model, form, channel, members, masks, beta=1.0, lm_matrix=None,
            rt_bits=None, rt_feats=None):
    """Identical boundary state, opposite required answers; trace may separate them.

    For `traced` soft arms the two priors produce *different* boundary states —
    the test measures whether the weight trace alone distinguishes histories.
    With rt_bits the history message is the report of the constructed-perfect
    transition (initial layout -> history -> history), matching the supplied
    convention's perfect-write assumption.
    """
    from itertools import product as iproduct

    if rt_bits is None:
        _validate_lm_matrix(channel, model, members, masks, lm_matrix)
    elif channel == "lm" and (rt_bits.dim() != 4 or rt_bits.shape[0] != len(members)):
        raise ValueError("RT twins require a [sources,16,16,16] bit matrix")
    representatives = masks[::2]
    if not masks or len(masks) % 2 or any(a ^ b != 15 for a, b in zip(representatives, masks[1::2])):
        raise ValueError("Need complement-paired masks")
    grid = torch.tensor(list(iproduct(range(len(members)), representatives, masks)))
    ix, prior, current = grid.T
    base = members[ix]
    prior_bits, current = truth_bits(prior), current
    # Shared table in member space: row w picks member1 where the prior bit is on.
    shared_w = torch.nn.functional.one_hot(prior_bits.long(), 2).float()
    results = []
    for history in (prior, 15 - prior):
        if channel == "supplied":
            written = convention_bits(history, model.width)
        elif channel == "lmtext":
            written = rt_feats[ix, history.new_full(history.shape, INITIAL_PRIOR),
                               history, history]
        elif channel == "lm" and rt_bits is not None:
            written = rt_bits[ix, history.new_full(history.shape, INITIAL_PRIOR),
                              history, history].unsqueeze(-1)
        elif channel == "lm":
            written = (lm_matrix[ix, history.new_full(history.shape, INITIAL_PRIOR), history]
                       if lm_matrix.dim() == 3 else lm_matrix[ix, history]).unsqueeze(-1)
        else:
            written = (model.write_logits(history) > 0).long()
        if channel in ("none", "traced"):
            incoming = torch.zeros(len(shared_w), dtype=torch.long)
        elif channel == "lmtext":
            incoming = written
        else:
            incoming = bits_to_int(written)
        # Histories differ only through the channel: bit value or, for traced,
        # the mark their own prior leaves in the weights.
        hist_w = mark(shared_w, truth_bits(history)[:, 0]) if channel == "traced" else shared_w
        for erased in (False, True):
            msg = torch.zeros_like(incoming) if erased else incoming
            table = realize(hist_w, base)
            if form == "hard":
                table = torch.nn.functional.one_hot(table.argmax(-1), 16).to(table)
            revised, w2, _, _ = eval_step(
                model, table, hist_w, msg, current, form, channel, base,
                lm_bits=(torch.zeros(len(ix), 1, dtype=torch.long)
                         if channel == "lm" and rt_bits is not None
                         else None if channel == "lmtext"
                         else (lm_matrix[ix, (hist_w.argmax(-1) * torch.tensor(
                                    [1, 2, 4, 8], device=hist_w.device)).sum(-1), current]
                               if lm_matrix.dim() == 3
                               else lm_matrix[ix, current]).unsqueeze(-1)
                         if channel == "lm" else None),
            )
            results.append(w2[..., 1].reshape(len(members), len(representatives), len(masks), 4))
    # Stack the four (history, erasure) conditions BEFORE the world axis.
    # Stacking at -1 and merely reshaping interleaves worlds with conditions.
    return torch.stack(results, -2).reshape(
        len(members), len(representatives), len(masks), 2, 2, 4
    ).numpy().astype(np.float16)

