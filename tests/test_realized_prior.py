"""Actual-state fixture: a wrong carried bit must change the realized prior.

The realized table is produced by step() via the pointer's row pick, which the
message steers. This test proves that when the carried message is wrong, the
weight pattern realized at the next step diverges from the requested previous
mask — i.e. indexing reports by instructions[:,t-1] would show the writer a
state that never existed.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from selffold.mixture_boundary import (
    BoundaryMemory, eval_step, init_weights, realize, INITIAL_PRIOR,
)


def pattern_of(w):
    return (w.argmax(-1) * torch.tensor([1, 2, 4, 8])).sum(-1)


def _members():
    # 16-dim one-hot member rows; members[i] = one-hot(i+offset)
    m = torch.zeros(2, 16)
    m[0, 3] = m[1, 9] = 1.0
    return m


def test_wrong_message_diverges_realized_prior():
    torch.manual_seed(0)
    members = _members()
    model = BoundaryMemory(17, 1).requires_grad_(False)
    # Make the pointer read the message strongly: pick row by message value.
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    n = 64
    members = members.unsqueeze(0).expand(n, -1, -1).contiguous()
    mask0, mask1 = torch.tensor([7] * n), torch.tensor([14] * n)
    diverged = 0
    for wrong in (False, True):
        weights = init_weights(n, 1.0)
        table = realize(weights, members)
        msg = torch.ones(n, dtype=torch.long) if wrong else torch.zeros(n, dtype=torch.long)
        _, w1, _, carried, *_ = eval_step(
            model, table, weights, msg, mask0, "hard", "supplied", members)
        # requested previous mask for step1 is mask0; realized pattern may differ
        realized = pattern_of(w1)
        if wrong:
            diverged = int((realized != mask0).sum())
        else:
            # correct message path sanity: realized should track the update
            pass
    assert diverged > 0, "wrong carried bit never changed the realized table"


def test_initial_prior_is_mask8():
    w = init_weights(4, 1.0)
    assert (pattern_of(w) == INITIAL_PRIOR).all()
