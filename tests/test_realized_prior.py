"""Actual-state fixture: when can the realized prior diverge from the request?

Under `hard` boundaries result_w argmax = the mask bit regardless of which row
the pointer picked, so realized == requested structurally — the write4/write5
identity is expected there. Under `soft` boundaries a wrong message picks a
different row and the carried mixture genuinely diverges. Both are tested so
the claim is bounded by form, not assumed.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from selffold.mixture_boundary import (
    BoundaryMemory, eval_step, init_weights, realize, rollout_w, INITIAL_PRIOR,
)


def pattern_of(w):
    return (w.argmax(-1) * torch.tensor([1, 2, 4, 8])).sum(-1)


def _members(n=1):
    m = torch.zeros(2, 16)
    m[0, 3] = m[1, 9] = 1.0
    return m.unsqueeze(0).expand(n, -1, -1).contiguous()


def _message_sensitive_model():
    model = BoundaryMemory(17, 1).requires_grad_(False)
    with torch.no_grad():
        model.address.zero_()
        model.message.zero_()
        # message=0 -> strong positive row contrast; message=1 -> opposite sign,
        # so the picked row differs by message whenever rows differ at all.
        model.address[0] = torch.tensor([5.0, -5.0])
        model.address[1] = torch.tensor([-5.0, 5.0])
    return model


def _divergent_weights(n):
    # Non-uniform carried weights so row choice actually matters.
    w = torch.zeros(n, 4, 2)
    w[:, 0, 1] = w[:, 1, 0] = w[:, 2, 1] = w[:, 3, 0] = 1.0  # pattern 0b0101
    w += torch.rand(n, 4, 2) * 0.05
    return w


def test_hard_form_wrong_message_can_diverge():
    # result_w copies the PICKED row's weight values, so a message-steered
    # wrong pick flips the argmax whenever rows disagree — hard form is not
    # structurally divergence-free, only empirically rare.
    members = _members(4)
    model = _message_sensitive_model()
    weights = _divergent_weights(4)
    table = realize(weights, members)
    out = {}
    for msg_val in (0, 1):
        _, w1, *_ = eval_step(model, table, weights, torch.tensor([msg_val] * 4),
                              torch.tensor([14] * 4), "hard", "supplied", members)
        out[msg_val] = pattern_of(w1)
    assert not torch.equal(out[0], out[1]), "message never steered the carried state"


def test_soft_form_wrong_message_diverges():
    members = _members(4)
    model = _message_sensitive_model()
    weights = _divergent_weights(4)
    table = realize(weights, members)
    out = {}
    for msg_val in (0, 1):
        _, w1, *_ = eval_step(model, table, weights, torch.tensor([msg_val] * 4),
                              torch.tensor([14] * 4), "soft", "supplied", members)
        out[msg_val] = w1
    assert not torch.equal(out[0], out[1]), "wrong message never changed soft weights"


def test_initial_prior_is_mask8():
    assert (pattern_of(init_weights(4, 1.0)) == INITIAL_PRIOR).all()
