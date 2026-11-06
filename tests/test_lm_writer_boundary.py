"""CPU-only regression tests for the message actually crossing W-RITE boundaries."""
import itertools

import numpy as np
import pytest
import torch

from selffold.message_boundary import BoundaryMemory, convention_bits
from selffold.mixture_boundary import (
    boundary_weights, eval_step, init_weights, realize, rollout_w, twins_w,
)
from selffold.role_memory import truth_bits


def oracle():
    actor = BoundaryMemory(17)
    with torch.no_grad():
        actor.address.copy_(torch.tensor([[-1., 1.], [1., -1.]]))
        # Deliberately contradict supplied/LM bits to test telemetry.
        actor.message.zero_()
        actor.message[-1] = -10.
    return actor


def members(n=1):
    return torch.eye(16)[:2].unsqueeze(0).repeat(n, 1, 1)


def test_twin_world_axis_and_erased_histories():
    masks = [6, 9, 10, 5, 12, 3]
    got = twins_w(oracle(), 'hard', 'supplied', members(), masks)
    assert got.shape == (1, 3, 6, 2, 2, 4)
    target = truth_bits(torch.tensor(masks)).numpy().astype(float)
    np.testing.assert_array_equal(got[0, :, :, 0, 0], np.broadcast_to(target, (3, 6, 4)))
    np.testing.assert_array_equal(got[0, :, :, 1, 0], np.broadcast_to(1-target, (3, 6, 4)))
    # Erasure leaves identical inputs, hence identical outputs in both histories.
    np.testing.assert_array_equal(got[..., 0, 1, :], got[..., 1, 1, :])


def test_lm_and_supplied_twins_match_for_the_same_convention():
    masks = [6, 9, 10, 5]
    matrix = (torch.arange(16) & 1).unsqueeze(0)
    np.testing.assert_array_equal(
        twins_w(oracle(), 'hard', 'lm', members(), masks, lm_matrix=matrix),
        twins_w(oracle(), 'hard', 'supplied', members(), masks),
    )


@pytest.mark.parametrize('channel', ['lm', 'supplied', 'none'])
def test_eval_step_records_carried_not_unused_writer_bits(channel):
    m = members(2)
    w = init_weights(2, 1.)
    masks = torch.tensor([7, 8])
    lm = torch.tensor([[1], [0]]) if channel == 'lm' else None
    _, _, message, written = eval_step(
        oracle(), realize(w, m), w, torch.zeros(2, dtype=torch.long),
        masks, 'hard', channel, m, lm_bits=lm,
    )
    torch.testing.assert_close(message, written[:, 0])
    target = torch.zeros(2, 1, dtype=torch.long) if channel == 'none' else convention_bits(masks, 1)
    torch.testing.assert_close(written, target)


@pytest.mark.parametrize('bad', [None, torch.tensor([1, 0]), torch.tensor([[2], [0]]),
                                  torch.tensor([[float('nan')], [0.]])])
def test_bad_lm_messages_are_rejected(bad):
    with pytest.raises(ValueError):
        boundary_weights(torch.zeros(2, 1, dtype=torch.long), torch.tensor([7, 8]), 'lm', 1, bad)


def test_unknown_channel_rejected():
    with pytest.raises(ValueError):
        boundary_weights(torch.zeros(1, 1), torch.tensor([7]), 'lmm', 1)


def test_rollout_records_actual_lm_bits_and_uses_donors():
    keys = [(0, f, 0, p, o) for f, p, o in itertools.product(range(2), repeat=3)]
    matrix = (torch.arange(16) & 1).repeat(len(keys), 1)
    # Opposed sources have deliberately different messages, so the donor path is observable.
    for i, key in enumerate(keys):
        if key[3]:
            matrix[i] = 1-matrix[i]
    got = rollout_w(oracle(), 'hard', 'lm', members(len(keys)), keys, [6, 9], lm_matrix=matrix)
    masks = torch.tensor([6, 9])[torch.from_numpy(got['paths'])]
    expected = matrix[:, masks[:, :3]].numpy()[..., None]
    np.testing.assert_array_equal(got['written'], expected)


def test_missing_matrix_is_not_a_zero_message_channel():
    with pytest.raises(ValueError):
        twins_w(oracle(), 'hard', 'lm', members(), [6, 9])
