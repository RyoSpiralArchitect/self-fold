from itertools import product
import json
from pathlib import Path

import torch

from selffold.role_memory import (
    RoleMemory,
    edit_paths,
    native_row_indices,
    train_batch,
    truth_bits,
)
from selffold.role_trials import rollout, twins

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "data/role_memory_20261104.json").read_text())


def test_split_crosses_complements_and_has_no_common_default_address():
    train, test = map(set, (CFG["fit_masks"], CFG["test_masks"]))
    assert train.isdisjoint(test) and train | test == set(range(1, 15))
    for split in (train, test):
        assert {15 - m for m in split} == split
        assert truth_bits(torch.tensor(sorted(split))).any(0).all()
    paths = edit_paths(CFG["test_masks"])
    assert len(paths) == 36 and len(set(tuple(p[:2]) for p in paths)) == 36
    assert all(len(p) == 4 and set(p) <= test for p in paths)


def test_a_known_address_convention_survives_four_edits_and_exact_erased_twins():
    keys = [
        (0, flag, "when", pole, order)
        for flag, pole, order in product(range(2), repeat=3)
    ]
    origins = torch.eye(16)[:2]
    rows = torch.stack([origins[[p, 1 - p]] for _, _, _, p, _ in keys])
    model = RoleMemory(11)
    with torch.no_grad():
        model.address.copy_(torch.tensor([[-6.0, 6.0], [6.0, -6.0]]))
        model.message.copy_(torch.tensor([6.0, 0.0, 0.0, 0.0, 0.0]))
    paths = edit_paths(CFG["test_masks"])
    data = rollout(model, "supervised", rows, keys, CFG["test_masks"])
    for stage, col in (("second", 1), ("fourth", 3)):
        oracle = truth_bits(torch.tensor(paths)[:, col]).to(torch.uint8)[None]
        assert (torch.from_numpy(data[stage][:, 0]) == oracle).all()
    pairs = torch.from_numpy(twins(model, "supervised", rows, CFG["test_masks"]))
    assert torch.equal(pairs[:, :, :, 0, 1], pairs[:, :, :, 1, 1])
    assert torch.equal(pairs[:, :, :, 0, 0], 1 - pairs[:, :, :, 1, 0])
    assert pairs.shape == (8, 3, 6, 2, 2, 4)


def test_reward_gradient_cannot_read_supervised_targets_and_copies_exact_rows():
    actor = RoleMemory(71)
    rows = torch.eye(16)[:2][None].expand(8, -1, -1)
    init = rows[:, [0, 0, 0, 1]]
    credit = torch.eye(2)[None].expand(8, -1, -1)
    masks = torch.tensor([[m, 15 - m] for m in CFG["fit_masks"]])
    noise = torch.rand(8, 2, 3, generator=torch.Generator().manual_seed(3))
    grads = []
    for ids in (
        torch.zeros(8, 2, dtype=torch.long),
        torch.full((8, 2), 15, dtype=torch.long),
    ):
        loss, _ = train_batch(actor, init, ids, credit, masks, noise, "reward", 0.12)
        grads.append(torch.autograd.grad(loss, tuple(actor.parameters())))
    assert all(torch.equal(a, b) for a, b in zip(*grads))
    assert any(g.abs().sum() > 0 for g in grads[0])
    assert torch.equal(
        native_row_indices(init, rows),
        torch.tensor([[0, 0, 0, 1]] * 8, dtype=torch.uint8),
    )
