"""Reference interventions must operate on the carried state, not target labels."""
import itertools

import numpy as np
import pytest
import torch

from selffold.message_boundary import BoundaryMemory
from selffold.mixture_boundary import rollout_rt, train_batch
from selffold.role_reference import (ReferenceReader, SlotReader, attach,
    rollout_reference, source_reference, train_reference_batch)


def fixture():
    keys = [(0, f, 0, p, o) for f in (0, 1) for p in (0, 1) for o in (0, 1)]
    members = torch.stack([torch.eye(16)[[k[3], 1-k[3]]] for k in keys])
    return keys, members


def test_reference_comes_from_actual_rows_and_rejects_invalid_bank():
    _, members = fixture()
    assert source_reference(members).tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    assert torch.equal(source_reference(members.flip(1)), 1-source_reference(members))
    with pytest.raises(ValueError):
        source_reference(members * .5)
    with pytest.raises(ValueError):
        source_reference(torch.eye(16)[[0, 3]][None])


def test_gating_preserves_parameters_start_and_zero_reference_behavior():
    plain, actor = BoundaryMemory(17, 1, 32), ReferenceReader(17, 1, 32)
    assert all(torch.equal(v, actor.state_dict()[k]) for k, v in plain.state_dict().items())
    table = torch.eye(16)[[0, 0, 0, 1]][None]
    features = torch.ones(1, 32)
    original = plain.read(table, features)
    assert torch.equal(actor.read(table, attach(features, torch.zeros(1))), original)
    assert torch.equal(actor.read(table, attach(features, torch.ones(1))), original.flip(1))
    assert torch.equal(actor.read(table, actor.initial_message(1)),
                       plain.read(table, plain.initial_message(1)))


def test_unreferenced_training_matches_legacy_loss_and_gradients():
    _, members = fixture()
    torch.manual_seed(41)
    n = len(members)
    bank = torch.randn(n, 16, 16, 16, 32)
    masks = torch.tensor([[6, 9]] * n)
    noise = torch.rand(n, 2, 3)
    plain, actor = BoundaryMemory(17, 1, 32), ReferenceReader(17, 1, 32)
    def reward(descriptor, mask):
        truth = ((mask[:, None] >> torch.arange(4)) & 1)
        return (descriptor == truth).float()[..., None].expand(-1, -1, 2)
    old, _ = train_batch(plain, members, reward, masks, noise, "hard", "lmtext", .12,
                          rt_feats=bank, cases=torch.arange(n))
    new, _ = train_reference_batch(actor, members, reward, masks, noise, bank,
                            torch.arange(n), torch.zeros(n), .12)
    assert torch.equal(old, new)
    old.backward(); new.backward()
    for p, q in zip(plain.parameters(), actor.parameters()):
        assert (p.grad is None and q.grad is None) or torch.equal(p.grad, q.grad)


def test_actual_slot_report_and_carried_reference_support_four_updates():
    keys, members = fixture()
    reference = source_reference(members)
    slots = torch.empty(len(keys), 16, 16, 16, dtype=torch.long)
    for i, ref in enumerate(reference):
        slots[i] = (torch.arange(16) & 1) ^ ref
    model = SlotReader(17, 1)
    with torch.no_grad():
        model.address[:] = torch.tensor([[-1., 1.], [1., -1.]])
    masks = [6, 9, 14, 1]
    intact = rollout_reference(model, members, keys, masks, slots, reference, "slot")
    flipped = rollout_reference(model, members, keys, masks, slots, reference, "slot", "opposed")
    erased = rollout_reference(model, members, keys, masks, slots, reference, "slot", "fixed_zero")
    instructions = np.asarray(list(itertools.product(masks, repeat=4)))
    truth = (instructions[..., None] >> np.arange(4)) & 1
    assert np.array_equal(intact["depths"], np.broadcast_to(truth, intact["depths"].shape))
    assert np.array_equal(flipped["depths"][:, :, 0], intact["depths"][:, :, 0])
    assert (flipped["depths"][:, :, 1:] == 1-intact["depths"][:, :, 1:]).all()
    assert (flipped["written"][..., -1] == (1-reference.numpy())[:, None, None]).all()
    assert (erased["written"][..., -1] == 0).all()
    assert (erased["depths"][reference.numpy() == 0] == intact["depths"][reference.numpy() == 0]).all()


def test_unreferenced_rollout_preserves_all_legacy_routes():
    keys, members = fixture()
    torch.manual_seed(17)
    features = torch.randn(len(keys), 16, 16, 16, 32)
    old = rollout_rt(BoundaryMemory(17, 1, 32), "hard", members, keys, [6, 9], None,
                     rt_feats=features)
    new = rollout_reference(ReferenceReader(17, 1, 32), members, keys, [6, 9],
                             features, torch.zeros(len(keys)), "text")
    for field in ("depths", "second", "fourth", "prior_mismatch"):
        assert np.array_equal(old[field], new[field]), field
    assert np.array_equal(old["written"], new["written"][..., :-1])


def test_missing_slot_fallback_and_reference_erasure_do_not_force_ties():
    model = SlotReader(17, 1)
    with torch.no_grad():
        model.address[:] = torch.tensor([[-1., 1.], [1., -1.]])
    table = torch.eye(16)[[0, 0, 0, 1]][None]
    for ref in (0, 1):
        assert torch.equal(model.read(table, torch.tensor([[-1, ref]])),
                           model.read(table, torch.tensor([0])))
    picks = model.read(table, torch.tensor([[1, 0]])).argmax(-1)
    assert picks[0, 0] != picks[0, 1]


def test_saved_patterns_reproduce_complete_and_paired_units(tmp_path):
    from selffold.reference_evaluation import (logical_hash, paired_reference,
                                              save_compact, summarize_panels)
    from selffold.report_replay import compact_rollout
    from selffold.role_reference import reference_twins
    keys = [(g, f, 0, p, o) for g in (0, 1) for f in (0, 1) for p in (0, 1) for o in (0, 1)]
    members = torch.stack([torch.eye(16)[[2*k[0]+k[3], 2*k[0]+1-k[3]]] for k in keys])
    reference = source_reference(members)
    slots = torch.empty(len(keys), 16, 16, 16, dtype=torch.long)
    for i, ref in enumerate(reference):
        slots[i] = (torch.arange(16) & 1) ^ ref
    model = SlotReader(17, 1)
    with torch.no_grad():
        model.address[:] = torch.tensor([[-1., 1.], [1., -1.]])
    masks = [6, 9]
    cfg = dict(fit_topics=[0], fit_masks=masks, test_masks=[])
    labels = np.array([101, 202])
    top = labels[np.array([k[3] for k in keys])[:, None, None]
                 ^ np.arange(2)[None, :, None] ^ np.arange(2)]
    outputs = {}
    for mode in ("intact", "opposed"):
        rolled = rollout_reference(model, members, keys, masks, slots, reference, "slot", mode)
        rolled["twins"] = reference_twins(model, members, masks, slots, reference, "slot", mode)
        outputs[mode] = compact_rollout(rolled)
    expected = summarize_panels(outputs["intact"], keys, cfg, top, labels)
    assert expected["all_menus"]["all_four_correct"] == [256, 256]
    assert expected["primary_fit_menus"]["all_four_correct"] == [128, 128]
    assert expected["unfitted_menus"]["all_four_correct"] == [128, 128]
    paired = paired_reference(outputs["intact"], outputs["opposed"], keys, cfg, top, labels)
    assert paired["all_menus"]["flipped_when_intact_complete"] == [64, 64]
    path = tmp_path / "trajectory.npz"
    manifest = save_compact(path, outputs["intact"])
    with np.load(path, allow_pickle=False) as data:
        for field, metadata in manifest.items():
            assert logical_hash(data[field]) == metadata["c_order_sha256"]
            assert np.array_equal(data[field], outputs["intact"][field])
        assert summarize_panels(data, keys, cfg, top, labels) == expected
