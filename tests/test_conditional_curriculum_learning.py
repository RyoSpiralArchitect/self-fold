"""Check the curriculum's true-world balance without invoking pretrained model weights."""

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from conditional_curriculum_learning import weighted_native_loss


def test_weighted_microbatches_equal_the_full_balanced_curriculum_loss():
    generator = torch.Generator(device="cpu").manual_seed(47)
    full = torch.randn(32, 7, generator=generator, requires_grad=True, device="cpu")
    targets = torch.arange(32, device="cpu") % 7
    worlds = [w for _ in range(4) for w in range(4) for _ in range(2)]
    weights = torch.tensor([3.0 if w == 3 else 1.0 for w in worlds], device="cpu")
    native = torch.nn.functional.cross_entropy(full, targets, reduction="none")
    expected = (weights * native).sum() / weights.sum()
    (expected_gradient,) = torch.autograd.grad(expected, full, retain_graph=True)
    actual = sum(
        weighted_native_loss(
            full[start : start + 4],
            targets[start : start + 4],
            worlds[start : start + 4],
        )
        for start in range(0, 32, 4)
    )
    (actual_gradient,) = torch.autograd.grad(actual, full)
    assert torch.allclose(actual, expected)
    assert torch.allclose(actual_gradient, expected_gradient)
    assert (
        weights[torch.tensor(worlds, device="cpu") == 3].sum()
        == weights[torch.tensor(worlds, device="cpu") != 3].sum()
    )


def test_curriculum_loss_rejects_missing_or_unknown_world_labels():
    logits = torch.zeros(4, 7, device="cpu")
    with pytest.raises(ValueError, match="per native score"):
        weighted_native_loss(
            logits, torch.zeros(4, dtype=torch.long, device="cpu"), [0, 1]
        )
    with pytest.raises(ValueError, match="Unexpected"):
        weighted_native_loss(
            logits, torch.zeros(4, dtype=torch.long, device="cpu"), [0, 1, 2, 4]
        )


def test_zero_margin_preserves_the_exact_full_vocabulary_loss_and_gradient():
    logits = torch.tensor(
        [[0.2, -0.4, 1.0], [-0.5, 0.7, 0.4]], device="cpu", requires_grad=True
    )
    targets = torch.tensor([0, 1], device="cpu")
    original = weighted_native_loss(logits, targets, [0, 3])
    explicit = weighted_native_loss(
        logits, targets, [0, 3], margin=0.0, label_ids=[0, 1]
    )
    assert torch.equal(original, explicit)
    (expected,) = torch.autograd.grad(original, logits, retain_graph=True)
    (actual,) = torch.autograd.grad(explicit, logits)
    assert torch.equal(expected, actual)


def test_positive_margin_adds_only_the_uniform_active_gold_foil_hinge():
    logits = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.5], [-1.0, 0.5, 0.0]],
        device="cpu",
        requires_grad=True,
    )
    targets = torch.tensor([0, 0, 1], device="cpu")
    base = weighted_native_loss(logits, targets, [0, 3, 1])
    augmented = weighted_native_loss(
        logits, targets, [0, 3, 1], margin=2.0, label_ids=[0, 1]
    )
    assert torch.allclose(augmented - base, torch.tensor(3.5 / 32.0, device="cpu"))
    (actual,) = torch.autograd.grad(augmented - base, logits)
    expected = (
        torch.tensor(
            [[-1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0]], device="cpu"
        )
        / 32.0
    )
    assert torch.allclose(actual, expected, atol=1e-7)


def test_margin_microbatches_sum_to_uniform_hinge_and_balanced_native_ce():
    generator = torch.Generator(device="cpu").manual_seed(947)
    logits = torch.randn(32, 7, generator=generator, requires_grad=True, device="cpu")
    targets = torch.arange(32, device="cpu") % 2
    worlds = [w for _ in range(4) for w in range(4) for _ in range(2)]
    weights = torch.tensor([3.0 if w == 3 else 1.0 for w in worlds], device="cpu")
    ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    gold = logits.gather(-1, targets[:, None]).squeeze(-1)
    foil = logits.gather(-1, (1 - targets)[:, None]).squeeze(-1)
    expected = (weights * ce).sum() / 48.0 + torch.relu(2.0 - (gold - foil)).mean()
    (expected_gradient,) = torch.autograd.grad(expected, logits, retain_graph=True)
    actual = sum(
        weighted_native_loss(
            logits[start : start + 4],
            targets[start : start + 4],
            worlds[start : start + 4],
            margin=2.0,
            label_ids=[0, 1],
        )
        for start in range(0, 32, 4)
    )
    (actual_gradient,) = torch.autograd.grad(actual, logits)
    assert torch.allclose(actual, expected)
    assert torch.allclose(actual_gradient, expected_gradient)


@pytest.mark.parametrize("ids", [None, [0], [0, 0], [-1, 1], [0, 3], [0.0, 1.0]])
def test_positive_margin_rejects_improper_native_label_ids(ids):
    logits = torch.zeros(2, 3, device="cpu")
    targets = torch.tensor([0, 1], device="cpu")
    with pytest.raises(ValueError, match="label"):
        weighted_native_loss(logits, targets, [0, 3], margin=1.0, label_ids=ids)


def test_positive_margin_rejects_targets_outside_the_two_native_labels():
    with pytest.raises(ValueError, match="margin target"):
        weighted_native_loss(
            torch.zeros(1, 3, device="cpu"),
            torch.tensor([2], device="cpu"),
            [3],
            margin=1.0,
            label_ids=[0, 1],
        )
