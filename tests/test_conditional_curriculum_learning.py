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
