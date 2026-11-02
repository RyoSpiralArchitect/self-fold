"""Conditional training still needs a separate full-vocabulary output screen."""
import importlib.util
from pathlib import Path
import sys
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("continuity_fit", SCRIPTS / "fit_continuity_recipient.py")
FIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIT)


def test_conditional_loss_can_improve_without_valid_native_output():
    logits = torch.tensor([[0., 4., 0., 1., 0.], [0., 1., 0., 4., 0.]], requires_grad=True)
    gold = torch.tensor([1, 3])
    conditional = FIT.training_loss(logits, gold, [1, 3], "native_binary")
    outside = logits.detach().clone()
    outside[:, 4] = 100.
    assert torch.equal(conditional.detach(), FIT.training_loss(outside, gold, [1, 3], "native_binary"))
    assert (outside.argmax(-1) == 4).all()
    assert FIT.training_loss(outside, gold, [1, 3], "native_full") > 90
    conditional.backward()
    assert logits.grad[:, [0, 2, 4]].abs().sum() == 0
    assert logits.grad[:, [1, 3]].abs().sum() > 0
    assert FIT.run_config("native_full")["candidate_order"] == ["raw", "standardized"]
    assert FIT.run_config("native_binary")["candidate_order"] == ["raw_binary"]
