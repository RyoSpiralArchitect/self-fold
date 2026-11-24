from pathlib import Path
import json
import runpy

import pytest
import torch


def test_worse_donor_receives_the_paired_role_gradient(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    loss = runpy.run_path(str(scripts / "fit_paired_role_writer.py"))["role_loss"]
    margins = torch.tensor([4.0, -2.0, -2.0, 4.0], requires_grad=True)
    loss(margins, True, 2.0).backward()
    assert margins.grad.tolist() == pytest.approx(
        [0.0, -0.4910069, -0.4910069, 0.0], abs=1e-6
    )
    assert loss(margins, True, 2.0) > loss(margins, False, 2.0)


def test_archived_pairs_reconstruct_from_reports():
    study = Path(__file__).resolve().parents[1] / "provenance/paired_writer_20261123"
    result = runpy.run_path(str(study / "verify.py"))["verify"](study / "produce")
    assert result == json.loads((study / "verification.json").read_text())
