"""Check equal-information arms, fresh crossings and the deployable small map."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from selffold.records import digest
from selffold.wording_transfer import (
    fit_policy_directions,
    load_policy_mixture,
    save_policy_mixture,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fit_within_topic_wording import DATA, make_bank, tasks_for


def test_style_intervention_matches_words_and_samples_and_holds_out_whole_styles():
    data = json.loads(DATA.read_text())
    single, many = (make_bank(data, arm) for arm in ("single", "many"))
    assert [g["id"] for g in data["groups"]] == [
        "archive",
        "route",
        "notes",
        "temperature",
        "length",
        "clock",
        "mail",
        "music",
    ]
    assert len({q for g in data["groups"] for q in g["questions"]}) == 32
    control_keys = lambda bank: {
        (c["group"], c["a"], c["b"], c["pole"], c["source_order"]) for c in bank["fit"]
    }
    assert control_keys(single) == control_keys(many)
    for bank in (single, many):
        assert len({c["source"] for c in bank["fit"]}) == 512
        assert set(Counter(c["class_id"] for c in bank["fit"]).values()) == {32}
        assert set(c["source"] for c in bank["fit"]).isdisjoint(
            c["source"] for c in bank["fresh"]
        )
        for part, cases in bank.items():
            by_key = {c["key"]: c for c in cases}
            for c in cases:
                opposite = by_key[c["opposed_key"]]
                assert (
                    opposite["class_id"] == c["class_id"] ^ 1
                    and opposite["source_order"] == c["source_order"]
                )
                if c["source_order"] == 0:
                    clauses = c["source"].removesuffix(".").split(". ")
                    assert (
                        len(clauses) == 2
                        and by_key[c["key"][:-1] + "1"]["source"]
                        == ". ".join(clauses[::-1]) + "."
                    )
    assert {c["style"] for c in single["fit"]} == {"prefer"}
    counts = Counter((c["class_id"], c["style"]) for c in many["fit"])
    assert len(counts) == 64 and set(counts.values()) == {8}
    assert all(single[part] == many[part] for part in ("fresh", "syntax", "lexicon"))
    assert (
        len(many["fresh"]) == 128 and len(many["syntax"]) == len(many["lexicon"]) == 64
    )
    assert {c["style"] for c in many["fresh"]} == set(data["fresh_styles"])
    assert all(c["a"] >= 4 and c["b"] >= 4 for c in many["fresh"])
    assert len(tasks_for(many["fresh"][0], data["groups"][0], (2, 3))) == 4


def test_policy_directions_remove_sample_axis_support_gradients_and_reload(tmp_path):
    generator = torch.Generator().manual_seed(47)
    teachers = torch.randn(4, 2, 9, generator=generator)
    labels = torch.arange(4).repeat_interleave(8)
    states = teachers[labels] + 0.1 * torch.randn(32, 2, 9, generator=generator)
    model, stats = fit_policy_directions(states, labels, teachers)
    assert stats["contrast_dimension_at_most"] == 3 and torch.allclose(
        model.projection.sum(1), torch.zeros(18), atol=1e-5
    )
    assert torch.equal(model.weights(states).argmax(1), labels)
    assert set(dict(model.named_buffers())) == {
        "mean",
        "scale",
        "projection",
        "bias",
        "teachers",
    }
    assert all(32 not in value.shape for value in model.buffers())
    model(states).square().mean().backward()
    assert model.calibration.grad.norm() > 0 and model.renderer.up.grad.norm() > 0
    host = {"slots": 2, "width": 9}
    path = tmp_path / "map.npz"
    save_policy_mixture(
        path, model, {"base_host_sha256": digest(host), "lexical_sha256": "a" * 64}
    )
    restored, _, _ = load_policy_mixture(path, host, "a" * 64, "cpu")
    assert torch.equal(model(states), restored(states))
    with pytest.raises(ValueError, match="binding"):
        load_policy_mixture(path, host, "b" * 64, "cpu")
    with np.load(path, allow_pickle=False) as artifact:
        assert "anchors" not in artifact
