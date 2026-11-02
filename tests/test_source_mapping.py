"""Check train-only coordinates, source independence and bound serialization."""

import json
from pathlib import Path

import pytest
import torch

from selffold.records import digest
from selffold.source_mapping import (
    fit_source_ridge,
    load_source_mapper,
    save_source_mapper,
)

ROOT = Path(__file__).resolve().parents[1]


def test_source_map_fits_paired_rows_keeps_input_only_contract_and_supports_residual_gradients(
    tmp_path,
):
    generator = torch.Generator().manual_seed(41)
    sources = torch.randn(20, 2, 9, generator=generator)
    teachers = 2 * sources + torch.arange(18).reshape(1, 2, 9)
    mapper, stats = fit_source_ridge(sources, teachers, rank=3)
    assert stats["training_sources"] == 20
    assert torch.allclose(mapper.source_mean, sources.flatten(1).mean(0), atol=1e-6)
    predicted = mapper(sources)
    assert torch.nn.functional.mse_loss(predicted, teachers) < 0.001
    assert torch.equal(mapper(sources[0]), mapper(sources[:1])[0])
    loss = (predicted + 1).square().mean()
    loss.backward()
    assert mapper.up.grad is not None and mapper.up.grad.norm() > 0
    assert all(value.grad is None for value in mapper.buffers())
    host = {"model": "toy", "width": 9, "slots": 2}
    metadata = {"base_host_sha256": digest(host), "lexical_sha256": "a" * 64}
    path = tmp_path / "mapper.npz"
    save_source_mapper(path, mapper, metadata=metadata)
    restored, _, _ = load_source_mapper(
        path, host_identity=host, lexical_sha256="a" * 64, device="cpu"
    )
    assert torch.equal(mapper(sources), restored(sources))
    with pytest.raises(ValueError, match="binding"):
        load_source_mapper(
            path, host_identity=host, lexical_sha256="b" * 64, device="cpu"
        )


def test_source_fixture_has_exact_clause_reversals_new_wordings_and_topic_separation():
    data = json.loads((ROOT / "data/source_lexical_20261103.json").read_text())
    old = json.loads((ROOT / "data/slot_contrasts.dev.json").read_text())
    old_sources = {
        text for g in old["groups"] for pole in ("plus", "minus") for text in g[pole]
    }
    current = []
    groups = data["groups"]
    assert len(groups) == 12 and sum(g["split"] == "train" for g in groups) == 8
    assert {g["id"] for g in groups if g["exposure"] == "new_transfer_topic"} == {
        "photos",
        "recipes",
    }
    for group in groups:
        for pole in ("plus", "minus"):
            source = group[pole]
            assert len(source) == 4
            for first, second in ((0, 1), (2, 3)):
                separator = ". I avoid " if first == 0 else ". I pass over "
                preferred, found, rejected = source[first].partition(separator)
                assert found and rejected.endswith(".")
                reverse_clause = (
                    "I avoid " if first == 0 else "I pass over "
                ) + rejected
                assert source[second] == reverse_clause + " " + preferred + "."
            current.extend(source)
    assert len(set(current)) == 96 and not set(current) & old_sources
