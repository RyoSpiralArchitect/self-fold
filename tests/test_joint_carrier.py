"""Gradient and scope contracts for a source-only learned E plus native recipient."""
import pytest
import torch
from test_recipient_learning import recipient_host, task
from selffold.carrier_probe import description_layout
from selffold.recipient_learning import (
    RecipientAdapter, differentiable_source_slots, recipient_loss, save_recipient, load_recipient,
)


def adapters(host):
    source = RecipientAdapter(host, rank=3, last_layers=1, seed=947, layer_indices=[0], scope="source_encoding")
    reader = RecipientAdapter(host, rank=3, last_layers=1, seed=47)
    return source, reader


def test_source_prefix_matches_native_capture_and_skips_head(recipient_host):
    host = recipient_host
    source, reader = adapters(host)
    text = "I use copper when the signal is on."
    ids, positions, _ = description_layout(host, text, "workspace", forward_length=256)
    _, baseline = host.forward(ids, capture_positions=positions)
    calls = []
    handle = host.model.lm_head.register_forward_hook(lambda *args: calls.append(True))
    try:
        slots, layout = differentiable_source_slots(host, source, text)
    finally:
        handle.remove()
    assert torch.equal(slots.detach(), baseline["pre_cut"][0])
    assert slots.requires_grad and not calls
    assert layout["source_prefix_layers_executed"] == host.cut and not layout["native_output_head_executed"]
    assert not source._active and not reader._active and not host.layers[host.cut]._forward_pre_hooks
    source.close()
    assert host._recipient_adapter is reader
    reader.close()


def test_end_to_end_source_recipient_gradients_leave_backbone_frozen(recipient_host):
    host = recipient_host
    source, reader = adapters(host)
    before = {name:p.detach().clone() for name,p in host.model.named_parameters()}
    inputs, mask, _, positions, score_at = task(host)
    slots, _ = differentiable_source_slots(host, source, "I use the copper tray.")
    labels = host.tokenizer.encode("A", add_special_tokens=False)
    loss = recipient_loss(host, reader, inputs, mask, slots[None], positions, score_at, labels)
    loss.backward()
    assert sum(f.up.grad.abs().sum() for f in source.factors) > 0
    assert sum(f.up.grad.abs().sum() for f in reader.factors) > 0
    assert all(p.grad is None and torch.equal(p, before[name]) for name,p in host.model.named_parameters())
    assert not source._active and not reader._active
    source.close()
    reader.close()


def test_scopes_are_exclusive_and_clean_up_on_prefix_failure(recipient_host, monkeypatch):
    host = recipient_host
    source, reader = adapters(host)
    with source.active():
        with pytest.raises(ValueError, match="Another adapter scope"):
            with reader.active():
                pass
        assert source._active and not reader._active
    def fail(*args, **kwargs):
        raise RuntimeError("prefix fixture failure")
    monkeypatch.setattr(host.layers[0], "forward", fail)
    with pytest.raises(RuntimeError, match="prefix fixture failure"):
        differentiable_source_slots(host, source, "A source.")
    assert not source._active and not reader._active and not host.layers[host.cut]._forward_pre_hooks
    source.close()
    reader.close()


def test_custom_encoder_and_legacy_recipient_artifacts_can_coexist(recipient_host, tmp_path):
    host = recipient_host
    source, reader = adapters(host)
    assert "layer_indices" not in reader.architecture and "adapter_scope" not in reader.architecture
    assert source.architecture["layer_indices"] == [0]
    source_path, reader_path = tmp_path/"source.npz", tmp_path/"reader.npz"
    for module, path in ((source, source_path), (reader, reader_path)):
        save_recipient(path, module, host.identity, dataset_sha256="a"*64, source_sha256="b"*64, training={"steps":0})
    source.close()
    loaded_source, metadata, _ = load_recipient(source_path, host)
    assert loaded_source.scope == "source_encoding" and metadata["kind"] == "scoped_source_encoder_qv_lora"
    assert host._recipient_adapter is reader
    reader.close()
    loaded_reader, metadata, _ = load_recipient(reader_path, host)
    assert loaded_reader.scope == "task" and metadata["architecture"] == reader.architecture
    assert len(host._scoped_adapters) == 2
    loaded_reader.close()
    assert host._scoped_adapters["source_encoding"] is loaded_source
    loaded_source.close()
    assert not host._scoped_adapters and host._recipient_adapter is None


def test_source_scope_rejects_wrong_layers_and_gradient_checkpointing(recipient_host, monkeypatch):
    host = recipient_host
    with pytest.raises(ValueError, match="precede"):
        RecipientAdapter(host, rank=3, last_layers=1, layer_indices=[host.cut], scope="source_encoding")
    source, reader = adapters(host)
    monkeypatch.setattr(host.model, "is_gradient_checkpointing", True, raising=False)
    with pytest.raises(ValueError, match="checkpointing disabled"):
        differentiable_source_slots(host, source, "A source.")
    source.close()
    reader.close()
