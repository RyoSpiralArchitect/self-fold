import numpy as np
import pytest
import torch
from selffold import prompts
from selffold.recipient_learning import (
    RecipientAdapter, batched_recipient_logits, recipient_loss, save_recipient, load_recipient,
)
from selffold.records import State


class SeparateAttention(torch.nn.Module):
    """Small CPU fixture exposing the same q/v projection boundaries as Gemma."""
    def __init__(self, width):
        super().__init__()
        self.q_proj = torch.nn.Linear(width, width, bias=False)
        self.k_proj = torch.nn.Linear(width, width, bias=False)
        self.v_proj = torch.nn.Linear(width, width, bias=False)
        self.o_proj = torch.nn.Linear(width, width, bias=False)

    def forward(self, query, key, value, *, attn_mask, need_weights=False):
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        scores = q @ k.transpose(-1, -2) / query.shape[-1]**.5
        scores = scores.masked_fill(attn_mask, float("-inf"))
        output = self.o_proj(torch.softmax(scores, dim=-1) @ v)
        return output, None


@pytest.fixture
def recipient_host(host):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(831)
        for layer in host.layers:
            layer.self_attn = SeparateAttention(host.width)
    host.model.requires_grad_(False)
    return host


def task(host):
    ids, positions = host.render(prompts.task_prompt("A = copper. B = silver. Choose A or B."))
    slots = host.encode("I use the copper tray.").slots
    inputs = torch.tensor([ids])
    mask = torch.ones_like(inputs)
    return inputs, mask, slots[None], torch.tensor([positions[prompts.INPUT_MARKER]]), [len(ids)-1]


def test_native_recipient_gradient_is_scoped_and_inference_parity_holds(recipient_host):
    host = recipient_host
    args = task(host)
    original = {n: p.detach().clone() for n, p in host.model.named_parameters()}
    before, _ = host.forward(args[0][0].tolist(), state=State(slots=args[2][0]), input_positions=args[3][0].tolist())
    rng_before = torch.random.get_rng_state().clone()
    adapter = RecipientAdapter(host, rank=3, last_layers=1)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert sum(p.numel() for p in adapter.parameters()) == 4 * host.width * 3
    active_initial = batched_recipient_logits(host, adapter, *args)
    assert torch.equal(before[0, -1], active_initial[0])
    gold = host.tokenizer.encode("A", add_special_tokens=False)
    loss = recipient_loss(host, adapter, *args, gold)
    loss.backward()
    assert all(p.grad is None for p in host.model.parameters())
    assert sum(f.up.grad.abs().sum() for f in adapter.factors) > 0
    assert not adapter._active and not host.layers[host.cut]._forward_pre_hooks
    torch.optim.SGD(adapter.parameters(), lr=.05).step()
    actual = batched_recipient_logits(host, adapter, *args)
    assert not torch.equal(active_initial, actual)
    with adapter.active():
        replay, _ = host.forward(args[0][0].tolist(), state=State(slots=args[2][0]), input_positions=args[3][0].tolist())
    assert torch.equal(actual[0], replay[0, -1])
    inactive, _ = host.forward(args[0][0].tolist(), state=State(slots=args[2][0]), input_positions=args[3][0].tolist())
    assert torch.equal(before, inactive)
    assert all(torch.equal(value, original[name]) for name, value in host.model.named_parameters())
    adapter.close()
    assert all(not host.layers[-1].self_attn.__getattr__(name)._forward_hooks for name in ("q_proj", "v_proj"))


def test_recipient_scope_and_injection_restore_after_exception(recipient_host, monkeypatch):
    host = recipient_host
    args = task(host)
    adapter = RecipientAdapter(host, rank=3, last_layers=1)
    with adapter.active():
        with pytest.raises(RuntimeError, match="test exception"):
            with adapter.active():
                raise RuntimeError("test exception")
        assert adapter._active
    assert not adapter._active
    def fail(**kwargs):
        raise RuntimeError("backbone exception")
    monkeypatch.setattr(host.model, "forward", fail)
    with pytest.raises(RuntimeError, match="backbone exception"):
        batched_recipient_logits(host, adapter, *args)
    assert not adapter._active and not host.layers[host.cut]._forward_pre_hooks
    adapter.close()
    with pytest.raises(ValueError, match="closed"):
        with adapter.active():
            pass


def test_recipient_artifact_roundtrip_and_failure_cleanup(recipient_host, tmp_path):
    host = recipient_host
    args = task(host)
    adapter = RecipientAdapter(host, rank=3, last_layers=1)
    with torch.no_grad():
        for factor in adapter.factors:
            factor.up.fill_(.01)
    expected = batched_recipient_logits(host, adapter, *args).detach()
    path = tmp_path / "recipient.npz"
    sha = save_recipient(path, adapter, host.identity, dataset_sha256="a"*64, source_sha256="b"*64, training={"steps": 1})
    adapter.close()
    restored, metadata, actual_sha = load_recipient(path, host)
    assert sha == actual_sha and metadata["architecture"] == restored.architecture
    assert all(not p.requires_grad for p in restored.parameters())
    assert torch.equal(expected, batched_recipient_logits(host, restored, *args))
    restored.close()
    with np.load(path, allow_pickle=False) as z:
        arrays = {name: z[name].copy() for name in z.files}
    first = next(name for name in arrays if name.endswith(".down"))
    arrays[first] = arrays[first][:1]
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **arrays)
    with pytest.raises(ValueError, match="Invalid recipient artifact parameter"):
        load_recipient(bad, host)
    assert host._recipient_adapter is None
    assert not host.layers[-1].self_attn.q_proj._forward_hooks
    changed = dict(host.identity)
    host.identity["cut"] += 1
    with pytest.raises(ValueError, match="another host"):
        load_recipient(path, host)
    host.identity.clear()
    host.identity.update(changed)


def test_recipient_batch_validates_masks_and_keeps_state_gradient(recipient_host):
    host = recipient_host
    ids, mask, slots, positions, at = task(host)
    adapter = RecipientAdapter(host, rank=3, last_layers=1)
    source = slots.clone().requires_grad_()
    gold = host.tokenizer.encode("A", add_special_tokens=False)
    loss = recipient_loss(host, adapter, ids, mask, source, positions, at, gold)
    loss.backward()
    assert source.grad is not None and source.grad.abs().sum() > 0
    for invalid_positions in (positions.repeat(1, 2), positions*0):
        with pytest.raises(ValueError, match="slot position"):
            batched_recipient_logits(host, adapter, ids, mask, slots, invalid_positions, at)
    masked = mask.clone()
    masked[0, at[0]] = 0
    with pytest.raises(ValueError, match="masked token"):
        batched_recipient_logits(host, adapter, ids, masked, slots, positions, at)
    with torch.inference_mode(), pytest.raises(ValueError, match="gradient path"):
        recipient_loss(host, adapter, ids, mask, slots, positions, at, gold)
    adapter.close()


def test_batched_recipient_maps_each_sources_positions_and_causal_score(recipient_host):
    host = recipient_host
    adapter = RecipientAdapter(host, rank=3, last_layers=1)
    source = [host.encode(text).slots for text in ("I use copper.", "I use silver.")]
    prompts_to_score = [prompts.task_prompt("Choose A."), "Extra context. " + prompts.task_prompt("Choose between A and B.")]
    rendered = [host.render(text) for text in prompts_to_score]
    count = max(len(ids) for ids, _ in rendered)
    ids = torch.tensor([value + [host.placeholder_id] * (count-len(value)) for value, _ in rendered])
    mask = torch.tensor([[1]*len(value) + [0]*(count-len(value)) for value, _ in rendered])
    positions = torch.tensor([locations[prompts.INPUT_MARKER] for _, locations in rendered])
    at = [len(value)-1 for value, _ in rendered]
    batch = batched_recipient_logits(host, adapter, ids, mask, torch.stack(source), positions, at)
    with adapter.active():
        for i, (value, _) in enumerate(rendered):
            single, _ = host.forward(value, state=State(slots=source[i]), input_positions=positions[i].tolist())
            assert torch.allclose(batch[i], single[0, -1], atol=1e-5, rtol=1e-5)
    adapter.close()
