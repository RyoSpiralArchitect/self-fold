import importlib.util
from pathlib import Path
import pytest
import torch
from selffold import prompts
from selffold.codec_learning import EncoderBridge, codec_loss, codec_objectives
from selffold.policy_binding import first_divergence, pair_schedule
from selffold.records import State


def test_common_prefix_loss_uses_only_visible_prefix_and_reaches_bridge(host):
    tok = host.tokenizer
    bridge = EncoderBridge(host.width, 3)
    slots = host.encode("A source").slots
    left, right = [tok.encode(text, add_special_tokens=False) + [2] for text in ("a red", "a tan")]
    i = first_divergence(left, right)
    nll, margin = codec_objectives(host, bridge, slots, left, contrast=(i, right[i]))
    assert nll.item() == codec_loss(host, bridge, slots, "a red", end_token_id=2).item()
    torch.nn.functional.softplus(-margin).backward()
    assert bridge.up.grad.abs().sum() > 0
    assert all(p.grad is None for p in host.model.parameters())
    ids, pos = host.render(prompts.read_prompt())
    masked = ids + left[:i] + [host.placeholder_id] * (len(left)-i)
    logits, _ = host.forward(masked, state=State(slots=bridge(slots)), input_positions=pos[prompts.INPUT_MARKER])
    at = logits[0, len(ids)+i-1].float()
    assert margin.item() == pytest.approx((at[left[i]]-at[right[i]]).item(), abs=1e-5)


def test_pairs_and_schedule_keep_opposites_together():
    for left, right in [([1, 2], [1, 2]), ([1, 2], [1, 3]), ([1, 2], [1, 4, 2])]:
        with pytest.raises(ValueError):
            first_divergence(left, right)
    pairs = [(2*i, 2*i+1) for i in range(96)]
    batch = pair_schedule(pairs)
    assert batch == pair_schedule(pairs)
    assert len(batch) == 64 and all(len(row) == 8 for row in batch)
    assert all(b == a+1 for row in batch for a, b in zip(row[::2], row[1::2]))
    assert len({x for row in batch[:24] for x in row}) == 192


def test_ignoring_opposed_donor_can_get_half_accuracy_and_zero_switches():
    path = Path(__file__).resolve().parents[1] / "provenance/policy_binding_20261103/summarize.py"
    spec = importlib.util.spec_from_file_location("binding_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    items = []
    for pole in ("plus", "minus"):
        for wording in ("p0", "p1"):
            for world in (0, 1):
                for order in (0, 1):
                    gold = "A" if bool(world) ^ bool(order) ^ (pole == "plus") else "B"
                    items.append(dict(pole=pole, wording=wording, world=world, order=order, A=1, B=0, gold=gold))
    assert module.group_metrics(items, 2) == {"accuracy": (8, 16), "opposed": (0, 8),
                                              "wrappers": (4, 8), "world_switch": (0, 8)}
    with pytest.raises(ValueError):
        module.group_metrics(items + items[:1], 2)
