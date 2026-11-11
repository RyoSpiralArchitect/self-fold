"""Completed capped reports survive a tight forward budget; no model downloads."""
from types import SimpleNamespace

import pytest
import torch

from selffold.lm_writer_reports import generate_kv


class CacheHost:
    def __init__(self, cap=1, context=10, forwards=1, eos=9):
        self.max_new_tokens, self.max_context = cap, context
        self.max_forwards = forwards
        self.device, self.eos_ids = torch.device("cpu"), {eos}
        self.meter = SimpleNamespace(forwards=0, processed_tokens=0, generated_tokens=0)
        self.model = self.Model()
        self.tokenizer = SimpleNamespace(decode=lambda ids, **_: " ".join(map(str, ids)))

    def render(self, prompt):
        return [7, 8], {}

    def _budget(self, tokens):
        if self.meter.forwards >= self.max_forwards:
            raise RuntimeError("Forward budget exhausted")

    class Model:
        config = SimpleNamespace(use_cache=True)

        def __call__(self, input_ids, past_key_values=(), **_):
            # Return a NEW cache rather than mutating an existing cache object.
            history = past_key_values + tuple(input_ids[0].tolist())
            logits = torch.zeros(1, 1, 12, device="cpu")
            logits[0, 0, len(history)] = 1
            return SimpleNamespace(logits=logits, past_key_values=history)


def test_last_permitted_token_needs_no_extra_forward():
    host = CacheHost()
    result = generate_kv(host, "prompt")
    assert result.text == "2"
    assert result.receipt()["stop"] == "length"
    assert result.receipt()["censored"] is True
    assert vars(host.meter) == dict(forwards=1, processed_tokens=2, generated_tokens=1)


@pytest.mark.parametrize("cap,context,stop", [(0, 10, "length"), (None, 2, "context_limit")])
def test_zero_output_budget_does_not_prefill(cap, context, stop):
    host = CacheHost(cap=cap, context=context, forwards=0)
    result = generate_kv(host, "prompt")
    assert result.text == ""
    assert result.receipt()["stop"] == stop
    assert result.receipt()["token_count"] == 0
    assert result.receipt()["censored"] is True
    assert vars(host.meter) == dict(forwards=0, processed_tokens=0, generated_tokens=0)


def test_context_cap_keeps_all_emitted_tokens_and_advances_immutable_cache():
    host = CacheHost(cap=None, context=5, forwards=3)
    result = generate_kv(host, "prompt")
    assert result.text == "2 3 4"
    assert result.receipt()["stop"] == "context_limit"
    assert result.receipt()["token_count"] == 3
    assert vars(host.meter) == dict(forwards=3, processed_tokens=4, generated_tokens=3)


def test_eos_needs_no_following_forward():
    host = CacheHost(cap=3, eos=2)
    result = generate_kv(host, "prompt")
    assert result.text == ""
    assert result.receipt()["stop"] == "eos"
    assert result.receipt()["stop_token_id"] == 2
    assert result.receipt()["censored"] is False
    assert host.meter.forwards == 1


def test_insufficient_budget_still_stops_an_incomplete_report():
    with pytest.raises(RuntimeError, match="Forward budget exhausted"):
        generate_kv(CacheHost(cap=2, forwards=1), "prompt")
