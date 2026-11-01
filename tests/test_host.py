import pytest
import torch
from selffold.records import State
from selffold.prompts import INPUT_MARKER,OUTPUT_MARKER
from selffold.host import SlotHost
from selffold.tiny import TinyCausalLM,ByteTokenizer


def test_native_parity(host):
 p=host.native_parity();assert p["exact"] and p["max_abs"]==0

def test_render_positions(host):
 ids,p=host.render(f"Before {INPUT_MARKER} after {OUTPUT_MARKER}")
 assert len(p[INPUT_MARKER])==3 and len(p[OUTPUT_MARKER])==3
 assert max(p[INPUT_MARKER])<min(p[OUTPUT_MARKER])
 assert all(ids[i]==host.placeholder_id for i in p[INPUT_MARKER])

def test_marker_ambiguity(host):
 with pytest.raises(ValueError):host.render(INPUT_MARKER+INPUT_MARKER)

def test_context_fail_closed(host):
 with pytest.raises(ValueError):host.render("x"*2000)

def test_generation_budget_context(host):
 host.max_context=40
 with pytest.raises(ValueError):host.generate("x"*30)

@pytest.mark.parametrize("cap,room,tokens,stop,text,stop_id", [
    (None, 5, [68, 2], "eos", "A", 2),
    (2, 2, [68, 1], "eos", "A", 1),
    (2, 5, [68, 69], "length", "AB", None),
    (None, 2, [68, 69], "context_limit", "AB", None),
    (8, 2, [68, 69], "context_limit", "AB", None),
    (None, 2, [2], "eos", "", 2),
    (None, 0, [], "context_limit", "", None),
])
def test_generation_stop_receipt(host, monkeypatch, cap, room, tokens, stop, text, stop_id):
    prompt_ids, _ = host.render("A")
    host.max_context = len(prompt_ids) + room
    host.max_new_tokens = cap
    host.eos_ids = {1, 2}
    next_tokens = iter(tokens)

    def forward(*args, **kwargs):
        logits = torch.zeros(1, 1, 259)
        logits[0, 0, next(next_tokens)] = 1
        return logits, {}

    monkeypatch.setattr(host, "forward", forward)
    output = host.generate("A")
    assert output.receipt() == {
        "text": text, "stop": stop, "token_count": len(text), "prompt_tokens": len(prompt_ids),
        "effective_max_new_tokens": min(cap, room) if cap is not None else room,
        "stop_token_id": stop_id, "censored": stop != "eos",
    }
    assert host.meter.generated_tokens == len(tokens)
    if text:
        assert output.as_state().text == text

def test_encode_and_real_latent_change(host):
 s=host.encode("I value reversible choices.");old=s.sha()
 assert s.slots.shape==(3,24)
 t=host.update_latent(s,"plain")
 assert t.slots.shape==s.slots.shape and s.sha()==old
 assert torch.isfinite(t.slots).all()
 assert t.sha()!=s.sha()
 assert "pre_cut" in host.last_trace and "post_writer" in host.last_trace
 assert "layer_1.self_attn" in host.last_trace
 assert "layer_2.mlp" in host.last_trace

def test_different_input_has_effect(host):
 s=host.encode("I value stable plans.")
 other=State(slots=-s.slots)
 a=host.update_latent(s,"plain");b=host.update_latent(other,"plain")
 assert not torch.equal(a.slots,b.slots)

def test_hooks_cleanup_on_model_exception(host,monkeypatch):
 def bad(*a,**kw):raise RuntimeError("deliberate failure")
 monkeypatch.setattr(host.model,"forward",bad)
 with pytest.raises(RuntimeError):host.encode("I value plans.")
 for m in host.model.modules():
  assert not m._forward_hooks and not m._forward_pre_hooks

def test_cache_disabled_every_time(host,monkeypatch):
 orig=host.model.forward;calls=[]
 def wrapper(*a,**kw):
  calls.append(kw["use_cache"])
  assert "past_key_values" not in kw
  return orig(*a,**kw)
 monkeypatch.setattr(host.model,"forward",wrapper)
 s=host.encode("I value evidence.");host.decode(s);host.update_latent(s,"boundary")
 assert calls and not any(calls)

def test_weights_frozen(host):
 before={k:v.clone() for k,v in host.model.state_dict().items()}
 host.update_text("I value evidence.","plain")
 assert all(not p.requires_grad and p.grad is None for p in host.model.parameters())
 assert all(torch.equal(v,host.model.state_dict()[k]) for k,v in before.items())

def test_bad_slot_shape(host):
 ids,pos=host.render(INPUT_MARKER)
 with pytest.raises(ValueError):host.forward(ids,state=State(slots=torch.zeros(2,24)),input_positions=pos[INPUT_MARKER])

@pytest.mark.parametrize("cut,writer",[(0,2),(2,1),(1,3),(-1,2)])
def test_boundaries(cut,writer):
 with pytest.raises(ValueError):SlotHost(TinyCausalLM(),ByteTokenizer(),cut=cut,writer=writer)

def test_budget_stops(host):
 host.max_forwards=0
 with pytest.raises(RuntimeError):host.encode("I value evidence.")

def test_label_scoring(host):
 s=State(text="I value evidence.")
 scores=host.score_choices(s,"Choose A or B.",["A","B"])
 assert set(scores)=={"A","B"} and scores["A"]["tokens"]==1
