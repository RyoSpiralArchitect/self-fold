"""Run on Furnace after installing the HF extra. Random configs, NOT pretrained checkpoints."""
import pytest
import torch
from selffold.host import SlotHost
from selffold.tiny import ByteTokenizer

@pytest.mark.hf
@pytest.mark.parametrize("family",["mistral","llama","gemma2","phimoe"])
def test_real_hf_architecture_without_download(family):
 hf=pytest.importorskip("transformers",reason="Transformers not installed; family qualification NOT RUN")
 names={"mistral":("MistralConfig","MistralForCausalLM"),
        "llama":("LlamaConfig","LlamaForCausalLM"),
        "gemma2":("Gemma2Config","Gemma2ForCausalLM"),
        "phimoe":("PhimoeConfig","PhimoeForCausalLM")}
 config_cls,model_cls=[getattr(hf,x) for x in names[family]]
 opts=dict(vocab_size=259,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
           num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=2048,
           attention_dropout=0.,bos_token_id=1,eos_token_id=2,pad_token_id=0)
 if family=="gemma2":opts.update(head_dim=8,query_pre_attn_scalar=8)
 if family=="phimoe":opts.update(num_local_experts=4,num_experts_per_tok=2,rope_scaling=None)
 cfg=config_cls(**opts);cfg._attn_implementation="eager"
 model=model_cls(cfg).eval()
 h=SlotHost(model,ByteTokenizer(),slots=3,max_new_tokens=2,max_context=1536)
 assert h.native_parity()["exact"]
 s=h.encode("I value evidence.");n=h.update_latent(s,"plain")
 assert n.slots.shape==(3,32) and torch.isfinite(n.slots).all()
 if family=="phimoe":
  assert any("router_logits" in name for name in h.last_trace)
