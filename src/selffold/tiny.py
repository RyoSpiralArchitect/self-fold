"""A REAL, randomly initialized small PyTorch causal decoder. Not a pretrained LLM."""
from types import SimpleNamespace
import torch
from torch import nn

class ByteTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    chat_template = "self-fold byte-tokenizer engineering fixture v1"
    def encode(self, text, add_special_tokens=False):
        return [b + 3 for b in text.encode("utf8")]
    def decode(self, ids, skip_special_tokens=True):
        return bytes([i-3 for i in ids if 3 <= i < 259]).decode("utf8", errors="replace")
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert len(messages) == 1 and messages[0]["role"] == "user"
        text = "User: " + messages[0]["content"] + "\nAssistant:"
        return self.encode(text) if tokenize else text

class TinyLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(width, 4, batch_first=True, dropout=0)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width))
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        x = self.norm1(hidden_states)
        n = x.shape[1]
        mask = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1)
        attn = self.self_attn(x, x, x, attn_mask=mask, need_weights=False)[0]
        h = hidden_states + attn
        return (h + self.mlp(self.norm2(h)),)

class TinyCausalLM(nn.Module):
    def __init__(self, width=24, depth=3, seed=47):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(259, width)
            self.model.layers = nn.ModuleList([TinyLayer(width) for _ in range(depth)])
            self.model.norm = nn.LayerNorm(width)
            self.lm_head = nn.Linear(width, 259, bias=False)
        self.config = SimpleNamespace(model_type="selffold_tiny", hidden_size=width, num_hidden_layers=depth)
        self.generation_config = SimpleNamespace(eos_token_id=2)
    def get_input_embeddings(self):
        return self.model.embed_tokens
    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True, **kwargs):
        if use_cache:
            raise ValueError("tiny fixture forbids cache")
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h, attention_mask=attention_mask)[0]
        logits = self.lm_head(self.model.norm(h))
        # Engineering fixture avoids empty decoded states, without a semantic claim.
        logits[..., :3] = -1e4
        return SimpleNamespace(logits=logits, past_key_values=None)
