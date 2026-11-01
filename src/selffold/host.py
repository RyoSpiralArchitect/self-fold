"""Hooked single-device causal decoder; no persistent KV and no implicit text feedback."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import time
import re
import torch
from .records import State, TextOutput, digest
from . import prompts
from .transport import Transport
from .metrics import tensor_summary, router_summary


@dataclass
class Meter:
    forwards: int = 0
    processed_tokens: int = 0
    generated_tokens: int = 0

    def snapshot(self):
        return asdict(self)


class SlotHost:
    def __init__(self, model, tokenizer, *, slots=8, cut=None, writer=None,
                 max_context=1536, max_new_tokens=64, transport=None,
                 max_forwards=20000, max_processed_tokens=20_000_000,
                 wall_seconds=600, min_free_cuda_gib=2.0):
        self.model = model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.tokenizer = tokenizer
        try:
            self.layers = model.model.layers
        except AttributeError as e:
            raise ValueError("Unsupported decoder layout: expected model.model.layers; no guessed hooks") from e
        self.depth = len(self.layers)
        self.cut = max(1, self.depth // 2) if cut is None else cut
        self.writer = self.depth - 1 if writer is None else writer
        if not 1 <= self.cut <= self.writer < self.depth:
            raise ValueError("require 1 <= pre-cut <= post-writer < depth; cut=0 cannot encode context")
        if not 1 <= slots <= 64 or (max_new_tokens is not None and not 1 <= max_new_tokens <= 512):
            raise ValueError("invalid slot or generation budget")
        self.slots = slots
        self.max_context = max_context
        self.max_new_tokens = max_new_tokens
        self.transport = transport or Transport()
        self.device = model.get_input_embeddings().weight.device
        self.dtype = model.get_input_embeddings().weight.dtype
        self.width = model.get_input_embeddings().weight.shape[1]
        self.meter = Meter()
        self.max_forwards = max_forwards
        self.max_processed_tokens = max_processed_tokens
        self.wall_seconds = wall_seconds
        self.started = time.monotonic()
        self.min_free_cuda_gib = min_free_cuda_gib
        neutral = tokenizer.encode(".", add_special_tokens=False)
        if not neutral:
            raise ValueError("Tokenizer has no neutral placeholder token")
        self.placeholder_id = neutral[-1]
        eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        eos = eos if eos is not None else getattr(tokenizer, "eos_token_id", None)
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        if max_new_tokens is None and not self.eos_ids:
            raise ValueError("EOS-first generation requires a model or tokenizer EOS id")
        self.last_trace = {}
        self.last_activations = {}
        self.last_action = {}
        self.identity = {"model_type": model.config.model_type, "width": self.width,
                         "depth": self.depth, "cut": self.cut, "writer": self.writer,
                         "slots": self.slots, "dtype": str(self.dtype),
                         "transport_sha256": self.transport.artifact_sha256,
                         "chat_template_sha256": digest(str(getattr(tokenizer, "chat_template", None))),
                         "max_context": max_context, "max_new_tokens": max_new_tokens,
                         "generation_policy": "until_eos_or_context" if max_new_tokens is None else "capped",
                         "placeholder_id": self.placeholder_id, "eos_ids": sorted(self.eos_ids),
                         "route": "pre_cut_slots_to_post_writer_v0", "cache_policy": "none"}

    def render(self, content: str):
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("Native chat template required; no silent fallback")
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        markers = (prompts.INPUT_MARKER, prompts.OUTPUT_MARKER)
        for marker in markers:
            if rendered.count(marker) > 1:
                raise ValueError("Duplicate reserved slot marker")
        pieces = re.split("(" + "|".join(re.escape(m) for m in markers) + ")", rendered)
        ids, found = [], {}
        for part in pieces:
            if part in markers:
                found[part] = list(range(len(ids), len(ids)+self.slots))
                ids += [self.placeholder_id] * self.slots
            elif part:
                ids += self.tokenizer.encode(part, add_special_tokens=False)
        if not ids or len(ids) > self.max_context:
            raise ValueError(f"Context length {len(ids)} exceeds limit {self.max_context}; never silently truncate")
        return ids, found

    @staticmethod
    def _hidden(value):
        h = value[0] if isinstance(value, (tuple, list)) else value
        if not isinstance(h, torch.Tensor):
            raise TypeError("Unsupported layer output; expected tensor or tensor-first tuple")
        return h

    def _budget(self, tokens):
        if self.meter.forwards >= self.max_forwards:
            raise RuntimeError("Forward budget exhausted")
        if self.meter.processed_tokens + tokens > self.max_processed_tokens:
            raise RuntimeError("Token-compute budget exhausted")
        if time.monotonic() - self.started > self.wall_seconds:
            raise RuntimeError("Wall budget exhausted at a forward boundary")
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            if free < self.min_free_cuda_gib * 2**30:
                raise RuntimeError("CUDA free-memory floor reached; no retry")

    def forward(self, ids, *, state=None, input_positions=(), capture_positions=(), instrument=False):
        if not ids or len(ids) > self.max_context:
            raise ValueError("Invalid or oversized forward context")
        self._budget(len(ids))
        if state is not None:
            if state.slots is None or tuple(state.slots.shape) != (self.slots, self.width):
                raise ValueError("Expected matching latent slots")
            if len(input_positions) != self.slots:
                raise ValueError("Must map each input slot explicitly")
        captured, summaries, activations, handles = {}, {}, {}, []
        positions = list(capture_positions)
        for p in [*positions, *input_positions]:
            if not 0 <= p < len(ids):
                raise ValueError("slot index outside context")
        def take(h):
            return h[:, positions, :].detach().clone() if positions else h[:, -1:, :].detach().clone()
        def pre(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            if state is not None:
                h = h.clone()
                h[:, list(input_positions), :] = state.slots.to(h.device, h.dtype).unsqueeze(0)
            captured["pre_cut"] = take(h)
            if args:
                return (h, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": h}
        def post(module, args, result):
            captured["post_writer"] = take(self._hidden(result))
        def component(name, router=False):
            def hook(module, args, result):
                h = self._hidden(result)
                if h.ndim == 2 and h.shape[0] == len(ids):
                    h = h.unsqueeze(0)
                if h.ndim != 3 or h.shape[1] != len(ids):
                    raise ValueError(f"Unsupported activation shape at {name}: {tuple(h.shape)}")
                x = take(h)
                summaries[name] = tensor_summary(x)
                if router:
                    summaries[name].update(router_summary(x, getattr(self.model.config, "num_experts_per_tok", 2)))
                activations[name] = x.cpu()
            return hook
        try:
            handles.append(self.layers[self.cut].register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(self.layers[self.writer].register_forward_hook(post))
            if instrument:
                for i in sorted({self.cut, self.writer}):
                    layer = self.layers[i]
                    for name in ("self_attn", "mlp", "block_sparse_moe"):
                        module = getattr(layer, name, None)
                        if module is not None:
                            handles.append(module.register_forward_hook(component(f"layer_{i}.{name}")))
                    moe = getattr(layer, "block_sparse_moe", None)
                    if moe is not None and hasattr(moe, "gate"):
                        handles.append(moe.gate.register_forward_hook(component(f"layer_{i}.router_logits", True)))
            with torch.inference_mode():
                x = torch.tensor([ids], dtype=torch.long, device=self.device)
                out = self.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False,
                                 return_dict=True, output_hidden_states=False)
            self.meter.forwards += 1
            self.meter.processed_tokens += len(ids)
            if getattr(out, "past_key_values", None) is not None:
                raise RuntimeError("Host returned KV despite use_cache=False")
            if not torch.isfinite(out.logits).all():
                raise FloatingPointError("Nonfinite logits")
            for name, x in captured.items():
                summaries[name] = tensor_summary(x)
                activations[name] = x.cpu()
            self.last_trace = summaries if instrument else {}
            self.last_activations = activations if instrument else {}
            return out.logits, captured
        finally:
            for handle in handles:
                handle.remove()

    def encode(self, text: str) -> State:
        ids, pos = self.render(prompts.encode_prompt(text))
        _, h = self.forward(ids, capture_positions=pos[prompts.OUTPUT_MARKER], instrument=True)
        # Native input to cut: initial E does not pretend a final residual is an embedding.
        self.last_action = {"operation": "encode", "tokens": len(ids)}
        return State(slots=h["pre_cut"][0])

    def update_latent(self, state: State, op: str) -> State:
        ids, pos = self.render(prompts.update_prompt(op))
        _, h = self.forward(ids, state=state, input_positions=pos[prompts.INPUT_MARKER],
                            capture_positions=pos[prompts.OUTPUT_MARKER], instrument=True)
        self.last_action = {"operation": "latent_update", "operator": op,
                            "transport": self.transport.artifact_sha256}
        return State(slots=self.transport(h["post_writer"][0]))

    def generate(self, prompt: str, state: State | None = None) -> TextOutput:
        ids, pos = self.render(prompt)
        budget = self.max_context - len(ids)
        stop = "context_limit"
        if self.max_new_tokens is not None and self.max_new_tokens <= budget:
            budget = self.max_new_tokens
            stop = "length"
        emitted = []
        trace, activations = {}, {}
        stop_token_id = None
        for n in range(budget):
            logits, _ = self.forward(ids + emitted, state=state,
                                      input_positions=pos.get(prompts.INPUT_MARKER, ()),
                                      capture_positions=pos.get(prompts.INPUT_MARKER, ()), instrument=n == 0)
            if n == 0:
                trace, activations = self.last_trace, self.last_activations
            token = int(logits[0, -1].argmax().item())
            self.meter.generated_tokens += 1
            if token in self.eos_ids:
                stop = "eos"
                stop_token_id = token
                break
            emitted.append(token)
        self.last_trace, self.last_activations = trace, activations
        text = self.tokenizer.decode(emitted, skip_special_tokens=True).strip()
        result = TextOutput(text, stop, len(emitted), len(ids), budget, stop_token_id)
        self.last_action = {"operation": "generate", "tokens": result.token_count, **result.receipt()}
        return result

    def update_text(self, text: str, op: str) -> TextOutput:
        return self.generate(prompts.text_prompt(text, op))

    def decode(self, state: State) -> TextOutput:
        return self.generate(prompts.read_prompt(), state)

    def score_choices(self, state: State | None, question: str, labels: list[str]) -> dict:
        """Full label likelihood. None leaves the same slot placeholders uninjected."""
        prompt = prompts.task_prompt(question, text=state.text if state is not None else None)
        ids, pos = self.render(prompt)
        result = {}
        for label in labels:
            answer = self.tokenizer.encode(label, add_special_tokens=False)
            if not answer:
                raise ValueError("Empty answer tokenization")
            logits, _ = self.forward(ids+answer, state=state if state is not None and state.slots is not None else None,
                                      input_positions=pos.get(prompts.INPUT_MARKER, ()))
            lp = logits[0, len(ids)-1:len(ids)+len(answer)-1].float().log_softmax(-1)
            idx = torch.tensor(answer, device=lp.device)
            result[label] = {"logprob": lp.gather(1, idx[:, None]).sum().item(), "tokens": len(answer)}
        return result

    def native_parity(self) -> dict:
        ids, _ = self.render("Return the letter A.")
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            ref = self.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True).logits
        hooked, _ = self.forward(ids, capture_positions=[len(ids)-1], instrument=True)
        return {"exact": torch.equal(ref, hooked), "max_abs": (ref-hooked).abs().max().item(),
                "scope": "unmodified_native_vs_observation_hooks_no_injection"}
