"""Train only a small Gemma report writer; score free and template-supported reports."""
from __future__ import annotations

from itertools import chain
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .structured_report import FIELDS, report_text


class WriterAdapter(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=torch.float32)
        self.up = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + (self.up(self.down(x.float())) * self.scale).to(y.dtype)
        return y


def attach_writer(host, cfg):
    torch.manual_seed(cfg["seed"])
    host.model.requires_grad_(False)
    adapters = {}
    for i in range(host.depth - cfg["last_layers"], host.depth):
        block = host.layers[i].self_attn
        for name in ("q_proj", "v_proj"):
            adapter = WriterAdapter(getattr(block, name), cfg["rank"], cfg["alpha"])
            setattr(block, name, adapter)
            adapters[i, name] = adapter
    return adapters


def detach_writer(host, adapters):
    for (i, name), adapter in adapters.items():
        setattr(host.layers[i].self_attn, name, adapter.base)


def tokenizer_choice(tokenizer, prior, alternatives):
    """Causal prefix ending immediately before the two full action strings diverge."""
    choices = [tokenizer.encode(prior + name, add_special_tokens=False) for name in alternatives]
    for i, (left, right) in enumerate(zip(*choices)):
        if left != right:
            return choices[0][:i], [left, right]
    raise ValueError("One action tokenization is a prefix of the other")


def field_prefix(completed):
    done = [f"{label} = {value}" for label, value in zip(FIELDS, completed)]
    return "\n".join([*done, FIELDS[len(completed)] + " = "])


def training_record(host, prompt, values, alternatives):
    source, _ = host.render(prompt)
    target = host.tokenizer.encode(report_text(values), add_special_tokens=False)
    end_token = host.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if end_token not in host.eos_ids:
        raise ValueError("Gemma turn terminator is not registered as EOS")
    choices = []
    for j, value in enumerate(values):
        prefix, tokens = tokenizer_choice(host.tokenizer, field_prefix(values[:j]), alternatives)
        answer = alternatives.index(value)
        if target[:len(prefix) + 1] != prefix + [tokens[answer]]:
            raise ValueError("Teacher-forced and causal field tokens differ")
        choices.append((len(source) + len(prefix) - 1, tokens, answer))
    return source + target + [end_token], len(source), choices


def pad_batch(sequences, pad, device, *, left):
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        start = width - len(seq) if left else 0
        ids[i, start:start+len(seq)] = torch.tensor(seq, device=device)
        mask[i, start:start+len(seq)] = 1
    return ids, mask


def fit_writer(host, adapters, samples, cfg, store):
    params = [p for p in host.model.parameters() if p.requires_grad]
    if {id(p) for p in params} != {id(p) for a in adapters.values() for p in (a.down.weight, a.up.weight)}:
        raise ValueError("Unexpected trainable parameter outside D")
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=0.)
    rng = np.random.default_rng(cfg["seed"])
    step = 0
    for epoch in range(cfg["epochs"]):
        inds = rng.permutation(len(samples)).tolist()
        for start in range(0, len(inds), cfg["batch"]):
            indices = inds[start:start+cfg["batch"]]
            batch = [samples[i] for i in indices]
            ids, mask = pad_batch([r[0] for r in batch], host.tokenizer.pad_token_id, host.device, left=False)
            logits = host.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            active = torch.zeros_like(mask, dtype=torch.bool)
            for j, (seq, src_len, _) in enumerate(batch):
                active[j, src_len - 1:len(seq) - 1] = True
            ce = F.cross_entropy(logits[active].float(), ids.roll(-1, dims=1)[active])
            branches = list(chain.from_iterable(r[2] for r in batch))
            positions = torch.tensor([r[0] for r in branches], device=host.device)
            which = torch.arange(len(batch), device=host.device).repeat_interleave(len(FIELDS))
            tokens = torch.tensor([r[1] for r in branches], device=host.device)
            gold = torch.tensor([r[2] for r in branches], device=host.device)
            field_logits = logits[which[:, None], positions[:, None], tokens].float()
            field_loss = F.cross_entropy(field_logits, gold)
            loss = ce + cfg["choice_loss_weight"] * field_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            opt.step()
            step += 1
            store.append("training.jsonl", dict(step=step, epoch=epoch+1, examples=indices,
                                                ce=float(ce.detach()), choice_ce=float(field_loss.detach())))
        print(f"Writer epoch {epoch+1}/{cfg['epochs']} closed, step {step}", flush=True)
    if step != cfg["steps"]:
        raise ValueError("Training schedule incomplete")
    return sum(p.numel() for p in params)


@torch.inference_mode()
def free_reports(host, prompts, max_tokens):
    seqs = [host.render(p)[0] for p in prompts]
    ids, attn = pad_batch(seqs, host.tokenizer.pad_token_id, host.device, left=True)
    if ids.shape[1] + max_tokens > host.max_context:
        raise ValueError("Generation would exceed the sealed context")
    outputs = host.model.generate(input_ids=ids, attention_mask=attn, do_sample=False, num_beams=1,
                                  max_new_tokens=max_tokens, use_cache=True,
                                  eos_token_id=sorted(host.eos_ids), pad_token_id=host.tokenizer.pad_token_id)
    records = []
    for tail in outputs[:, ids.shape[1]:].tolist():
        stop = next((i for i, token in enumerate(tail) if token in host.eos_ids), None)
        ids_used = tail if stop is None else tail[:stop+1]
        records.append(dict(text=host.tokenizer.decode(ids_used, skip_special_tokens=True),
                            token_ids=ids_used, stop="token_cap" if stop is None else "native_eos"))
    return records


@torch.inference_mode()
def choice_reports(host, prompts, alternatives):
    """The same five forced labels for both writers. Earlier model choices stay in context."""
    starts = [host.render(p)[0] for p in prompts]
    selected = [[] for _ in prompts]
    margins = [[] for _ in prompts]
    for _ in FIELDS:
        nodes = [tokenizer_choice(host.tokenizer, field_prefix(prior), opts)
                 for prior, opts in zip(selected, alternatives, strict=True)]
        seqs = [s + prefix for s, (prefix, _) in zip(starts, nodes, strict=True)]
        ids, mask = pad_batch(seqs, host.tokenizer.pad_token_id, host.device, left=True)
        if ids.shape[1] > host.max_context:
            raise ValueError("Choice context exceeds the registered limit")
        pos = (mask.cumsum(1) - 1).clamp_min(0)
        logits = host.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                            logits_to_keep=1, use_cache=False).logits[:, -1]
        tokens = torch.tensor([t for _, t in nodes], device=host.device)
        candidates = logits.gather(1, tokens).float().cpu()
        for j, idx in enumerate(candidates.argmax(-1).tolist()):
            selected[j].append(alternatives[j][idx])
            margins[j].append(candidates[j].tolist())
    return [dict(text=report_text(values), branch_logits=values_ab, stop="five_fields")
            for values, values_ab in zip(selected, margins, strict=True)]
