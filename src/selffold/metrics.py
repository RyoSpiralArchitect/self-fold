"""Descriptive diagnostics, not automatic semantic or consciousness scores."""
import re
from collections import Counter
import math
import torch
from .records import State


def transition(a: State, b: State) -> dict:
    out = {"exact_repeat": a.sha() == b.sha()}
    if a.slots is not None and b.slots is not None:
        x, y = a.slots.float(), b.slots.float()
        if x.shape != y.shape:
            raise ValueError("cannot compare differently shaped states")
        out.update({"rms": y.square().mean().sqrt().item(),
                    "rms_delta": (y-x).square().mean().sqrt().item(),
                    "cosine": torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()})
    elif a.text is not None and b.text is not None:
        x, y = set(re.findall(r"\w+", a.text.lower())), set(re.findall(r"\w+", b.text.lower()))
        out["lexical_jaccard_proxy_not_semantic"] = len(x & y) / max(1, len(x | y))
    else:
        raise ValueError("transition types differ")
    return out


def tensor_summary(t: torch.Tensor) -> dict:
    x = t.detach().float()
    if not torch.isfinite(x).all():
        raise FloatingPointError("nonfinite activation")
    return {"shape": list(t.shape), "rms": x.square().mean().sqrt().item(),
            "mean": x.mean().item(), "max_abs": x.abs().max().item()}


def router_summary(logits: torch.Tensor, top_k: int) -> dict:
    p = logits.detach().float().softmax(-1)
    k = min(top_k, p.shape[-1])
    selected = p.topk(k, dim=-1).indices.flatten().cpu().tolist()
    return {"entropy": (-(p * p.clamp_min(1e-12).log()).sum(-1)).mean().item(),
            "topk_logit_candidates_not_dispatch_counts": dict(Counter(selected)),
            "experts": p.shape[-1], "top_k": k}
