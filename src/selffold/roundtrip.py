"""RoundTrip: report the realized before/after pair, not the requested mask.

Separates state updating from message writing: eval_step first produces the
actual output; the writer then sees the REALIZED (prior, instruction, output)
triple — including pointer errors and collapsed states — and its report feeds
the next invocation. This is the actual-state half the S-WRITE plan's
current-table side was missing.
"""
from __future__ import annotations

import torch

from .lm_writer import assignment_words, decode_state_slot

WORLD = ["off/off", "off/on", "on/off", "on/on"]
RT_STYLES = ("rt_directed", "rt_natural")


def pattern_of(w):
    """Realized carried pattern: argmax member per row packed as a 4-bit mask."""
    return (w.argmax(-1) * torch.tensor([1, 2, 4, 8], device=w.device)).sum(-1)


def render_table(options, pole, pattern):
    """The member-action row list for a realized pattern value."""
    actions = assignment_words(options, pole, int(pattern))
    return "; ".join(f"{w} -> {a}" for w, a in zip(WORLD, actions))


def rt_prompt(style, options, pole, prior_pattern, mask, out_pattern):
    """Prompt showing the actual realized transition (prior -> output)."""
    if style not in RT_STYLES:
        raise ValueError(f"Unknown roundtrip style: {style}")
    base = (
        "You are the self-report module of a small state machine.\n"
        f"The table previously read: {render_table(options, pole, prior_pattern)}.\n"
        f"An instruction was applied with these assignments: "
        f"{render_table(options, pole, mask)}.\n"
        f"The table now reads: {render_table(options, pole, out_pattern)}.\n"
    )
    if style == "rt_directed":
        return base + (
            "Question: which action did this instruction assign to the "
            "off/off row? Answer with the action name only."
        )
    return base + (
        "Task: write one short log sentence describing what changed."
    )


def rt_decode(text, options):
    """Pole-free slot decode over the realized report."""
    return decode_state_slot(text, options)


def rt_key(source, prior, mask, out):
    """Report-cache key for one realized transition."""
    return (int(source), int(prior), int(mask), int(out))


def feasible_outs(prior, cur):
    """Every out pattern a (prior, cur) transition can realize.

    Each world row copies the weights of the row the pointer picked for its
    order, so an off row lands on the dominant member of pick0 and an on row on
    pick1. Enumerating both picks covers every out a controller (trained or
    exploring) can produce from this prior — the report-cache coverage bound.
    """
    pb = [(prior >> r) & 1 for r in range(4)]
    outs = set()
    for p0 in range(4):
        for p1 in range(4):
            outs.add(sum(pb[p1 if (cur >> i) & 1 else p0] << i for i in range(4)))
    return sorted(outs)


TEXT_FEAT_DIM = 32
TEXT_FEAT_VERSION = "hashbag_v1"


def text_features(text):
    """Frozen, deterministic report featurizer (no learned semantics).

    24 hashed-token features (md5-signed word counts) plus 8 structural
    flags: row-mention density, arrow presence, negation, past/now markers,
    and a length bucket. Declared, pole-free side information — the
    recipient learns whatever these features predict.
    """
    import hashlib
    import re
    feats = torch.zeros(TEXT_FEAT_DIM)
    toks = re.findall(r"[^\W_]+", text.casefold())
    for t in toks:
        h = hashlib.md5(t.encode()).digest()
        feats[h[0] % 24] += 1.0 if h[1] & 1 else -1.0
    row = len(re.findall(r"(?:off|on)\s*[/\-]\s*(?:off|on)", text, re.I))
    feats[24] = min(row, 3) / 3
    feats[25] = 1.0 if re.search(r"->", text) else 0.0
    feats[26] = 1.0 if re.search(r"\bno longer\b|\bnot\b|\binstead of\b", text, re.I) else 0.0
    feats[27] = 1.0 if re.search(r"\bprevious|\bwas\b|\bbefore\b|\bold\b", text, re.I) else 0.0
    feats[28] = 1.0 if re.search(r"\bnow\b|\bcurrent|\bnew\b", text, re.I) else 0.0
    feats[29] = min(len(toks), 64) / 64
    feats[30] = 1.0 if re.search(r"\boff\s*[/\-]\s*off\b", text, re.I) else 0.0
    feats[31] = 1.0 if re.search(r"\bfirst\b|\bsecond\b|\blatter\b|\bformer\b", text, re.I) else 0.0
    return feats


def realize_next(weights, mask_bits):
    """The ideal update: each world row takes the member named by its mask bit.

    This is the gold transition; the controller's actual output can diverge
    from it when a wrong message steers the row pick.
    """
    on = (mask_bits[..., None] >> torch.arange(4, device=weights.device)) & 1
    on = on.bool()
    out = torch.where(on[..., None], weights[..., 1:2], weights[..., 0:1])
    return out
