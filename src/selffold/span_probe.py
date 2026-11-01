"""Partition a lexical donor's downstream residuals at one fixed decoder cut."""
from dataclasses import dataclass

import torch

from . import prompts
from .records import digest
from .replay_probe import _capture, _difference, _logprobs, _replace_span


BLOCKS = ("instruction", "question", "options", "closing_prefix", "score")
CONTEXT_WINDOWS = ("closing", "last4", "score")


@dataclass
class Layout:
    ids: list[int]
    slots: list[int]
    windows: dict
    suffix: str
    offsets: list


def make_layout(host, group, plus_label):
    first, second = group["options"] if plus_label == "A" else group["options"][::-1]
    q = f"{group['question']}\nA = {first}. B = {second}. Use the supplied description."
    content = prompts.task_prompt(q)
    prompt, markers = host.render(content)
    slots = markers[prompts.INPUT_MARKER]
    start, end = slots[-1] + 1, len(prompt)
    rendered = host.tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                                   tokenize=False, add_generation_prompt=True)
    _, suffix = rendered.split(prompts.INPUT_MARKER)
    tokenized = host.tokenizer(suffix, add_special_tokens=False, return_offsets_mapping=True)
    offsets = tokenized["offset_mapping"]
    if list(tokenized["input_ids"]) != prompt[start:] or len(offsets) != end - start:
        raise ValueError("Offset tokenization disagrees with the actual injected prompt")

    def boundary(character):
        # Include a token crossing a character boundary in the block to its right.
        return start + next(i for i, (_, right) in enumerate(offsets) if right > character)

    q_char = suffix.index(q)
    question = boundary(q_char)
    choices = boundary(q_char + len(group["question"]) + 1)
    closing = boundary(q_char + q.index("Use the supplied description."))
    score = end - 1
    points = (start, question, choices, closing, score, end)
    if not all(a < b for a, b in zip(points, points[1:])) or closing - choices < 8 or question - start < 8:
        raise ValueError("Prompt does not support the fixed partition and eight-position controls")
    if end - closing < 4 or slots != list(range(slots[0], start)):
        raise ValueError("Invalid closing window or slot geometry")
    windows = dict(zip(BLOCKS, zip(points, points[1:]), strict=True))
    windows.update(closing=(closing, end), pre_options=(start, choices), first8=(start, start + 8),
                   options8=(closing - 8, closing), last8=(end - 8, end), last4=(end - 4, end))
    return Layout(prompt + [host.placeholder_id], slots, windows, suffix, offsets)


@dataclass
class Forward:
    ids: list[int]
    logits: torch.Tensor
    residual: torch.Tensor


@dataclass
class Context:
    layout: Layout
    neutral: Forward
    lexical: dict


def _forward(host, ids):
    logits, residual = _capture(host, ids, list(range(len(ids))))
    return Forward(ids, logits, residual)


def _context(host, layout, words, *, blind=False):
    ids = layout.ids.copy()
    if blind:
        a = layout.windows["question"][0]
        b = layout.windows["closing"][0]
        ids[a:b] = [host.placeholder_id] * (b-a)
    donors = {}
    for pole, token_ids in words.items():
        donor_ids = ids.copy()
        donor_ids[layout.slots[0]:layout.slots[-1]+1] = token_ids
        donors[pole] = _forward(host, donor_ids)
    return Context(layout, _forward(host, ids), donors)


def _metrics(logits, native, labels):
    return {"scores": _logprobs(logits, labels), "max_abs_vocab_from_native": _difference(logits, native)}


def _reference(host, target, donor, labels):
    start, end = target.layout.slots[-1]+1, len(target.layout.ids)-1
    native, neutral = donor.logits, target.neutral
    suffix = _replace_span(host, neutral.ids, donor.residual, start, end)
    complete = _replace_span(host, neutral.ids, donor.residual, target.layout.slots[0], end)
    self_replay = _replace_span(host, neutral.ids, neutral.residual, start, end)
    return {
        "native": _metrics(native, native, labels), "neutral": _metrics(neutral.logits, native, labels),
        "suffix": _metrics(suffix, native, labels), "slots_and_suffix": _metrics(complete, native, labels),
        "neutral_replay_max_abs": _difference(self_replay, neutral.logits),
    }


def _localize(host, target, donor, labels):
    regions = {}
    start, end = target.layout.slots[-1]+1, len(target.layout.ids)-1
    for name, (lo, hi) in target.layout.windows.items():
        # For leave-out, start with the successful downstream donor and put back just this neutral block.
        omitted = donor.residual.clone()
        omitted[:, lo:hi] = target.neutral.residual[:, lo:hi]
        only = _replace_span(host, target.neutral.ids, donor.residual, lo, hi)
        leave_out = _replace_span(host, target.neutral.ids, omitted, start, end)
        native_knockout = _replace_span(host, donor.ids, target.neutral.residual, lo, hi)
        regions[name] = {k: _metrics(v, donor.logits, labels) for k, v in
                         (("only", only), ("omit", leave_out), ("native_knockout", native_knockout))}
    return regions


def _context_swap(host, target, source, pole, labels):
    donor = source.lexical[pole]
    output = {"source_lexical_ids_sha256": digest(donor.ids), "source_neutral_ids_sha256": digest(source.neutral.ids),
              "source_native_scores": _logprobs(donor.logits, labels), "windows": {}}
    if len(target.neutral.ids) != len(source.neutral.ids):
        output["status"] = "different_forward_lengths"
        return output
    output["status"] = "matched_length"
    end_prefix = target.layout.windows["question"][0]
    output["pre_question_max_abs"] = _difference(target.lexical[pole].residual[:, :end_prefix],
                                                 donor.residual[:, :end_prefix])
    base = target.neutral.residual
    # Preserve the recipient's prompt background while isolating the source's lexical-minus-null change.
    rebased = (base.float() + (donor.residual.float() - source.neutral.residual.float())).to(base.dtype)
    for name in CONTEXT_WINDOWS:
        lo, hi = target.layout.windows[name]
        if source.layout.windows[name] != (lo, hi) or donor.ids[lo:hi] != target.neutral.ids[lo:hi]:
            output["windows"][name] = {"status": "different_positions_or_token_ids"}
            continue
        raw = _replace_span(host, target.neutral.ids, donor.residual, lo, hi)
        delta = _replace_span(host, target.neutral.ids, rebased, lo, hi)
        output["windows"][name] = {"status": "measured", "raw": _logprobs(raw, labels),
                                   "delta": _logprobs(delta, labels),
                                   "raw_delta_max_abs_vocab": _difference(raw, delta)}
    return output


def run_span_probe(host, store, groups):
    labels = {}
    for key in ("A", "B"):
        ids = host.tokenizer.encode(key, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError("Single-token labels required")
        labels[key] = ids[0]
    store.write("span_settings.json", {"labels": labels, "dummy_id": host.placeholder_id, "partition": BLOCKS,
                                       "context_windows": CONTEXT_WINDOWS, "indexing": "zero_based_half_open"})
    for group in groups:
        words = {}
        for pole, phrase in zip(("plus", "minus"), group["options"], strict=True):
            ids = host.tokenizer.encode(phrase, add_special_tokens=False)
            if not 1 <= len(ids) <= host.slots:
                raise ValueError("Lexical source does not fit eight slots")
            words[pole] = ids + [host.placeholder_id] * (host.slots-len(ids))
        layouts = {k: make_layout(host, group, k) for k in labels}
        natural = {k: _context(host, layout, words) for k, layout in layouts.items()}
        for order, other in (("A", "B"), ("B", "A")):
            target = natural[order]
            layout = layouts[order]
            blind = _context(host, layout, words, blind=True)
            store.append("prompts.jsonl", {"group": group["id"], "plus_label": order, "question": group["question"],
                                          "options_plus_minus": group["options"], "padded_lexical_ids": words,
                                          "recipient_ids": layout.ids, "slot_positions": layout.slots,
                                          "windows": layout.windows, "rendered_suffix": layout.suffix,
                                          "suffix_offsets": layout.offsets,
                                          "masked_neutral_ids": blind.neutral.ids})
            for pole in words:
                donor = target.lexical[pole]
                reference = _reference(host, target, donor, labels)
                row = {"group": group["id"], "pole": pole, "plus_label": order,
                       "recipient_ids_sha256": digest(layout.ids), "reference": reference,
                       "regions": _localize(host, target, donor, labels),
                       "contexts": {name: _context_swap(host, target, source, pole, labels)
                                    for name, source in (("same", target), ("reversed", natural[other]),
                                                         ("masked", blind))}}
                store.append("interventions.jsonl", row)
                if reference["neutral_replay_max_abs"] != 0 or reference["slots_and_suffix"]["max_abs_vocab_from_native"] != 0:
                    raise ValueError("Full-geometry replay did not reconstruct its reference")
