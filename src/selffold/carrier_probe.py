"""Description-only carriers, frozen before any question is presented."""
from __future__ import annotations
import torch
from . import prompts
from .records import State
from .rng import isolated_rng

ROUTES = ("workspace", "native_tail")
CONTROLS = ("carrier", "reverse", "norm_only", "random", "unrelated", "zero", "uninjected", "text")
QUESTIONS = (None, "Which alternative matches the current description?")


def capture_description(host, text, route, *, forward_length=256):
    """The signature deliberately takes no task, options, label or operator target."""
    if route == "workspace":
        content = prompts.encode_prompt(text)
        ids, slots = host.render(content)
        positions = slots[prompts.OUTPUT_MARKER]
        method = "workspace_marker"
    elif route == "native_tail":
        content = "Read this current description.\nCurrent description:\n" + text
        ids, _ = host.render(content)
        rendered = host.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        if rendered.count(text) != 1:
            raise ValueError("Source description must occur exactly once")
        start, end = rendered.index(text), rendered.index(text) + len(text)
        if getattr(host.tokenizer, "is_fast", False):
            native = host.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
            if native["input_ids"] != ids:
                raise ValueError("Offset tokenization differs from the actual native forward")
            source = [i for i, (a, b) in enumerate(native["offset_mapping"]) if b > start and a < end]
            method = "native_offsets"
        else:
            before = host.tokenizer.encode(rendered[:start], add_special_tokens=False)
            through = host.tokenizer.encode(rendered[:end], add_special_tokens=False)
            if ids[:len(before)] != before or ids[:len(through)] != through:
                raise ValueError("Non-fast tokenizer has no exact source boundaries")
            source = list(range(len(before), len(through)))
            method = "exact_token_prefixes"
        if len(source) < host.slots:
            raise ValueError("Description is too short for the fixed native tail")
        positions = source[-host.slots:]
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError("Native trailing positions are not contiguous")
    else:
        raise ValueError("Unknown carrier route")
    if len(ids) > forward_length:
        raise ValueError("Description exceeds the fixed donor length; no truncation")
    padded = ids + [host.placeholder_id] * (forward_length - len(ids))
    _, captured = host.forward(padded, capture_positions=positions)
    state = State(slots=captured["pre_cut"][0])
    return state, {"route": route, "source_text": text, "native_ids": ids,
                   "forward_length": forward_length, "padding_id": host.placeholder_id,
                   "capture_positions": positions, "capture_token_ids": [ids[p] for p in positions],
                   "position_method": method, "state_sha256": state.sha()}


def question_orders(group):
    for qindex, wording in enumerate(QUESTIONS):
        for plus_label, first, second in (("A", *group["options"]), ("B", *group["options"][::-1])):
            yield qindex, plus_label, f"{wording or group['question']}\nA = {first}. B = {second}. Use the supplied description."


def neutral_replay(host, question):
    ids, pos = host.render(prompts.task_prompt(question))
    # Match score_choices' full-label forward, including its causally later label token.
    label = host.tokenizer.encode("A", add_special_tokens=False)
    other = host.tokenizer.encode("B", add_special_tokens=False)
    if len(label) != 1 or len(other) != 1:
        raise ValueError("This diagnostic requires one-token balanced labels")
    ids = ids + label
    slots = pos[prompts.INPUT_MARKER]
    reference, captures = host.forward(ids, capture_positions=slots)
    state = State(slots=captures["pre_cut"][0])
    replay, _ = host.forward(ids, state=state, input_positions=slots)
    result = {"exact": torch.equal(reference, replay),
              "max_abs_logit_difference": (reference - replay).abs().max().item(),
              "forward_length": len(ids), "slot_positions": slots}
    if not result["exact"]:
        raise ValueError(f"Neutral self-replay failed: {result}")
    return result


def run_carrier_probe(host, store, groups, *, routes=ROUTES, forward_length=256):
    if len(groups) < 2:
        raise ValueError("An unrelated-state control requires multiple groups")
    # Capture every donor before even constructing a question.
    donors = {}
    for route in routes:
        for group in groups:
            for pole in ("plus", "minus"):
                for phrasing, text in enumerate(group[pole]):
                    key = (route, group["id"], pole, phrasing)
                    state, metadata = capture_description(host, text, route, forward_length=forward_length)
                    donors[key] = state
                    case = f"{route}/{group['id']}-{pole}-p{phrasing}"
                    store.save_state("states/" + case, state)
                    store.append("donors.jsonl", {"case": case, "group": group["id"], "pole": pole,
                                                "phrasing": phrasing, **metadata})
    for group in groups:
        for qindex, plus_label, question in question_orders(group):
            store.append("neutral.jsonl", {"group": group["id"], "question_variant": qindex,
                                          "plus_label": plus_label, **neutral_replay(host, question)})
    shape = (host.slots, host.width)
    generator = torch.Generator().manual_seed(4711)
    directions = torch.nn.functional.normalize(torch.randn(shape, generator=generator), dim=1)
    random_direction = torch.nn.functional.normalize(
        torch.randn(shape, generator=torch.Generator().manual_seed(9292)), dim=1)
    zero = State(slots=torch.zeros(shape, device=host.device, dtype=host.dtype))
    for route in routes:
        # One donor-independent random control per route, with the route's mean row norm.
        mean_norm = torch.stack([s.slots.float().cpu().norm(dim=1).mean()
                                 for key, s in donors.items() if key[0] == route]).mean()
        random_state = State(slots=(random_direction * mean_norm).to(host.device, host.dtype))
        for index, group in enumerate(groups):
            next_group = groups[(index + 1) % len(groups)]
            for pole in ("plus", "minus"):
                for phrasing, text in enumerate(group[pole]):
                    state = donors[(route, group["id"], pole, phrasing)]
                    original = state.sha()
                    other_phrasing = phrasing % len(next_group[pole])
                    unrelated = donors[(route, next_group["id"], pole, other_phrasing)]
                    norms = state.slots.float().cpu().norm(dim=1, keepdim=True)
                    controls = {"carrier": state, "reverse": State(slots=state.slots.flip(0)),
                                "norm_only": State(slots=(directions * norms).to(state.slots)),
                                "random": random_state, "unrelated": unrelated, "zero": zero,
                                "uninjected": None, "text": State(text=text)}
                    case = f"{route}/{group['id']}-{pole}-p{phrasing}"
                    with isolated_rng():
                        output = host.decode(state.clone())
                    store.append("generations.jsonl", {"case": case, "route": route, **output.receipt()})
                    for control, supplied in controls.items():
                        fingerprint = supplied.sha() if supplied is not None else None
                        for qindex, plus_label, question in question_orders(group):
                            with isolated_rng():
                                scores = host.score_choices(supplied.clone() if supplied is not None else None,
                                                            question, ["A", "B"])
                            if supplied is not None and supplied.sha() != fingerprint:
                                raise RuntimeError("Question readout mutated the saved state")
                            delta = scores["A"]["logprob"] - scores["B"]["logprob"]
                            prediction = None if delta == 0 else ("A" if delta > 0 else "B")
                            gold = plus_label if pole == "plus" else ("B" if plus_label == "A" else "A")
                            store.append("scores.jsonl", {"case": case, "route": route, "group": group["id"],
                                "pole": pole, "phrasing": phrasing, "control": control,
                                "state_sha256": fingerprint, "unrelated_group": next_group["id"] if control == "unrelated" else None,
                                "question_variant": qindex, "plus_label": plus_label, "question": question,
                                "scores": scores, "plus_margin": delta if plus_label == "A" else -delta,
                                "prediction": prediction, "gold": gold, "correct": prediction == gold,
                                "tie": delta == 0})
                    if state.sha() != original:
                        raise RuntimeError("Carrier changed after observation")
