"""One-forward geometry controls for the development probe. No state enters recurrence."""
from . import prompts
from .records import State, digest


def _capture(host, ids, positions):
    logits, captured = host.forward(ids, capture_positions=positions)
    return logits[0, -2].float().clone(), captured["pre_cut"]


def _slots(host, ids, positions, values):
    logits, _ = host.forward(ids, state=State(slots=values[0]), input_positions=positions)
    return logits[0, -2].float().clone()


def _replace_span(host, ids, values, start, end):
    """Isolated positive control on a single recipient forward through the same cut."""
    def replace(module, args, kwargs):
        h = (args[0] if args else kwargs["hidden_states"]).clone()
        h[:, start:end] = values[:, start:end]
        if args:
            return (h, *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": h}

    handle = host.layers[host.cut].register_forward_pre_hook(replace, with_kwargs=True)
    try:
        logits, _ = host.forward(ids)
        return logits[0, -2].float().clone()
    finally:
        handle.remove()


def _difference(first, second):
    return (first.float() - second.float()).abs().max().item()


def _logprobs(logits, labels):
    lp = logits.log_softmax(-1)
    return {label: lp[idx].item() for label, idx in labels.items()}


def run_replay_probe(host, store, groups):
    labels = {}
    for name in ("A", "B"):
        ids = host.tokenizer.encode(name, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError("This control requires single-token labels")
        labels[name] = ids[0]
    store.write("replay_settings.json", {"label_ids": labels, "trailing_dummy": host.placeholder_id,
                                        "scoring_position": "last prompt token before the dummy"})
    for group in groups:
        plus, minus = group["options"]
        for plus_label, first, second in (("A", plus, minus), ("B", minus, plus)):
            question = f"{group['question']}\nA = {first}. B = {second}. Use the supplied description."
            prompt, markers = host.render(prompts.task_prompt(question))
            positions = markers[prompts.INPUT_MARKER]
            first_slot, after_slots = positions[0], positions[-1] + 1
            assert positions == list(range(first_slot, after_slots))
            ids = prompt + [host.placeholder_id]
            all_positions = list(range(len(ids)))
            neutral_logits, full = _capture(host, ids, all_positions)
            _, short = _capture(host, ids[:after_slots], positions)
            opaque_ids = ids[:after_slots] + [host.placeholder_id] * (len(ids) - after_slots)
            _, same_length = _capture(host, opaque_ids, positions)
            neutral_scores = _logprobs(neutral_logits, labels)
            row = dict(group=group["id"], plus_label=plus_label, question=question,
                       token_ids=ids, slot_positions=positions, native_scores=neutral_scores)
            for key, captured in (("short", short), ("same_length", same_length)):
                replay = _slots(host, ids, positions, captured)
                row[key] = {"scores": _logprobs(replay, labels),
                            "max_abs_slot_difference": _difference(full[:, positions], captured),
                            "max_abs_logit_difference": _difference(neutral_logits, replay)}
            store.append("neutral.jsonl", row)

            for pole, words in (("plus", plus), ("minus", minus)):
                tokens = host.tokenizer.encode(words, add_special_tokens=False)
                if not 1 <= len(tokens) <= len(positions):
                    raise ValueError("Phrase does not fit the frozen slot positions")
                padded = tokens + [host.placeholder_id] * (len(positions) - len(tokens))
                donor_ids = ids[:first_slot] + padded + ids[after_slots:]
                native_logits, donor = _capture(host, donor_ids, all_positions)
                opaque_ids = donor_ids[:after_slots] + [host.placeholder_id] * (len(ids) - after_slots)
                _, lexical_slots = _capture(host, opaque_ids, positions)
                routes = {
                    "native_words": native_logits,
                    "slots_only": _slots(host, ids, positions, lexical_slots),
                    "suffix_only": _replace_span(host, ids, donor, after_slots, len(prompt)),
                    "slots_and_suffix": _replace_span(host, ids, donor, first_slot, len(prompt)),
                }
                store.append("interventions.jsonl", {
                    "group": group["id"], "pole": pole, "plus_label": plus_label, "words": words,
                    "padded_source_token_ids": padded, "recipient_ids_sha256": digest(ids),
                    "slot_positions": positions, "suffix_span_half_open": [after_slots, len(prompt)],
                    "native_vs_opaque_slot_max_abs": _difference(donor[:, positions], lexical_slots),
                    "opaque_slots_sha256": State(slots=lexical_slots[0]).sha(),
                    "scores": {r: _logprobs(v, labels) for r, v in routes.items()},
                    "max_abs_logit_difference_from_native": {r: _difference(v, native_logits) for r, v in routes.items()},
                })
