"""Matched topic headers and two-field preference records for a frozen readout."""

from collections import Counter

from . import prompts
from .polarity_readout import source_cases

HEADERS = ("bare", "topic", "placebo", "rotated")
POSITIONS = ("before", "after")


def question_groups(groups, fixture):
    labels = [fixture["topic_cues"][g["id"]] for g in groups]
    result = []
    for i, group in enumerate(groups):
        words = (
            "",
            labels[i],
            fixture["placebo"],
            labels[(i + fixture["rotated_offset"]) % len(labels)],
        )
        questions = [
            f"{word}\n{stem}" if word else stem
            for stem in fixture["questions"]
            for word in words
        ]
        result.append(group | {"fresh_questions": questions})
    return result


def preference_records(groups, fixture):
    for family, verdicts in fixture["verdicts"].items():
        for position in POSITIONS:

            def clause(verdict):
                first, second = (
                    (verdict, "{}") if position == "before" else ("{}", verdict)
                )
                return f"My preference record: [ {first} ] [ {second} ]."

            forms = {
                f"{family}_{position}": dict(
                    zip(("chosen", "rejected"), map(clause, verdicts), strict=True)
                )
            }
            for c in source_cases(groups, forms, fixture["familiar_pairs"]):
                yield c | {"family": family, "position": position}


def header_geometry(rendered):
    """Require a single token substitution or the deletion of the heading and newline."""
    bare, topic, placebo, rotated = rendered
    if not len(topic) == len(placebo) == len(rotated) == len(bare) + 2:
        raise ValueError("Topic headers must add exactly two native tokens")
    changed = [i for i, (a, b) in enumerate(zip(topic, placebo, strict=True)) if a != b]
    if len(changed) != 1:
        raise ValueError("Header word must occupy exactly one token")
    i = changed[0]
    if (
        len({topic[i], placebo[i], rotated[i]}) != 3
        or topic[:i] + topic[i + 1 :] != rotated[:i] + rotated[i + 1 :]
        or topic[:i] + topic[i + 2 :] != bare
    ):
        raise ValueError("Header changed the remaining tokenized question")
    return {
        "header_index": i,
        "bare_tokens": len(bare),
        "named_tokens": len(topic),
        "header_token_ids": [v[i] for v in (topic, placebo, rotated)],
        "newline_id": topic[i + 1],
    }


def source_geometry(host, before, after):
    left, m = host.render(prompts.encode_prompt(before))
    right, n = host.render(prompts.encode_prompt(after))
    if left == right or Counter(left) != Counter(right) or m != n:
        raise ValueError(
            "Field swap must preserve token multiset and workspace positions"
        )
    return {"tokens": len(left), "slots": m[prompts.OUTPUT_MARKER]}
