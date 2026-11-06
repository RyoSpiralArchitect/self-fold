"""Frozen-LM writer for the W-RITE experiment (docs/SELF_REFERENCE_READING.md).

The writer reads the post-update table plus the applied instruction and emits a
short self-report. A fixed decoder extracts the role bit: does the report name
the member the off/off row was just assigned? Directed reports echo the
assignment (channel-capacity ceiling); natural log entries measure whether the
model spontaneously writes the self-referential coordinate.
"""

from __future__ import annotations

STYLES = ("directed", "natural")


def assignment_words(options, pole, mask):
    """The action name each world selects under `mask` for this source."""
    return [options[pole ^ b] for b in ((mask >> i) & 1 for i in range(4))]


def prompt_for(style, options, pole, mask):
    world = ["off/off", "off/on", "on/off", "on/on"]
    actions = assignment_words(options, pole, mask)
    assignments = "; ".join(f"{w} -> {a}" for w, a in zip(world, actions))
    text = (
        "You are the self-report module of a small state machine.\n"
        f"An instruction was applied with these assignments: {assignments}.\n"
    )
    if style == "directed":
        return text + (
            "Question: which action did this instruction assign to the "
            "off/off row? Answer with the action name only."
        )
    return text + (
        "Task: write one short log sentence describing what this instruction "
        "changed."
    )


def decode_bit(text, options, pole):
    """Extract the role bit from a self-report.

    Returns (bit, missing): bit=1 iff the report names member1's action
    (off/off -> switched). missing=1 when no member action is named.
    """
    w0, w1 = options[pole], options[1 - pole]
    i0, i1 = text.find(w0), text.find(w1)
    if i0 < 0 and i1 < 0:
        return 0, 1
    return (1 if 0 <= i1 and (i0 < 0 or i1 < i0) else 0), 0


def mentions_offoff(text):
    return "off/off" in text or "off-off" in text.lower()
