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


def _distinctive(option, other):
    """Word sets that identify `option` against its partner."""
    mine = {w.strip(".,'s") for w in option.lower().split()}
    theirs = {w.strip(".,'s") for w in other.lower().split()}
    return mine - theirs


def decode_bit(text, options, pole):
    """Extract the role bit from a self-report.

    bit=1 iff the report names member1's action. A member counts as named when
    the text contains all of its words that distinguish it from its partner
    (exact phrase or close paraphrase), so semantic restatements count.
    Returns (bit, missing): missing=1 when neither member is identifiable.
    """
    w0, w1 = options[pole], options[1 - pole]
    d0, d1 = _distinctive(w0, w1), _distinctive(w1, w0)
    low = text.lower()
    h0 = w0 in text or (d0 and all(t in low for t in d0))
    h1 = w1 in text or (d1 and all(t in low for t in d1))
    if not (h0 or h1):
        return 0, 1
    if h0 and h1:
        i0 = low.find(next(t for t in d0 if t in low))
        i1 = low.find(next(t for t in d1 if t in low))
        return (1 if i1 < i0 else 0), 0
    return (1 if h1 else 0), 0


def mentions_offoff(text):
    return "off/off" in text or "off-off" in text.lower()
