"""Frozen-LM writer for the W-RITE experiment (docs/SELF_REFERENCE_READING.md).

The writer reads the post-update table plus the applied instruction and emits a
short self-report. A fixed decoder extracts the role bit: does the report name
the member the off/off row was just assigned? Directed reports echo the
assignment (channel-capacity ceiling); natural log entries measure whether the
model spontaneously writes the self-referential coordinate.
"""

from __future__ import annotations

STYLES = ("directed", "natural", "guided")


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
    if style == "guided":
        return text + (
            "Task: write one short log entry naming each world whose "
            "assignment changed and its new action, in the form "
            "'world -> action'. If nothing changed for a world, omit it."
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


def _score(text, option, other):
    """Distinctive-token hits for `option` (exact phrase counts fully)."""
    if option in text:
        return 1 << 20, text.find(option)
    low = text.lower()
    hits = [t for t in _distinctive(option, other) if t in low]
    if not hits:
        return 0, 1 << 30
    return len(hits), min(low.find(t) for t in hits)


def decode_bit(text, options, pole):
    """Extract the role bit from a self-report.

    Each member is scored by distinctive-keyword hits; the higher score wins,
    first-position breaking ties. Single-word answers ("summarize") count via
    their distinctive verb. Returns (bit, missing); missing only when neither
    member scores at all.
    """
    w0, w1 = options[pole], options[1 - pole]
    s0, i0 = _score(text, w0, w1)
    s1, i1 = _score(text, w1, w0)
    if s0 == 0 and s1 == 0:
        return 0, 1
    return (1 if (s1, -i1) > (s0, -i0) else 0), 0


def mentions_offoff(text):
    return "off/off" in text or "off-off" in text.lower()
