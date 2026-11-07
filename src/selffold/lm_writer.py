"""Frozen-LM reports of supplied ideal assignments (W-RITE development test).

The prompt is rendered from source options/polarity and a requested mask, not
from the controller's realized recurrent tensor. Decoding also receives the
source's option-to-role mapping. This tests that supplied table-to-report-to-bit
interface, not an autonomous D(Z) or a bound on natural-language capacity.
Historical reports retain their original decoder and results. New runs use the
version below and must not overwrite or relabel those archived experiments.
"""

from __future__ import annotations

from numbers import Integral
import re

STYLES = ("directed", "natural", "guided",
          "state_directed", "state_natural", "state_guided", "state_prior_natural")
DECODER_VERSION = "distinctive_tokens_v3"
STATE_DECODER_VERSION = "rowbound_v1"


def _validate_options(options, pole):
    if (len(options) != 2 or any(not isinstance(x, str) or not x.strip() for x in options)
            or options[0].casefold() == options[1].casefold()
            or not isinstance(pole, Integral) or pole not in (0, 1)):
        raise ValueError("Two distinct nonempty actions and binary source polarity required")


def assignment_words(options, pole, mask):
    """The action name each world selects under `mask` for this source."""
    _validate_options(options, pole)
    if not isinstance(mask, Integral) or not 0 <= mask < 16:
        raise ValueError("Expected a four-world integer mask")
    return [options[pole ^ b] for b in ((mask >> i) & 1 for i in range(4))]


def prompt_for(style, options, pole, mask):
    if style not in STYLES:
        raise ValueError(f"Unknown report style: {style}")
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
    if style == "natural":
        return text + (
            "Task: write one short log sentence describing what this instruction "
            "changed."
        )
    # S-WRITE: the writer also sees the realized post-update table, and for the
    # prior arm the pre-change table (referent for "what changed").
    table = f"The table now reads: {assignments}."
    prior = "The table previously read: " + "; ".join(
        f"{w} -> {a}" for w, a in zip(world, assignment_words(options, pole, 0))) + "."
    if style == "state_directed":
        return text + table + (
            "\nQuestion: which action did this instruction assign to the "
            "off/off row? Answer with the action name only."
        )
    if style == "state_guided":
        return text + table + (
            "\nTask: write one short log entry naming each world whose "
            "assignment changed and its new action, in the form "
            "'world -> action'. If nothing changed for a world, omit it."
        )
    if style == "state_prior_natural":
        return text + prior + "\n" + table + (
            "\nTask: write one short log sentence describing what changed."
        )
    return text + table + (
        "\nTask: write one short log sentence describing what this instruction "
        "changed."
    )


def _tokens(text):
    # Do not use strip("s") or substring hits: "east" is not a mention in "least".
    matches = re.finditer(r"[^\W_]+(?:['’][^\W_]+)*", text.casefold())
    return [(m.group().replace("’", "'"), m.start()) for m in matches]


def _score(text, option, other):
    """Token-boundary lexical heuristic; not entailment or a row-aware parser."""
    words = _tokens(text)
    mine = [w for w, _ in _tokens(option)]
    theirs = {w for w, _ in _tokens(other)}
    for i in range(len(words) - len(mine) + 1):
        if mine and [w for w, _ in words[i:i + len(mine)]] == mine:
            return 1 << 20, words[i][1]
    distinctive = set(mine) - theirs
    hits = [(w, pos) for w, pos in words if w in distinctive]
    return (len({w for w, _ in hits}), min(pos for _, pos in hits)) if hits else (0, 1 << 30)


def decode_bit(text, options, pole):
    """Extract the role bit from a self-report.

    Each member is scored by distinctive-keyword hits; the higher score wins,
    first-position breaking ties. Single-word answers ("summarize") count via
    their distinctive verb. Returns (bit, missing); missing only when neither
    member scores at all.
    """
    _validate_options(options, pole)
    w0, w1 = options[pole], options[1 - pole]
    s0, i0 = _score(text, w0, w1)
    s1, i1 = _score(text, w1, w0)
    if s0 == 0 and s1 == 0:
        return 0, 1
    return (1 if (s1, -i1) > (s0, -i0) else 0), 0


_ROW = r"\boff\s*[/\-]\s*off\b"
_TO = re.compile(r"\bto\b", re.IGNORECASE)
_NEG = re.compile(r"\bno longer\b|\bnot\b|\binstead of\b|\breplaced\b", re.IGNORECASE)


def _clauses(text):
    return [c for c in re.split(r"[.;!\n]+", text) if c.strip()]


def _bound_segment(clause):
    """The span describing off/off's CURRENT value inside an anchored clause.

    Handles 'X -> Y', 'changed from X to Y', 'no longer selects X' and
    'X for/was assigned to off/off' orderings; returns (segment, excluded).
    """
    if "->" in clause:
        tail = clause.split("->", 1)[1]
        return tail, None
    m = re.search(r"\bfrom\b.*?\bto\b", clause, re.IGNORECASE)
    if m:
        return clause[re.search(r"\bto\b", clause[m.start():], re.IGNORECASE).end() + m.start():], None
    if _NEG.search(clause):
        # The named value after a negator is the OLD value; exclude it.
        tail = clause[_NEG.search(clause).end():]
        seg = clause[: _NEG.search(clause).start()]
        return seg or tail, tail
    # Action-before-coordinate ("ridge path for off/off") needs the whole clause.
    return clause, None


def decode_state(text, options, pole):
    """Row-binding off/off decode for S-WRITE reports.

    Collects clauses that mention the off/off coordinate (plus pronoun 'it'
    continuations), decodes the value bound to it per clause with a majority
    vote, then falls back to the unanchored global decode. Returns (bit,
    missing); missing only when no member can be bound at all.
    """
    _validate_options(options, pole)
    votes = []
    anchored = False
    for clause in _clauses(text):
        is_row = re.search(_ROW, clause, re.IGNORECASE) is not None
        is_it = anchored and re.search(r"^\s*it\b", clause, re.IGNORECASE) is not None
        if not (is_row or is_it):
            continue
        anchored = anchored or is_row
        seg, excluded = _bound_segment(clause)
        b, miss = decode_bit(seg, options, pole)
        if miss and is_row:
            b, miss = decode_bit(clause, options, pole)
        if miss:
            continue
        if excluded is not None:
            eb, emiss = decode_bit(excluded, options, pole)
            if not emiss and eb == b:
                continue  # only the negated old value was found
        votes.append(b)
    if votes:
        return (1 if sum(votes) * 2 >= len(votes) else 0), 0
    return decode_bit(text, options, pole)


def mentions_offoff(text):
    return re.search(r"\boff\s*[/\-]\s*off\b", text, re.IGNORECASE) is not None
