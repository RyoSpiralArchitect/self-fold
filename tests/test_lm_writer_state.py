"""Regression tests for the row-binding decode_state parser (PR #34 review)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from selffold.lm_writer import decode_state, decode_bit

OPTS = ["river path", "ridge path"]


def dec(text, pole=0):
    return decode_state(text, OPTS, pole)


def test_changed_from_to():
    assert dec("off/off changed from river path to ridge path.") == (1, 0)


def test_arrow_rows_neighbors():
    assert dec("off/on -> river path; off/off -> ridge path.") == (1, 0)
    assert dec("off/off -> river path; off/on -> ridge path.") == (0, 0)


def test_negated_old_value_pronoun():
    assert dec("off/off no longer selects river path; it now selects ridge path.") == (1, 0)


def test_action_before_coordinate():
    assert dec("ridge path for off/off.") == (1, 0)


def test_simple_binding():
    assert dec("off/off -> river path.") == (0, 0)
    assert dec("off/off -> ridge path.") == (1, 0)


def test_other_rows_only_is_unanchored():
    # Mentions of other rows never bind off/off; falls back to global decode.
    bit, miss = dec("off/on changed from river path to ridge path.")
    assert bit == 0  # global fallback: equal scores, first mention wins


def test_no_member_missing():
    assert dec("The state machine's rules were updated.") == (0, 1)


def test_wrong_binding_stays_wrong():
    # Report binds the WRONG member to off/off — parser must not rescue it.
    assert dec("off/on -> ridge path; off/off -> river path.") == (0, 0)


def test_multi_row_single_clause():
    assert dec("off/off -> ridge path, off/on -> river path.") == (1, 0)


def test_temporal_precedence():
    assert dec("off/off previously selected ridge path. "
               "off/off now selects river path.") == (0, 0)


def test_negation_resolves_complement():
    # Binary state: 'no longer selects X' asserts the other member.
    assert dec("off/off no longer selects river path.") == (1, 0)


def test_pronoun_does_not_steal_other_row():
    # 'it' refers to off/on (its subject): no bound vote for off/off. The
    # unanchored fallback still decodes — unanchored use is tracked via
    # offoff_rate, not the missing flag.
    assert dec("off/on -> ridge path. it now selects river path.") == (1, 0)
