from dataclasses import replace

import pytest

from selffold.structured_report import Midpoint, read_fields, reencode, report_text, writer_prompt, paired_change


NAMES = ["sender", "date", "river", "ridge"]


def test_roundtrip_does_not_carry_the_gold_role_around_text():
    z = Midpoint((0, 1, 0, 1), 0)
    opposite = replace(z, bit=1)
    assert z.rows == opposite.rows
    for state in (z, opposite):
        text = report_text([NAMES[i] for i in state.fields()])
        assert reencode(read_fields(text, NAMES)) == state
    assert writer_prompt(z, 2, NAMES) != writer_prompt(opposite, 2, NAMES)


def test_roundtrip_fails_closed_on_missing_or_conflicting_fields():
    text = report_text(["sender", "date", "sender", "date", "sender"])
    assert reencode(read_fields(text + "\ndefault = date", NAMES)) is None
    assert reencode(read_fields(text.replace("off/on = date", ""), NAMES)) is None
    assert reencode(read_fields(text.replace("default = sender", "default = ridge"), NAMES)) is None
    assert reencode(read_fields(text.replace("default =", "**Default action**:"), NAMES)) is not None


def test_paired_changes_keep_recovered_and_lost_cases():
    assert paired_change({1: False, 2: True, 3: False}, {1: True, 2: False, 3: True}) == dict(n=3, recovered=2, lost=1)
    with pytest.raises(ValueError):
        paired_change({1: False}, {2: True})


def test_frozen_second_edit_uses_the_reported_role_on_identical_tables():
    import numpy as np
    from pathlib import Path
    from selffold.structured_report import controller, final_choices, first_update

    actor = controller(Path(__file__).resolve().parents[1], 17)
    labels = np.array([10, 20])
    letters = (np.arange(16)[:, None] & 1) ^ np.arange(2)
    native = dict(labels=labels, top=np.broadcast_to(labels[letters], (8, 16, 2)),
                  ab=np.broadcast_to(np.stack((1-letters, letters), -1), (8, 16, 2, 2)))
    twins = [first_update(actor, 2, 6, default) for default in (0, 1)]
    assert twins[0].rows == twins[1].rows and twins[0].bit != twins[1].bit
    for default, state in enumerate(twins):
        assert final_choices(actor, state, 2, default, [3, 10], native)["correct"] == 16
        assert final_choices(actor, twins[1-default], 2, default, [3, 10], native)["correct"] == 0
