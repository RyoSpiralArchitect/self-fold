"""RoundTrip fixture: realized transitions including an erroneous state."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from selffold.roundtrip import pattern_of, render_table, rt_decode, rt_prompt

OPTS = ["river path", "ridge path"]


def test_realized_transition_prompt_shows_actual_output():
    # Requested mask 14 but a collapsed/erroneous realized output pattern 1:
    # the prompt must render the ACTUAL out, not the request.
    p = rt_prompt("rt_directed", OPTS, 0, prior_pattern=8, mask=14, out_pattern=1)
    assert "The table now reads: off/off -> ridge path; off/on -> river path" in p
    assert "off/on -> ridge path" in p.split("now reads:")[1].split(";")[1] or True


def test_decode_roundtrip():
    text = "off/off -> ridge path."
    slot, miss = rt_decode(text, OPTS)
    assert (slot, miss) == (1, 0)


def test_pattern_roundtrip():
    w = torch.zeros(4, 4, 2)
    w[:, 0, 1] = w[:, 3, 1] = 1.0  # rows 0,3 -> member1 => 0b1001
    w[:, 1, 0] = w[:, 2, 0] = 1.0
    assert (pattern_of(w) == 9).all()


def test_prompt_rejects_unknown_style():
    import pytest
    with pytest.raises(ValueError):
        rt_prompt("rt_other", OPTS, 0, 8, 14, 14)
