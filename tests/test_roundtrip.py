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
    now = p.split("The table now reads: ")[1].split("\n")[0]
    assert now == ("off/off -> ridge path; off/on -> river path; "
                   "on/off -> river path; on/on -> river path.")


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


def test_feasible_outs_covers_all_picks():
    from selffold.roundtrip import feasible_outs
    # prior 0b1000 = [m0,m0,m0,m1]: off rows share one pick, on rows the other.
    assert feasible_outs(8, 14) == [0, 1, 14, 15]
    # A collapsed prior can only produce uniform tables.
    assert feasible_outs(15, 6) == [15]
    assert feasible_outs(0, 6) == [0]


def test_rt_report_requests_cover_feasible_bound():
    from selffold.lm_writer_reports import rt_report_requests
    groups = [{"options": OPTS}]
    reqs = rt_report_requests("rt_natural", [14], [(0, 0)], groups)
    # 14 mixed priors x 4 feasible outs + 2 uniform priors x 1
    assert len(reqs) == 14 * 4 + 2
    assert all({"source", "prior", "mask", "out", "prompt"} <= set(r) for r in reqs)
    key = next(r for r in reqs if (r["prior"], r["mask"], r["out"]) == (8, 14, 1))
    assert "The table now reads" in key["prompt"]


def _closed_keys():
    return [(0, f, 0, p, o) for f in (0, 1) for p in (0, 1) for o in (0, 1)]


def test_rollout_rt_uses_decoded_bit_as_next_message():
    from selffold.message_boundary import BoundaryMemory
    from selffold.mixture_boundary import rollout_rt
    torch.manual_seed(0)
    keys = _closed_keys()
    members = torch.randn(len(keys), 2, 16)
    # Perfect writer: the report bit equals the realized out's off/off bit.
    rt_bits = torch.zeros(len(keys), 16, 16, 16, dtype=torch.long)
    for out in range(16):
        rt_bits[:, :, :, out] = out & 1
    actor = BoundaryMemory(7, 1)
    rolled = rollout_rt(actor, "hard", members, keys, [6, 9], rt_bits)
    assert {"second", "fourth", "written", "prior_mismatch", "depths"} <= set(rolled)
    num = rolled["depths"].shape[1]
    assert rolled["depths"].shape == (len(keys), num, 4, 4)
    assert rolled["written"].shape[0] == len(keys)


def test_rollout_rt_rejects_uncovered_lookup():
    import pytest
    from selffold.message_boundary import BoundaryMemory
    from selffold.mixture_boundary import rollout_rt
    keys = _closed_keys()
    members = torch.randn(len(keys), 2, 16)
    rt_bits = torch.full((len(keys), 16, 16, 16), -1, dtype=torch.long)
    with pytest.raises(ValueError):
        rollout_rt(BoundaryMemory(7, 1), "hard", members, keys, [6, 9], rt_bits)


def test_rollout_rt_uses_out_axis_not_requested_mask():
    """Swapping the realized-out coordinate for the requested mask must
    change the selected report — catches lookups indexed by instruction."""
    from selffold.message_boundary import BoundaryMemory
    from selffold.mixture_boundary import rollout_rt
    keys = _closed_keys()
    members = torch.randn(len(keys), 2, 16)
    base = torch.zeros(len(keys), 16, 16, 16, dtype=torch.long)
    base[:, :, :, :] = torch.arange(16) & 1          # report = out & 1
    by_out = rollout_rt(BoundaryMemory(7, 1), "hard", members, keys, [6, 9], base.clone())
    req = torch.zeros(len(keys), 16, 16, 16, dtype=torch.long)
    req[:, :, :, :] = (torch.arange(16)[None, :, None, None] & 1)  # report = mask & 1
    by_req = rollout_rt(BoundaryMemory(7, 1), "hard", members, keys, [6, 9], req)
    assert not (by_out["written"] == by_req["written"]).all()
    assert not (by_out["depths"] == by_req["depths"]).all()


def test_save_report_matrices_4d_instruction_axis_and_counts(tmp_path):
    """4D panels select the instruction axis, keep all outs, and report
    unique generated records separately from broadcast cells."""
    import numpy as np
    from selffold.lm_writer_reports import save_report_matrices
    n = 2
    bits = {"rt_x": torch.zeros(n, 16, 16, 16, dtype=torch.long)}
    gen = {"rt_x": np.zeros((n, 16, 16, 16), bool)}
    miss = {"rt_x": np.zeros((n, 16, 16, 16))}
    off = {"rt_x": np.zeros((n, 16, 16, 16))}
    # Only instruction 6, prior 8: all 4 feasible outs generated;
    # the two uniform outs (0, 15) are missing-reports.
    for o in (0, 6, 9, 15):
        gen["rt_x"][:, 8, 6, o] = True
    miss["rt_x"][:, 8, 6, 0] = miss["rt_x"][:, 8, 6, 15] = 1.0
    summary = save_report_matrices(tmp_path, [6, 9], bits, miss, off, gen,
                                   record_counts={"rt_x": 4})
    assert summary["report_counts"]["rt_x"] == 4          # unique records
    assert summary["broadcast_cells"]["rt_x"] == 8        # 2 sources x 4 outs
    assert summary["missing"]["rt_x"]["6"] == 0.5         # 2 of 4 outs missing
    assert summary["missing"]["rt_x"]["9"] is None        # ungenerated mask
    import json
    written = json.loads((tmp_path / "lm_matrix.json").read_text())
    assert written["missing"]["rt_x"]["6"] == 0.5
