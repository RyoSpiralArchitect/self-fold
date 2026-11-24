"""Deterministic CPU checks. No checkpoint, model generation or optimizer step."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from selffold.paired_row_objectives import ARMS, batch_objective, branch_objective


@pytest.fixture(scope="module")
def scripts():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        import audit_paired_writer_rows as audit
        import fit_paired_row_objectives as experiment

        yield audit, experiment


def targets():
    return torch.tensor([[[1, 0, 1, 0, 0], [1, 0, 1, 0, 1]], [[0, 1, 0, 1, 0], [0, 1, 0, 1, 1]]])


def test_batch_matches_old_mean_value_and_every_logit_gradient(scripts):
    _, exp = scripts
    gold = targets().reshape(4, 5)
    ids = torch.arange(4 * 12).reshape(4, 12) % 13
    seq_lens, prompt_lens = [11, 12, 9, 10], [4, 5, 3, 4]
    positions = [[n - 1 + k for k in range(5)] for n in prompt_lens]
    batch = [
        (
            ids[i, :length].tolist(),
            prompt_lens[i],
            [(p, [j, j + 6], int(gold[i, j])) for j, p in enumerate(positions[i])],
        )
        for i, length in enumerate(seq_lens)
    ]
    logits = torch.arange(4 * 12 * 13, dtype=torch.float32).reshape(4, 12, 13).sin().requires_grad_()
    active = torch.zeros_like(ids, dtype=torch.bool)
    for i, (n, length) in enumerate(zip(prompt_lens, seq_lens)):
        active[i, n - 1 : length - 1] = True
    ce = F.cross_entropy(logits[active], ids.roll(-1, dims=1)[active])
    choices = torch.stack(
        [logits[i, p, [j, j + 6]] for i, pos in enumerate(positions) for j, p in enumerate(pos)]
    )
    binary = F.cross_entropy(choices, gold.flatten())
    margins = (choices[4::5, 0] - choices[4::5, 1]) * (1 - 2 * gold[:, 4])
    old = ce + binary + exp.prior.role_loss(margins, False, 2)
    base, parts = batch_objective(logits, ids, batch, "mean", 2)
    torch.testing.assert_close(base, old, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(base, logits)[0], torch.autograd.grad(old, logits)[0]
    )
    for arm, (c, j) in ARMS.items():
        loss, _ = batch_objective(logits, ids, batch, arm, 2)
        torch.testing.assert_close(
            loss - base, c * parts["row_ce"] + j * parts["row_js"], atol=1e-6, rtol=1e-6
        )


def test_equal_wrong_donors_need_grounding_and_opposed_role_gradients():
    gold = targets()
    one_hot = F.one_hot(gold, 2).double()
    correct, wrong = 6 * one_hot, -6 * one_hot
    wrong.requires_grad_()
    p = branch_objective(wrong, gold, "combined", 2)
    assert p["row_ce"] > 6
    assert branch_objective(correct, gold, "combined", 2)["row_ce"] < 0.003
    assert abs(p["row_js"].item()) < 1e-12
    (js_grad,) = torch.autograd.grad(p["row_js"], wrong, retain_graph=True)
    torch.testing.assert_close(js_grad, torch.zeros_like(js_grad), rtol=0, atol=1e-12)
    (row_grad,) = torch.autograd.grad(p["auxiliary"], wrong, retain_graph=True)
    assert (row_grad[:, :, :4].gather(-1, gold[:, :, :4, None]) < 0).all()
    assert torch.count_nonzero(row_grad[:, :, 4]) == 0
    (role_grad,) = torch.autograd.grad(p["role"], wrong)
    assert torch.count_nonzero(role_grad[:, :, :4]) == 0
    assert (role_grad[:, 0, 4, 0] < 0).all() and (role_grad[:, 1, 4, 0] > 0).all()


def test_js_aligned_symmetric_and_backpropagates_only_at_disagreements():
    gold = targets()
    logits = torch.zeros(2, 2, 5, 2, dtype=torch.float64)
    logits[0, 0, 2] = torch.tensor([4.0, -3.0])
    logits[0, 1, 2] = torch.tensor([-3.0, 4.0])
    logits.requires_grad_()
    p = branch_objective(logits, gold, "agreement", 2)
    assert 0 < p["row_js"] < 0.6932 / 8
    (grad,) = torch.autograd.grad(p["auxiliary"], logits)
    assert grad[0, 0, 2, 0] > 0 and grad[0, 1, 2, 0] < 0
    assert (grad.abs().sum(-1) > 1e-12).sum() == 2
    for name, value in branch_objective(logits.flip(1), gold.flip(1), "agreement", 2).items():
        torch.testing.assert_close(value, p[name])
    changed = logits.detach().clone()
    changed[:, :, 4] = 1000
    q = branch_objective(changed, gold, "agreement", 2)
    for k in ("row_ce", "row_js", "auxiliary"):
        torch.testing.assert_close(q[k], p[k])


def test_reject_misaligned_pairs():
    logits = torch.zeros(2, 2, 5, 2)
    for field in (0, 4):
        gold = targets()
        gold[0, 1, field] ^= 1
        with pytest.raises(ValueError, match="Twin"):
            branch_objective(logits, gold, "mean", 2)
    with pytest.raises(ValueError, match="token order"):
        choices = [(k, [0, 1], 0) for k in range(5)]
        wrong = [(0, [1, 0], 0), *choices[1:]]
        batch_objective(
            torch.zeros(2, 7, 2),
            torch.zeros(2, 7, dtype=torch.long),
            [([0] * 7, 1, choices), ([0] * 7, 1, wrong)],
            "agreement",
            2,
        )


def test_audit_preserves_missing_and_wrong_role_direction(scripts):
    a, _ = scripts
    assert [a.relation(*v, 3) for v in ((3, 3), (2, 2), (2, 3), (3, None), (None, None))] == list(
        a.RELATIONS
    )
    assert [a.role_status(v, [2, 3]) for v in ([2, 3], [3, 2], [2, 2], [0, 1], [None, None])] == list(
        a.ROLE_STATUS
    )


def test_archive_keys_and_joint_counts(scripts):
    a, _ = scripts
    records = json.loads(a.ARCHIVE.read_bytes())
    panel = a.audit(list(reversed(records)))["fresh"]
    values = [
        (
            r["joint"],
            r["final_with_row_error"],
            r["rotation_histogram"]["joint"][4],
            r["any_shared_wrong"],
            r["same_wrong_table"],
        )
        for r in (panel[k] for k in a.OLD_ARMS)
    ]
    assert values == [
        (1, 1, 0, 12, 11),
        (64, 23, 4, 7, 4),
        (50, 31, 1, 9, 6),
        (96, 0, 24, 0, 0),
        (96, 0, 24, 0, 0),
    ]
    with pytest.raises(ValueError, match="Incomplete"):
        a.audit(records[1:])
    with pytest.raises(ValueError, match="Duplicate"):
        a.audit([*records[:-1], records[0]])
    with pytest.raises(ValueError, match="scoring flags"):
        a.audit([{**records[0], "role": not records[0]["role"]}, *records[1:]])


def test_future_scorer_replays_saved_outputs_and_schedule(scripts):
    a, exp = scripts
    cfg, actor, names, train, test, schedules = exp.checked_inputs()
    twins = exp.prior.twin_indices(train)
    for steps in schedules.values():
        for start in range(0, 256, 64):
            exposed = [
                tuple(step[i : i + 2]) for step in steps[start : start + 64] for i in range(0, 8, 2)
            ]
            assert sorted(exposed) == sorted(twins)
    # Reuse archived text as fixtures of all four new slots, never as new fit results.
    aliases = {
        "parent": "parent",
        "mean": "individual",
        "row_ce": "paired",
        "agreement": "individual",
        "combined": "paired",
        "L": "L",
        "oracle": "oracle",
    }
    raw = {(r["arm"], r["id"]): r for r in exp.prior.read_lines(exp.ARCHIVE / "reports.jsonl")}
    records = {
        (arm, q["id"]): raw[old, q["id"]]
        for q in test
        for arm, old in aliases.items()
        if arm not in ("L", "oracle")
    }
    saved = {(r["arm"], r["id"]): r for r in json.loads(a.ARCHIVE.read_bytes())}
    outcomes = exp.evaluate(cfg, actor, names, test, records, exp.prior.banks_from(exp.ARCHIVE, cfg))
    for r in outcomes:
        old = aliases[r["arm"]]
        assert {**r, "arm": old} == saved[old, r["id"]]
    panel = a.audit(outcomes, a.NEXT_ARMS, details=True)["fresh"]
    result = exp.contrast(panel, "combined", "mean")
    assert result["changes"]["joint"] == {"gained": 8, "lost": 22}
    assert result["passed"] is False
