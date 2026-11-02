import itertools
import json
from pathlib import Path

import pytest

from selffold.continuity_data import (FRAMES, OPERATIONS, evaluate_condition, examples,
    load_groups, public_groups, question_cases, split_audit, write_public)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/continuity_20261103"


def test_two_stage_material_counts_scope_and_counterbalancing():
    groups = list(public_groups())
    audit = split_audit(groups)
    assert (audit["train"]["groups"], audit["dev"]["groups"]) == (96, 24)
    assert (audit["train"]["variants"], audit["dev"]["variants"]) == (1152, 288)
    assert set(audit["train"]["construction_families"]).isdisjoint(audit["dev"]["construction_families"])
    assert set(audit["train"]["flags"]).isdisjoint(audit["dev"]["flags"])
    assert set(audit["train"]["actions"]).isdisjoint(audit["dev"]["actions"])
    assert audit["overlap"]["base_truth_signatures"] == ["0001", "0010"]
    assert (audit["train"]["explicit_case_noop_groups"], audit["dev"]["explicit_case_noop_groups"]) == (24, 6)
    all_ids = set()
    for group in groups:
        for pole in ("plus", "minus"):
            variants = group["poles"][pole]["variants"]
            assert {(v["frame"], v["order"]) for v in variants} == set(itertools.product(FRAMES, (0, 1)))
            for v in variants:
                assert v["id"] not in all_ids
                all_ids.add(v["id"])
                assert all(group["id"] not in v[s] for s in ("source", "stage1", "stage2"))
                assert (v["source"] == v["stage1"]) == v["no_op_stage1"]
                assert all(op not in v["source"] for op in OPERATIONS.values())
    assert len(all_ids) == 1440


def test_ast_and_independently_authored_ledgers_agree_with_both_formal_stages():
    for group in public_groups():
        x, y = group["flags"]
        for world, bit in zip(itertools.product((False, True), repeat=2), group["truth_bits"], strict=True):
            assert evaluate_condition(group["rule_ast"], dict(zip((x, y), world))) == bool(bit)
        plus, minus = group["poles"]["plus"], group["poles"]["minus"]
        assert len(plus["truth_ledger"]) == len(minus["truth_ledger"]) == 4
        for a, b in zip(plus["truth_ledger"], minus["truth_ledger"], strict=True):
            assert a["world"] == b["world"] and a["choice"] != b["choice"]
        for details in (plus, minus):
            for row in details["truth_ledger"]:
                predicted = group["actions"][0 if evaluate_condition(details["stage2_condition_ast"], row["world"]) else 1]
                assert predicted == row["choice"]
            for variant in details["variants"]:
                subject = {"self": "I choose", "other": "Agent A chooses", "impersonal": "the policy selects"}[variant["frame"]]
                for row in details["truth_ledger"]:
                    conditions = " and ".join(f"{k} is {'on' if v else 'off'}" for k, v in row["world"].items())
                    assert f"When {conditions}, {subject} {row['choice']}." in variant["stage1"]
                assert variant["stage1"] != variant["stage2"]


def test_recipient_questions_balance_labels_and_reuse_truth_under_wording_order_and_polarity():
    for group in public_groups():
        cases = question_cases(group)
        assert len(cases) == 32
        assert sum(c["gold"] == "A" for c in cases) == 16
        assert len({(c["pole"], c["world_index"], c["wording"], c["order"]) for c in cases}) == 32
        for case in cases:
            assert case["gold"] == case["answer"]
            gold_action = group["poles"][case["pole"]]["truth_ledger"][case["world_index"]]["choice"]
            options = group["actions"][::1 if case["order"] == 0 else -1]
            assert options[0 if case["gold"] == "A" else 1] == gold_action
            assert f"A = {options[0]}. B = {options[1]}." in case["question"]
            assert group["site"] not in case["question"] and group["id"] not in case["question"]


def test_loader_checks_bytes_denies_qualification_and_preserves_stage_pair_ids(tmp_path):
    folder = tmp_path / "public"
    write_public(folder)
    for split, count in (("train", 1152), ("dev", 288)):
        sources = examples(folder, split)
        assert len(sources) == count
        for stage in ("stage1", "stage2"):
            targets = examples(folder, split, stage)
            assert [r["id"] for r in sources] == [r["id"] for r in targets]
            assert all(a["source"] == b["original_source"] and a[stage] == b["source"]
                       and b["source"] == b["text"] and b["variant_id"] == b["id"]
                       for a, b in zip(sources, targets, strict=True))
    with pytest.raises(ValueError, match="public train and dev"):
        load_groups(folder, "qualification")
    (folder / "dev.groups.jsonl").write_text((folder / "dev.groups.jsonl").read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        load_groups(folder, "dev")


def test_committed_public_material_regenerates_without_qualification(tmp_path):
    generated = tmp_path / "generated"
    write_public(generated)
    assert {p.name for p in DATA.iterdir()} == {"train.groups.jsonl", "dev.groups.jsonl", "manifest.json"}
    for p in generated.iterdir():
        assert p.read_bytes() == (DATA / p.name).read_bytes()
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["qualification"] == {"status": "new_two_stage_panel_not_authored", "content_in_checkout": False}
    assert manifest["human_transition_review"] == "pending"
