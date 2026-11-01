"""Scientific design invariants, not model-outcome tests."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1] / "provenance/separator_factorization_20261101"
SPEC = importlib.util.spec_from_file_location("separator_plan", HERE / "verify.py")
plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan)


@pytest.fixture
def config():
    return plan.design()


@pytest.fixture
def layout(config):
    return plan.compile_layout("mistral", "prospective", "calendar", "period", "AB", "A", config=config)


def test_native_ids_and_every_planned_cell_have_compatible_geometry(config):
    layouts = plan.compile_all(config)
    receipt = plan.validate_grids(layouts, config)
    assert receipt["layouts"] == 528
    assert receipt["donor_order_rows_per_display_panel_cell"] == 24
    assert receipt["max_forward_length"] <= 384
    assert all(x["score_index"] == x["dummy_index"] - 1 for x in layouts if x["cell"]["block"] != "bridge")
    assert any(x["score_index"] != len(x["recipient_ids"]) - 2 for x in layouts)


def test_word_label_mapping_and_display_order_are_distinct(config):
    for display in ("AB", "BA"):
        for mapping in ("A", "B"):
            p = plan.compile_layout("gemma", "original", "archive", "period", display, mapping, config=config)
            assert p["displayed_labels"] == list(display)
            first_phrase = p["recipient_ids"][slice(*p["phrase_spans"][0])]
            rows, _ = plan.archived("gemma", "original")
            native = next(x for x in rows if x["group"] == "archive" and x["plus_label"] == mapping)
            assert first_phrase == native["recipient_ids"][slice(*native["windows"][f"{display[0]}_words"])]


def test_misaligned_or_mismatched_residuals_are_rejected(layout):
    shifted = deepcopy(layout)
    shifted["windows"]["second_punct"] = [x + 1 for x in shifted["windows"]["second_punct"]]
    with pytest.raises(ValueError, match="anchor positions"):
        plan.compatible(layout, shifted, "second_punct")
    changed = deepcopy(layout)
    changed["recipient_ids"][changed["windows"]["second_punct"][0]] += 1
    with pytest.raises(ValueError, match="anchor IDs"):
        plan.compatible(layout, changed, "second_punct")
    assert plan.compatible(layout, changed, "second_punct", punctuation_cross=True) == "intentional_punctuation_id_cross"
    with pytest.raises(ValueError, match="punctuation-ID cross"):
        plan.compatible(layout, changed, "first_punct", punctuation_cross=True)
    changed["score_index"] += 1
    with pytest.raises(ValueError, match="score indices"):
        plan.compatible(layout, changed, "second_punct", punctuation_cross=True)


def test_distance_grid_has_equal_length_and_isoresidual_prefix(config):
    grid = [plan.compile_layout("gemma", "prospective", "photo", "period", "BA", "B", u, v, config=config)
            for u in (0, 16) for v in (0, 16)]
    assert len({len(p["recipient_ids"]) for p in grid}) == 1
    assert grid[1]["score_index"] == grid[2]["score_index"]
    for left, right in (grid[:2], grid[2:]):
        end = left["windows"]["receiver_late"][1]
        assert left["recipient_ids"][:end] == right["recipient_ids"][:end]
        assert right["score_index"] - left["score_index"] == 16
    ranks = plan.design_ranks()
    assert ranks["source_punctuation_recipient_punctuation_display_mapping"]["rank"] == 16
    assert ranks["intercept_anchor_distance_score"]["rank"] == 3
    assert ranks["intercept_anchor_distance"]["rank"] == 3


def test_complete_path_blocker_closes_relays_but_direct_only_blocker_leaks(layout):
    n = len(layout["recipient_ids"])
    protected, release = plan.mask_spec(layout, "release_late")
    early = layout["windows"]["receiver_early"][0]
    late = layout["windows"]["receiver_late"][0]
    flags = plan.propagate_sources(n, layout["word_cells"], 3,
                                  lambda q, k: plan.edge_allowed(q, k, protected, release))
    assert flags[early] == 0 and flags[late] == 3
    leaked = plan.propagate_sources(n, layout["word_cells"], 3,
                                   lambda q, k: k <= q and not (q == late and k in protected))
    assert leaked[late] == 3
    assert not plan.edge_allowed(early, early + 1, protected, release)


def test_mask_shape_hides_word_length_and_mapping_and_shams_are_equal_width(config):
    left = plan.compile_layout("gemma", "prospective", "paint", "period", "AB", "A", config=config)
    right = plan.compile_layout("gemma", "prospective", "paint", "period", "AB", "B", config=config)
    assert left["word_cells"] == right["word_cells"]
    for suffix in ("early", "late", "never"):
        context = suffix if suffix == "never" else "release_" + suffix
        hidden, release = plan.mask_spec(left, context)
        sham, sham_release = plan.mask_spec(left, "sham_" + suffix)
        assert len(hidden) == len(sham) == 16
        assert hidden.isdisjoint(sham)
        assert release == sham_release
        assert (hidden, release) == plan.mask_spec(right, context)


def test_budget_includes_donors_nulls_both_routes_and_native_canaries(config):
    budget = plan.compute_budget(config)
    assert budget["sum_per_topic_model"] == 1498
    assert budget["outcome_bridge_total"] == 35952
    assert budget["maximum_with_canaries"] == 36464
    assert budget["maximum_processed_tokens"] == 14002176
    broken = deepcopy(config)
    broken["budget"]["access_forwards_per_context_layout"] -= 1
    with pytest.raises(ValueError, match="Access cost"):
        plan.compute_budget(broken)


def test_checkpoint_cut_and_source_identity_are_checked_against_the_archive(config):
    broken = deepcopy(config)
    broken["models"]["gemma"]["cut"] += 1
    with pytest.raises(ValueError, match="Cut/slot"):
        plan.archive_identity("gemma", "original", broken)
    broken = deepcopy(config)
    broken["models"]["mistral"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Checkpoint fingerprint"):
        plan.archive_identity("mistral", "original", broken)


def test_missing_cell_or_shifted_distance_cannot_silently_enter_analysis(config):
    layouts = plan.compile_all(config)
    with pytest.raises(ValueError, match="Incomplete planned layout grid"):
        plan.validate_grids(layouts[:-1], config)
    bad = deepcopy(layouts)
    next(p for p in bad if p["cell"]["block"] == "distance")["score_index"] += 1
    with pytest.raises(ValueError, match="score displacement"):
        plan.validate_grids(bad, config)


def test_fixed_panel_rule_keeps_missing_nonpositive_and_unstable_results_unresolved():
    decide = plan.finite_panel_decision
    assert decide([1] * 6, [10] * 6)["decision"] == "positive"
    assert decide([-1] * 6, [10] * 6)["decision"] == "negative"
    assert decide([0] * 6, [10] * 6)["decision"] == "equivalent"
    assert decide([1] * 5, [10] * 5)["reason"] == "incomplete_topics"
    assert decide([1] * 6, [-1] * 6)["reason"] == "nonpositive_reference"
    assert decide([1] * 6, [10, -1, -1, -1, -1, -1])["reason"] == "nonpositive_reference"
    assert decide([6, 0, 0, 0, 0, 0], [10] * 6)["decision"] == "unresolved"
    with pytest.raises(ValueError, match="Nonfinite"):
        decide([float("nan")] * 6, [10] * 6)


def test_preflight_receipt_is_reproducible_and_never_claims_gpu_readiness():
    receipt, _ = plan.verify()
    assert receipt == json.loads((HERE / "verification.json").read_text())
    assert receipt["gpu_ready"] is False
    assert receipt["new_model_forwards"] == 0
    assert receipt["frozen_baseline"]["changed_baseline_files"] == []
    assert "per_checkpoint_numeric_mask_adapter_and_tests" in receipt["remaining_execution_gates"]


def test_cell_reducer_reuses_frozen_likelihood_metrics_without_mutation():
    spec = importlib.util.spec_from_file_location("separator_cell_reducer", HERE / "cell_reducer.py")
    reducer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reducer)
    path = HERE.parent / "downstream_span_20261101/trials/clauses-prospective-mistral/interventions.jsonl"
    records = [json.loads(s) for s in path.read_text().splitlines()]
    original = deepcopy(records)
    topics = {r["group"] for r in records}
    select = reducer.context_select("reversed", "B_punct", "delta")
    result = reducer.reduce_cell(records, select, topics=topics)
    assert result["primary_complete"] and result["planned_topics"] == 6
    assert result["legacy_metrics"] == reducer.legacy.panel_metrics(records, select)
    renamed = [{**r, "contexts": {"mapping_flip": r["contexts"]["reversed"]}} for r in records]
    assert reducer.mapping_conflicts(renamed, context="mapping_flip", site="B_punct", method="delta") == \
        reducer.legacy.mapping_conflicts(records, "B_punct", "delta")
    assert records == original
    missing = reducer.reduce_cell(records[:-1], select, topics=topics)
    assert not missing["primary_complete"] and len(missing["missing_topics"]) == 1
    with pytest.raises(ValueError, match="Duplicated"):
        reducer.reduce_cell(records + records[:1], select, topics=topics)
    mixed = [{**r, "cell": {"display_order": "AB" if i == 0 else "BA"}} for i, r in enumerate(records)]
    with pytest.raises(ValueError, match="Mixed factor/display"):
        reducer.reduce_cell(mixed, select, topics=topics)
    invalid = deepcopy(records)
    invalid[0]["reference"]["suffix"]["scores"]["A"] = float("nan")
    with pytest.raises(ValueError, match="Invalid/nonfinite"):
        reducer.reduce_cell(invalid, select, topics=topics)
