"""The exploratory paired readout retains all native data and cannot alter admission."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/conditional_margin_20261103"
SPEC = importlib.util.spec_from_file_location(
    "conditional_margin_diagnostic", STUDY / "diagnose.py"
)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


@pytest.fixture(scope="module")
def rows():
    return [
        json.loads(line)
        for line in (STUDY / "run/decisions.jsonl").read_text().splitlines()
    ]


def test_saved_diagnostic_separates_native_labels_from_external_paired_decisions():
    result = DIAGNOSTIC.summary(STUDY / "run/decisions.jsonl")
    assert result == json.loads((STUDY / "diagnostic.json").read_text())
    assert result["native_observations"] == 8960
    assert (
        not result["native_gates_changed"]
        and not result["native_admission_established"]
    )
    assert result["fit_error_asymmetry"]["errors"] == 89
    assert result["fit_error_asymmetry"]["gold_A_predicted_B"] == 85
    assert result["fit_error_asymmetry"]["by_stage"]["source"][
        "all_on_errors_by_source_order"
    ] == {"0": 0, "1": 23}
    assert result["fit_error_asymmetry"]["by_stage"]["cases"][
        "all_on_errors_by_source_order"
    ] == {"0": 25, "1": 0}
    assert result["fit_native_initial"] == [3687, 3840]
    assert result["fit_native_final"] == [3751, 3840]
    expected = {
        "fit": {
            "source": ([767, 768], [47, 48]),
            "cases": ([757, 768], [38, 48]),
            "rule": ([384, 384], [48, 48]),
        },
        "fresh": {
            "source": ([157, 256], [0, 16]),
            "cases": ([241, 256], [6, 16]),
            "rule": ([92, 128], [0, 16]),
        },
    }
    for split, stages in expected.items():
        for stage, (correct, strict) in stages.items():
            measured = result["paired_order"][split][stage]
            assert (
                measured["forced_correct"] == correct
                and measured["forced_strict"] == strict
            )
            assert measured["excluded_native_non_ab"] == 0
            assert measured["native_correct"][1] == 2 * correct[1]
            assert (
                sum(measured["native_correct_mappings_per_pair"].values()) == correct[1]
            )
    cases = result["paired_order"]["fresh"]["cases"]
    assert cases["native_coverage"] == [496, 512] and cases["forced_correct"][1] == 256
    assert (
        sum(
            result["fit_world_transitions"][s]["3"]["net_change"]
            for s in DIAGNOSTIC.STAGES
        )
        == -21
    )


@pytest.mark.parametrize("change", ["missing", "duplicate", "gold", "question"])
def test_diagnostic_rejects_incomplete_or_changed_native_pair_ledger(rows, change):
    edited = list(rows)
    if change == "missing":
        edited.pop()
    elif change == "duplicate":
        edited.append(edited[0])
    else:
        row = dict(edited[0])
        row["gold" if change == "gold" else "question_sha256"] = (
            ("A" if row["gold"] == "B" else "B") if change == "gold" else "0" * 64
        )
        edited[0] = row
    with pytest.raises(ValueError):
        DIAGNOSTIC.diagnose(edited)


def test_diagnostic_rejects_an_unsealed_input_file(tmp_path):
    path = tmp_path / "decisions.jsonl"
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="sealed"):
        DIAGNOSTIC.summary(path)
