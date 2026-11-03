"""Verify Boolean support, physical one-bit controls and both frozen development endpoints."""

import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from selffold.gates import REQUIRED, verify_gate
from selffold.records import digest, file_sha
from selffold.supported_boundary import cases, condition

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/supported_g_20261104"


def read(path):
    return json.loads(path.read_text())


def script(name):
    # Evidence reducers live beside the immutable study data.
    spec = importlib.util.spec_from_file_location(
        "boundary_evidence_" + name, STUDY / (name + ".py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_truth_ledger_requires_changes_and_justified_stability_under_one_rule():
    config = read(ROOT / "data/supported_g_20261104.json")
    panel = cases(config, "fresh")
    assert len(panel) == 1280 and len(cases(config, "fresh", raw=True)) == 128
    assert len(set(panel)) == len(panel)
    counts = [0, 0, 0]
    for c in panel:
        rows = c.ledger()
        expected = [
            c.pole ^ int(condition(c.base, w) and not condition(c.veto, w))
            for w in range(4)
        ]
        assert expected == [r["choice"] for r in rows]
        twin = c.donor("twin").ledger()
        assert [r["base_choice"] for r in rows] == [r["base_choice"] for r in twin]
        assert all(
            (r["choice"] != o["choice"]) == r["veto"]
            for r, o in zip(rows, twin, strict=True)
        )
        assert (
            c.donor("reordered").ledger() == rows
            and c.donor("equivalent").ledger() == rows
        )
        counts[0] += any(r["affected"] for r in rows)
        counts[1] += sum(r["affected"] for r in rows)
        counts[2] += sum(r["veto"] for r in rows)
    assert counts == [768, 1024, 2048]


@pytest.mark.parametrize(
    "folder,summary",
    [("structured", "structured"), ("readout", "readout"), ("raw-bound", "raw")],
)
def test_all_six_reductions_recompute_on_the_registered_denominators(folder, summary):
    result = script("summarize").summarize(STUDY, folder)
    assert result == read(STUDY / (summary + "-summary.json"))
    assert set(result["checks"]) == REQUIRED
    assert (
        result["formal_challenge"] == "not_opened"
        and result["pilot_status"] == "NOT_QUALIFIED"
    )
    if folder == "raw-bound":
        assert result["routes"]["boundary"]["score"] == 507
        assert (
            result["checks"]["meaning_swap_effect"]["correct_opposed_reversal"]["score"]
            == 0
        )
    else:
        with np.load(STUDY / folder / "dispatch.npz", allow_pickle=False) as routes:
            assert np.array_equal(routes["analytic"], routes["updated"])
            assert np.array_equal(routes["erased"], routes["twin_erased"])
        assert (
            result["exact_table_twins"]["veto_erased"]["score"],
            result["exact_table_twins"]["veto_intact"]["n"],
        ) == (0, 4096)
    n = sum(x["mechanical"] == "pass" for x in result["checks"].values())
    assert n == {"structured": 5, "readout": 2, "raw-bound": 1}[folder]


def test_field_intervention_reproduces_unmodified_output_and_localizes_source_errors(
    monkeypatch,
):
    monkeypatch.syspath_prepend(str(STUDY))
    monkeypatch.setitem(sys.modules, "summarize", script("summarize"))
    # The reducer has no access to model generation or data selection.
    module = script("diagnose_fields")
    for folder in ("structured", "readout"):
        result = module.diagnose(STUDY, folder)
        assert result == read(STUDY / (folder + "-fields.json"))
    assert read(STUDY / "structured-fields.json")["true_veto"]["correct"] == 10240
    assert read(STUDY / "readout-fields.json")["both"]["correct"] == 9920


def test_new_literal_calibration_does_not_refit_G_or_reuse_test_prompts():
    p = STUDY / "readout"
    cfg = read(ROOT / "data/supported_g_readout_20261104.json")
    fit = read(p / "fit_closed.json")
    term = read(p / "terminal.json")
    assert (
        fit["fresh_captures"],
        fit["target_labels"],
        fit["updates"],
        term["G_updates"],
    ) == (0, 3072, 192, 0)
    assert (
        file_sha(STUDY / "structured/heads.npz")
        == cfg["source_and_update_sha256"]
        == fit["source_and_update"]
    )
    assert read(p / "binding.json")["literal_reader"] == file_sha(
        p / "literal_reader.npz"
    )
    schedule = read(p / "schedule.json")
    assert len(schedule) == 192 and {t for t, m in schedule} == {0, 2, 4, 6}
    fit_prompts = set(read(p / "fit_prompts.json"))
    test_prompts = set()
    menus = read(ROOT / "provenance/role_memory_20261104/run/produce/packet/menus.json")
    with gzip.open(p / "rendered.jsonl.gz", "rt") as f:
        for row in map(json.loads, f):
            for w in range(4):
                for o in range(2):
                    a, b = menus[row["topic"]][:: 1 if o == 0 else -1]
                    q = cfg["fresh_question"].format(
                        left=("off", "on")[w // 2], right=("off", "on")[w % 2]
                    )
                    test_prompts.add(
                        digest(
                            [
                                row["text"],
                                q + f"\nA = {a}. B = {b}. Respond with only A or B.",
                            ]
                        )
                    )
    assert len(test_prompts) == 1016 and not fit_prompts & test_prompts
    comp = read(STUDY / "readout-summary.json")["readout_comparison"]["held_menus"]
    assert (
        comp["D_updated_parent"]["score"],
        comp["D_updated"]["score"],
        comp["D_updated"]["n"],
    ) == (3710, 3960, 5120)
    assert not comp["passes"] and not comp["reader_gain"]


def test_preserved_binding_stop_native_eos_and_exact_physical_replay():
    p = STUDY / "raw-bound"
    first = STUDY / "raw-initial-stop"
    assert read(first / "terminal.json")["status"] == "TECHNICAL_STOP"
    for line in (first / "captured.sha256").read_text().splitlines():
        sha, path = line.split("  ")
        assert sha == file_sha(p / Path(path).name)
    end = read(p / "continued.json")
    assert end["refits"] == 0 and end["free_text"] == file_sha(p / "free_text.jsonl")
    sources = {
        tuple(r["key"]): r
        for r in map(json.loads, (p / "sources.jsonl").read_text().splitlines())
    }
    outputs = list(map(json.loads, (p / "free_text.jsonl").read_text().splitlines()))
    assert len(sources) == 128 and len(outputs) == 32
    for row in outputs:
        i = {"initial": 0, "updated": 1}[row["stage"]]
        assert row["state"] == sources[tuple(row["key"])]["hashes"][i]
        assert row["stop"] == "eos" and not row["censored"]
    binding = read(p / "binding.json")
    assert binding["task_host"] == read(
        ROOT / "provenance/preprint_v1_20261104/raw/host.json"
    )
    assert binding["host"]["max_context"] == 512 and binding["host"][
        "transport_sha256"
    ] == file_sha(p / "transport.npz")
    replay = read(STUDY / "replay/receipt.json")
    assert (
        sum(replay["exact"].values()) == 4320
        and replay["exact"]["raw_state_decisions"] == 192
    )
    assert replay["literal"] == file_sha(STUDY / "readout/literal_reader.npz")


def test_development_record_cannot_admit_a_latent_pilot_and_files_are_bound(
    monkeypatch,
):
    gate = read(STUDY / "gate.json")
    manifest = read(ROOT / "data/codec_preparation_20261102/manifest.json")
    assert (
        not gate["qualification_opened"]
        and gate["qualification_commitment_sha256"]
        == manifest["qualification"]["sha256"]
    )
    assert set(gate["checks"]) == REQUIRED and all(
        v["verdict"] == "not_run" for v in gate["checks"].values()
    )
    for data in gate["checks"].values():
        assert file_sha(STUDY / data["evidence_path"]) == data["evidence_sha256"]
    with pytest.raises(ValueError, match="has not passed"):
        verify_gate(STUDY / "gate.json", host_identity={}, source_sha256="0" * 64)
    monkeypatch.syspath_prepend(str(STUDY))
    monkeypatch.setitem(sys.modules, "summarize", script("summarize"))
    for k, v in script("screen_record").build(STUDY).items():
        assert (STUDY / k).read_bytes() == v
    files = {
        str((ROOT / n.decode()).relative_to(STUDY))
        for n in subprocess.check_output(
            ["git", "ls-files", "-z", "--", str(STUDY)], cwd=ROOT
        ).split(b"\0")
        if n
    }
    hashes = {
        name: sha
        for sha, name in (
            line.split("  ", 1)
            for line in (STUDY / "SHA256SUMS").read_text().splitlines()
        )
    }
    assert files == set(hashes) | {"SHA256SUMS"}
    assert all(file_sha(STUDY / name) == v for name, v in hashes.items())
