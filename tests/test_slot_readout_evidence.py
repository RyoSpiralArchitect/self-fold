"""Verify the completed readout cycle without loading any base checkpoint."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from selffold.records import digest, file_sha

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/slot_readout_20261103"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_readout_archive_membership_digests_and_reducer_replay():
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", str(STUDY.relative_to(ROOT))], cwd=ROOT
    )
    paths = {
        Path(x.decode()).relative_to(STUDY.relative_to(ROOT)).as_posix()
        for x in tracked.split(b"\0")
        if x
    }
    lines = (STUDY / "SHA256SUMS").read_text().splitlines()
    manifest = {name: sha for sha, name in (line.split("  ", 1) for line in lines)}
    assert len(lines) == len(manifest) and set(manifest) == paths - {"SHA256SUMS"}
    for name, sha in manifest.items():
        assert hashlib.sha256((STUDY / name).read_bytes()).hexdigest() == sha
    assert (
        subprocess.check_output([sys.executable, str(STUDY / "summarize.py")])
        == (STUDY / "summary.json").read_bytes()
    )


def test_readout_budget_artifact_host_bindings_and_original_E_states():
    runs = STUDY / "run"
    original = ROOT / "provenance/slot_content_20261101/trials/contrasts-gemma"
    updates = artifacts = 0
    for arm, count in (("order", 128), ("decoder", 64), ("lexical", 64)):
        folder = runs / arm
        seal, request, terminal = [
            read(folder / (name + ".json"))
            for name in ("source", "request", "terminal")
        ]
        assert seal["git_commit"] == "ad2d1a2e9016fa5fb368549ef1ee20320f1ffb2e"
        assert (
            seal["git_status"] == "" and digest(seal["files"]) == seal["source_sha256"]
        )
        assert request["plan_sha256"] == file_sha(STUDY / "PLAN.md")
        assert request["fixture_sha256"] == file_sha(
            ROOT / "data/slot_contrasts.dev.json"
        )
        assert read(folder / "host.json") == read(original / "host_identity.json")
        assert read(folder / "native_parity.json")["exact"]
        assert (
            terminal["status"] == "COMPLETED_BOUNDED_READOUT"
            and terminal["updates"] == count
        )
        assert terminal["backbone_sha256_before"] == terminal["backbone_sha256_after"]
        assert (
            not terminal["G_trained"] and not terminal["qualification_content_loaded"]
        )
        history = rows(folder / "training.jsonl")
        assert len(history) == count and all(
            np.isfinite(r["gradient_norm"]) and r["gradient_norm"] > 0 for r in history
        )
        updates += count
        fits = rows(folder / "fits.jsonl")
        for fit in fits:
            path = folder / fit["artifact"]
            assert file_sha(path) == fit["sha256"] and fit["serialization_exact"]
            with np.load(path, allow_pickle=False) as arrays:
                metadata = json.loads(arrays["metadata_utf8"].tobytes())
                assert metadata["source_sha256"] == seal["source_sha256"]
                assert metadata["base_host_sha256"] == digest(
                    read(folder / "host.json")
                )
                if arm == "lexical":
                    assert metadata["kind"] == "lexical_slot_low_rank_residual"
                    assert (
                        metadata["capture_geometry"]
                        == "matched_recipient_length_lexical_slots"
                    )
                if arm == "decoder":
                    assert fit["active_layers"] == list(range(13, 26))
            artifacts += 1
    assert updates == 256 and artifacts == 4
    donors = rows(runs / "decoder/donors.jsonl")
    assert len(donors) == 24
    for donor in donors:
        assert (
            donor["state_sha256"]
            == read(original / "states" / donor["case"] / "E.json")["sha256"]
        )
    lexical = rows(runs / "lexical/donors.jsonl")
    assert len(lexical) == 24 and all(
        d["neutral_native_exact"] and d["opaque_lexical_exact"] for d in lexical
    )


def test_frozen_forward_reader_replays_prior_matched_orders_and_group_schedule():
    prior = ROOT / "provenance/policy_form_20261103/run"
    old = {
        (
            r["condition"],
            r["form"],
            r["split"],
            r["case"],
            r["world"],
            r["label_order"],
        ): r
        for r in rows(prior / "decisions.jsonl")
        if r["endpoint"] == "case_fit" and r["form"] in ("forward", "reverse")
    }
    seen = set()
    for r in rows(STUDY / "run/order/decisions.jsonl"):
        if (
            r["endpoint"] != "forward_fit"
            or r["form"] not in ("forward", "reverse")
            or r["split"] == "additional_dev"
        ):
            continue
        key = (
            r["condition"],
            r["form"],
            "dev" if r["split"] == "previous_dev" else "train",
            r["case"],
            r["world"],
            r["label_order"],
        )
        assert all(
            r[k] == old[key][k]
            for k in (
                "A",
                "B",
                "gold",
                "greedy_token",
                "source_sha256",
                "question_sha256",
            )
        )
        assert key not in seen
        seen.add(key)
    assert seen == old.keys() and len(seen) == 768
    assert read(STUDY / "run/order/schedule.json") == read(
        ROOT / "provenance/continuity_20261103/trainability/schedule.json"
    )
