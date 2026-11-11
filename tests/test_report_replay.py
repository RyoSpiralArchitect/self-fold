"""Offline evidence must preserve actual rows, score joins and input bytes."""
import hashlib
import itertools
import json

import numpy as np
import pytest

from selffold.records import file_sha
from selffold.report_replay import pack_patterns, reduce_compact, restore_native_rows, verify_files
from selffold.role_trials import donor_indices


def native_fixture(tmp_path):
    # The recorded actions intentionally differ from topic*2 + polarity.
    # Restoring them must use their fingerprints, not the expected answer.
    rows = np.eye(16, dtype=np.float32)[[10, 11]]
    calls = [[hashlib.sha256(row.tobytes()).hexdigest(), 0, order]
             for row in rows for order in (0, 1)]
    (tmp_path / "calls.json").write_text(json.dumps(calls))
    np.savez(tmp_path / "scores.npz", dispatch=np.arange(4).reshape(1, 2, 2),
             top=np.array([11, 12, 12, 11]), ab=np.zeros((4, 2)))
    (tmp_path / "native.json").write_text(json.dumps(dict(
        reader_sha256="reader", scores_sha256=file_sha(tmp_path / "scores.npz"), labels=[11, 12])))
    return rows


def test_native_row_recovery_uses_recorded_fingerprints(tmp_path):
    rows = native_fixture(tmp_path)
    recovered, top, labels = restore_native_rows(tmp_path, [[0, 0, 0, 0, 0]], "reader")
    assert np.array_equal(recovered.numpy()[0], rows)
    assert top.tolist() == [[[11, 12], [12, 11]]] and labels.tolist() == [11, 12]


def test_native_row_recovery_rejects_unknown_or_misrouted_row(tmp_path):
    native_fixture(tmp_path)
    path = tmp_path / "calls.json"
    calls = json.loads(path.read_text())
    calls[0][0] = "unrecoverable"
    path.write_text(json.dumps(calls))
    with pytest.raises(ValueError, match="Unrecoverable"):
        restore_native_rows(tmp_path, [[0, 0, 0, 0, 0]], "reader")


def test_archive_hash_mismatch_stops_before_reuse(tmp_path):
    path = tmp_path / "report.jsonl"
    path.write_text("first\n")
    manifest = {path.name: file_sha(path)}
    verify_files(tmp_path, manifest)
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="Archived input changed"):
        verify_files(tmp_path, manifest)


def test_packed_patterns_are_exact_and_reject_soft_rows():
    masks = np.arange(16, dtype=np.uint8)
    bits = (masks[:, None] >> np.arange(4)) & 1
    assert np.array_equal(pack_patterns(bits), masks)
    with pytest.raises(ValueError, match="exact"):
        pack_patterns(np.full((1, 4), 0.5))


def test_compact_reducer_counts_complete_paths_and_opposed_twins():
    keys = [(0, family, 0, pole, order) for family in (0, 1)
            for pole in (0, 1) for order in (0, 1)]
    masks = [6, 9]
    paths = np.array(list(itertools.product(range(2), repeat=4)), np.uint8)
    actual = np.array(masks)[paths]
    n, count = len(keys), len(paths)
    labels = np.array([11, 12])
    top = np.array([labels[pole ^ np.arange(2)[:, None] ^ np.arange(2)]
                    for _, _, _, pole, _ in keys])
    twin = np.empty((n, 1, 2, 2, 2), np.uint8)
    for history in (0, 1):
        twin[:, :, :, history, :] = (np.asarray(masks)[None, None, :, None]
                                     if history == 0 else 15-np.asarray(masks)[None, None, :, None])
    data = dict(paths=paths, depths=np.broadcast_to(actual, (n, count, 4)),
                second=np.broadcast_to(actual[None, None, :, 1], (n, 4, count)),
                fourth=np.broadcast_to(actual[None, None, :, 3], (n, 4, count)),
                twins=twin, prior_mismatch=np.zeros((n, 3, count), bool))
    result = reduce_compact(data, keys, masks, labels, top, donor_indices(keys), masks, [0])
    assert result["all_four_correct"] == [n*count, n*count]
    assert result["first_complete_source_mask"] == [n*2, n*2]
    assert result["native_per_depth"] == [[n*count*8, n*count*8]] * 4
    assert result["twins_complete_units"]["intact"] == [n*2, n*2]
    assert result["opposed"]["depth4"]["opposed_correct"] == [n*count*8, n*count*8]
