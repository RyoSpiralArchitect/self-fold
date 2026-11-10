"""Offline reuse of sealed report text and content-addressed native row scores.

No model host, generation, source compilation or new native scoring is used.
The hard one-hot rows are recovered by matching their bytes to saved call
fingerprints, never by assuming the compiler produced the gold action.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .lm_writer_reports import (fingerprint, rt_report_requests, source_pairs,
                               validate_records)
from .records import file_sha
from .roundtrip import TEXT_FEAT_DIM, TEXT_FEAT_VERSION, text_features


def read(path):
    return json.loads(Path(path).read_text())


def verify_files(root, manifest):
    for name, sha in manifest.items():
        if file_sha(Path(root) / name) != sha:
            raise ValueError(f"Archived input changed: {name}")


def restore_native_rows(folder, keys, reader_sha):
    folder = Path(folder)
    meta, calls = read(folder / "native.json"), read(folder / "calls.json")
    if (meta["reader_sha256"] != reader_sha
            or file_sha(folder / "scores.npz") != meta["scores_sha256"]):
        raise ValueError("Native score/reader binding mismatch")
    basis = np.eye(16, dtype=np.float32)
    hashes = {hashlib.sha256(row.tobytes()).hexdigest(): i
              for i, row in enumerate(basis)}
    with np.load(folder / "scores.npz", allow_pickle=False) as f:
        dispatch, top = f["dispatch"].copy(), f["top"].copy()
    if (dispatch.shape != (len(keys), 2, 2) or top.shape != (len(calls),)
            or dispatch.min() < 0 or dispatch.max() >= len(calls)):
        raise ValueError("Invalid native dispatch")
    recovered = np.empty((len(keys), 2, 16), dtype=np.float32)
    for i, key in enumerate(keys):
        for member in range(2):
            rows = []
            for order in range(2):
                sha, topic, label_order = calls[dispatch[i, member, order]]
                if sha not in hashes or (topic, label_order) != (key[0], order):
                    raise ValueError("Unrecoverable or misrouted hard row")
                rows.append(hashes[sha])
            if rows[0] != rows[1]:
                raise ValueError("Answer order changed the stored action row")
            recovered[i, member] = basis[rows[0]]
    if np.any(recovered[:, 0].argmax(-1) == recovered[:, 1].argmax(-1)):
        raise ValueError("Archived source members collide")
    return torch.from_numpy(recovered), top[dispatch], np.asarray(meta["labels"])


def load_archive(root, cfg, manifest):
    root = Path(root)
    verify_files(root, manifest["files"])
    old = root / manifest["archive"]
    fit_keys, test_keys = [read(old / f"{s}_keys.json")["keys"] for s in ("fit", "test")]
    keys = fit_keys + test_keys
    members, tops, labels = [], [], None
    for split, ks in (("fit", fit_keys), ("test", test_keys)):
        rows, top, lab = restore_native_rows(old / f"{split}_native_hard", ks,
                                             cfg["native_reader_sha256"])
        if labels is not None and not np.array_equal(labels, lab):
            raise ValueError("Label mapping changed")
        members.append(rows); tops.append(top); labels = lab
    with np.load(old / "test_variants_hard_b100/variants.npz", allow_pickle=False) as f:
        if not np.array_equal(f["w"], [0., 1.]) or not np.array_equal(f["top"], tops[1]):
            raise ValueError("Published native variant table differs")
    with np.load(old / "lm_matrix.npz", allow_pickle=False) as f:
        bits = {s: torch.from_numpy(f[s].copy()) for s in
                ("rt_directed", "rt_natural", "state_prior_true")}
        if f["evaluated_masks"].tolist() != cfg["fit_masks"] + cfg["test_masks"]:
            raise ValueError("Evaluation mask set changed")
    pairs, index = source_pairs(keys)
    groups = read(root / "data/within_topic_wording_20261103.json")["groups"]
    masks = sorted(cfg["fit_masks"] + cfg["test_masks"])
    feats, counts = {}, {}
    for style in ("rt_directed", "rt_natural", "state_prior_true"):
        binding = read(old / f"reports_{style}.binding.json")
        rows = [json.loads(line) for line in
                (old / f"reports_{style}.jsonl").read_text().splitlines()]
        raw = [{k: v for k, v in r.items()
                if k not in ("bit", "missing", "decoder_version")} for r in rows]
        requests = binding["requests"]
        contract = dict(schema="lm_report_cache_v3", writer=binding["writer"], requests=requests)
        if (fingerprint(raw) != binding["records_sha256"]
                or fingerprint(contract) != binding["contract_sha256"]
                or len(rows) != binding["count"]):
            raise ValueError("Report corpus binding mismatch")
        validate_records(raw, requests)
        if style.startswith("rt_"):
            if (binding["text_feature_version"] != TEXT_FEAT_VERSION
                    or requests != rt_report_requests(style, masks, pairs, groups)):
                raise ValueError("Report request/feature semantics changed")
            pair_feats = torch.full((len(pairs), 16, 16, 16, TEXT_FEAT_DIM), float("nan"))
            for r in rows:
                pair_feats[r["source"], r["prior"], r["mask"], r["out"]] = text_features(r["text"])
                sources = np.flatnonzero(index == r["source"])
                if not bool((bits[style][sources, r["prior"], r["mask"], r["out"]] == r["bit"]).all()):
                    raise ValueError("Archived decoded bit differs from matrix")
            feats[style] = pair_feats[torch.from_numpy(index)]
        else:
            for r in rows:
                if int(bits[style][r["source"], r["prior"], r["mask"]]) != r["bit"]:
                    raise ValueError("Archived prior-aware bit differs from matrix")
        counts[style] = dict(unique_records=len(rows),
                            censored=sum(bool(r["generation"]["censored"]) for r in rows),
                            records_sha256=binding["records_sha256"])
    return dict(fit_keys=fit_keys, keys=test_keys, members=torch.cat(members),
                fit_top=tops[0], top=tops[1], labels=labels, bits=bits, feats=feats,
                counts=counts, archive=old, pairs=pairs)


def reward_function(case_ix, keys, labels, top):
    """Same lookup reward as fit_lm_writer.make_reward, from saved native calls."""
    def reward(descriptor, mask):
        pole = np.array([keys[i][3] for i in case_ix])[:, None, None]
        worlds = (mask.numpy()[:, None, None] >> np.arange(4)[None, :, None]) & 1
        truth = labels[pole ^ worlds ^ np.arange(2)[None, None, :]]
        got = top[case_ix[:, None], descriptor.long().numpy()]
        return torch.from_numpy((got == truth).astype(np.float32))
    return reward


def pack_patterns(weights):
    weights = np.asarray(weights)
    if weights.shape[-1] != 4 or not np.isin(weights, [0., 1.]).all():
        raise ValueError("Only exact four-world hard patterns may be packed")
    return (weights.astype(np.uint8) * np.array([1, 2, 4, 8], np.uint8)).sum(-1).astype(np.uint8)


def compact_rollout(rolled):
    result = {k: pack_patterns(rolled[k]) for k in ("depths", "second", "fourth", "twins")}
    result["paths"] = rolled["paths"].astype(np.uint8)
    result["prior_mismatch"] = rolled["prior_mismatch"]
    # Reports are exactly regenerable from the sealed corpus and transition
    # patterns. Retain the full carried payload hash without duplicating it.
    written = np.ascontiguousarray(rolled["written"])
    result["written_sha256"] = np.asarray(hashlib.sha256(written.tobytes()).hexdigest())
    result["written_dtype"] = np.asarray(str(written.dtype))
    result["written_shape"] = np.asarray(written.shape)
    result["written_semantics"] = rolled["written_semantics"]
    return result


def reduce_compact(data, keys, masks, labels, top, donors, test_masks, fit_topics):
    """Score packed realized patterns against saved full-vocabulary winners."""
    p = np.asarray(masks)[data["paths"]]
    n, paths = len(keys), len(p)
    pole = np.asarray([k[3] for k in keys])

    def bits(x):
        return (np.asarray(x)[..., None] >> np.arange(4)) & 1

    def expected(ms):
        return labels[pole[:, None, None, None] ^ bits(ms)[None, ..., None] ^ np.arange(2)]

    def got(pattern, ids):
        idx = bits(pattern)
        ids = np.asarray(ids).reshape(-1, *([1] * (idx.ndim - 1)))
        return top[ids, idx]

    def count(a):
        return [int(a.sum()), int(a.size)]

    complete = np.ones((n, paths), bool)
    per_depth = []
    first = None
    for depth in range(4):
        hit = got(data["depths"][:, :, depth], donors[0]) == expected(p[:, depth])
        per_depth.append(count(hit))
        complete &= hit.all((-1, -2))
        if depth == 0:
            first_indices = np.unique(p[:, 0], return_index=True)[1]
            first = count(hit[:, first_indices].all((-1, -2)))
    role = np.concatenate([np.zeros((paths, 1), int), bits(p)[..., 0]], -1)
    flips = (np.diff(role, axis=-1)[:, :3] != 0).sum(-1)
    opposed = {}
    for label, field, depth in (("depth2", "second", 1), ("depth4", "fourth", 3)):
        truth = expected(p[:, depth])
        hit = got(data[field][:, 0], donors[0]) == truth
        opp = got(data[field][:, 1], donors[1]) == truth[..., ::-1]
        equiv = got(data[field][:, 2], donors[2]) == truth
        erased = got(data[field][:, 3], donors[0]) == truth
        opposed[label] = dict(matched=count(hit), opposed_correct=count(hit & opp),
                              equivalent_correct=count(hit & equiv), erased_intact=count(erased))
    tgot = got(data["twins"], np.arange(n))
    truth = labels[pole[:, None, None, None, None] ^ bits(np.asarray(masks))[None, None, ..., None] ^ np.arange(2)]
    twins, twin_units = {}, {}
    for i, name in enumerate(("intact", "erased")):
        hits = (tgot[:, :, :, 0, i] == truth) & (tgot[:, :, :, 1, i] == truth[..., ::-1])
        twins[name], twin_units[name] = count(hits), count(hits.all((-1, -2)))
    novel = np.isin(p, test_masks).all(-1)
    return dict(native_per_depth=per_depth, first_complete_source_mask=first,
                all_four_correct=count(complete),
                all_four_novel_mask_strings=count(complete[:, novel]),
                all_four_by_role_exchanges={str(i): count(complete[:, flips == i]) for i in range(4)},
                all_four_by_topic={str(t): count(complete[np.asarray([k[0] == t for k in keys])])
                                   for t in sorted({k[0] for k in keys})},
                all_four_unfitted_topics=count(complete[np.asarray([k[0] not in fit_topics for k in keys])]),
                opposed=opposed, twins_both_histories=twins, twins_complete_units=twin_units,
                prior_mismatch=count(data["prior_mismatch"]))
