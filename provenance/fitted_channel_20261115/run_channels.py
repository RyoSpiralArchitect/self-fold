"""Fitted-lexical-reader channel: end-to-end boundary endpoints, fully offline.

Experiment B (see PLAN.md). Plugs alternate report->bit mappings into the
sealed write7 rollout machinery — no model, no generation. Everything reuses
stored artifacts: fitted supplied-controller weights, frozen native scores,
the sealed report corpus, and the same rollout_rt/twins_w code paths.

Arms per style (rt_natural, rt_directed), all through the shared
supplied-fitted controller (channel quality isolated from per-channel refits):
- decoder:      sealed rowbound bits (replicates write7 endpoints = sanity)
- oracle:       bit = realized out & 1 (perfect writer ceiling)
- fitted_text:  word-TFIDF logistic reader, GroupKFold over unique texts
                (familiar wording, held-out exact strings)
- fitted_topic: same reader, GroupKFold over topic groups (cross-topic)
- none:         all-zero messages (12.5% floor)

Usage:
  PYTHONPATH=src python3 provenance/fitted_channel_20261115/run_channels.py [style ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from selffold.lm_writer import decode_state_slot  # noqa: E402
from selffold.lm_writer_reports import source_pairs  # noqa: E402
from selffold.message_boundary import BoundaryMemory  # noqa: E402
from selffold.mixture_boundary import rollout_rt, twins_w  # noqa: E402
from selffold.roundtrip import feasible_outs  # noqa: E402

PROD = ROOT / "provenance/message_boundary_20261106/write7/produce"
OUT = Path(__file__).resolve().parent
SEEDS = (17, 41, 73)


def bits(mask):
    return (np.asarray(mask)[..., None] >> np.arange(4)) & 1


def expected(keys, masks, labels):
    polarity = np.array([k[3] for k in keys]).reshape(
        (-1,) + (1,) * (np.ndim(masks) + 2))
    return labels[polarity ^ bits(masks)[..., None] ^ np.arange(2)]


def select_top(top, widx, case_ids):
    ids = np.asarray(case_ids).reshape(-1, *([1] * (widx.ndim - 1)))
    return top[ids, widx]


def load_bank():
    fit_keys = json.loads((PROD / "fit_keys.json").read_text())["keys"]
    test_keys = json.loads((PROD / "test_keys.json").read_text())["keys"]
    all_keys = fit_keys + test_keys
    pairs, index = source_pairs(all_keys)
    groups = json.loads((ROOT / "data/within_topic_wording_20261103.json").read_text())["groups"]
    join = np.load(PROD / "join.npz")
    variants = np.load(PROD / "test_variants_hard_b100/variants.npz")
    labels = np.array(json.loads(
        (PROD / "test_native_hard/native.json").read_text())["labels"])
    return dict(fit_n=len(fit_keys), keys=test_keys, all_keys=all_keys,
                pairs=pairs, index=index, groups=groups,
                donors=join["donors"], eval_masks=join["eval_masks"].tolist(),
                top=variants["top"], labels=labels)


def load_reports(style):
    """pair-level records: {(pair, prior, mask, out): text} plus slot label."""
    recs = {}
    for line in open(PROD / f"reports_{style}.jsonl"):
        r = json.loads(line)
        recs[(r["source"], r["prior"], r["mask"], r["out"])] = (
            r["text"], (r["out"] & 1) if "out" in r else (r["mask"] & 1))
    return recs


def fitted_predictions(recs, bank, group_by):
    """Held-out slot predictions per record. group_by: 'text' or 'topic'."""
    keys = list(recs)
    texts = np.array([recs[k][0] for k in keys])
    slots = np.array([recs[k][1] ^ bank["pairs"][k[0]][1] for k in keys])
    if group_by == "text":
        groups = np.array([hash(t) for t in texts])
    else:
        groups = np.array([bank["pairs"][k[0]][0] for k in keys])
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit_transform(texts)
    clf = LogisticRegression(max_iter=2000, C=10)
    splits = GroupKFold(5).split(X, slots, groups)
    pred = cross_val_predict(clf, X, slots, cv=splits, n_jobs=-1)
    return dict(zip(keys, pred))


def build_matrix(style, arm, bank):
    """[n_test_sources,16,16,16] bit matrix for one channel arm."""
    pairs, index = bank["pairs"], bank["index"]
    test_index = index[bank["fit_n"]:]
    if arm == "decoder":
        stored = np.load(PROD / "lm_matrix.npz")[style]
        return torch.from_numpy(stored[bank["fit_n"]:])
    if arm == "none":
        return torch.zeros(len(bank["keys"]), 16, 16, 16, dtype=torch.long)
    recs = load_reports(style)
    M = torch.zeros(len(pairs), 16, 16, 16, dtype=torch.long)
    if arm == "oracle":
        for (p, prior, mask, out) in recs:
            M[p, prior, mask, out] = out & 1
    else:
        group_by = {"fitted_text": "text", "fitted_topic": "topic"}[arm]
        pred = fitted_predictions(recs, bank, group_by)
        for (p, prior, mask, out), slot in pred.items():
            M[p, prior, mask, out] = int(slot) ^ bank["pairs"][p][1]
    return M[torch.from_numpy(test_index)]


def actor_for(seed):
    actor = BoundaryMemory(seed, 1).requires_grad_(False)
    with np.load(PROD / "fits.npz", allow_pickle=False) as f:
        prefix = f"hard_supplied_w1_{seed}_"
        actor.load_state_dict(
            {k: torch.from_numpy(f[prefix + k]) for k in actor.state_dict()})
    return actor


def canonical_members(n):
    m = torch.zeros(n, 2, 16)
    m[:, 0, 0] = 1.0
    m[:, 1, 1] = 1.0
    return m


def score(rolled, twins, bank):
    """Endpoint counting, same semantics as summarize_mixture.py."""
    keys, donors, eval_masks = bank["keys"], bank["donors"], bank["eval_masks"]
    top, labels = bank["top"], bank["labels"]
    paths = rolled["paths"]
    path_masks = np.asarray(eval_masks)[paths]
    hits = []
    for d in range(4):
        widx = rolled["depths"][:, :, d].astype(np.int64)
        hits.append(select_top(top, widx, donors[0])
                    == expected(keys, path_masks[:, d], labels))
    hits = np.stack(hits, 2)
    complete = hits.all((2, 3, 4))
    tgot = select_top(top, twins.astype(np.int64), np.arange(len(keys)))
    base = expected(keys, np.asarray(eval_masks), labels)
    truth0 = np.broadcast_to(
        base[:, None], (len(keys), twins.shape[1], len(eval_masks), 4, 2))
    twin_out = {}
    for ci, cond in enumerate(("intact", "erased")):
        hit0 = tgot[:, :, :, 0, ci] == truth0
        hit1 = tgot[:, :, :, 1, ci] == truth0[..., ::-1]
        twin_out[cond] = [int((hit0 & hit1).sum()), int(hit0.size)]
    return dict(
        native_per_depth=[[int(hits[:, :, d].sum()), int(hits[:, :, d].size)]
                          for d in range(4)],
        all_four_correct=[int(complete.sum()), int(complete.size)],
        twins_both_histories=twin_out,
    )


def main():
    bank = load_bank()
    styles = sys.argv[1:] or ["rt_natural", "rt_directed"]
    arms = ("decoder", "oracle", "fitted_text", "fitted_topic", "none")
    members = canonical_members(len(bank["keys"]))
    out_path = OUT / "summary.json"
    summary = json.loads(out_path.read_text()) if out_path.exists() else {"arms": {}}
    for style in styles:
        matrices = {arm: build_matrix(style, arm, bank) for arm in arms}
        for arm in arms:
            for seed in SEEDS:
                name = f"{style}_{arm}_{seed}"
                if name in summary["arms"]:
                    continue
                actor = actor_for(seed)
                rolled = rollout_rt(actor, "hard", members, bank["keys"],
                                    bank["eval_masks"], matrices[arm])
                twins = twins_w(actor, "hard", "lm", members,
                                bank["eval_masks"], 1.0,
                                rt_bits=matrices[arm])
                summary["arms"][name] = score(rolled, twins, bank)
                print(name, json.dumps(summary["arms"][name]), flush=True)
                out_path.write_text(json.dumps(summary, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
