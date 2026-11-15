"""Ideal-parser ceiling: supervised text->bit classification on sealed reports.

Development measurement for provenance/writer_parser_20261115/PLAN.md.
Reads the sealed write6 corpora; writes summary.json next to itself.
No generation, no model weights, no boundary fitting.

Usage:
  PYTHONPATH=src python3 provenance/writer_parser_20261115/analyze.py
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from selffold.lm_writer_reports import source_pairs  # noqa: E402

PROD = ROOT / "provenance/message_boundary_20261106/write6/produce"
CORPORA = {
    "rt_natural": "reports_rt_natural.jsonl",
    "rt_directed": "reports_rt_directed.jsonl",
    "state_prior_true": "reports_state_prior_true.jsonl",
}
COORD = re.compile(r"\b(?:off|on)\s*[/\-]\s*(?:off|on)\b", re.IGNORECASE)
N_PERM = 200
RNG = np.random.default_rng(20261115)


def load(name):
    keys = json.loads((PROD / "fit_keys.json").read_text())["keys"] + \
        json.loads((PROD / "test_keys.json").read_text())["keys"]
    pairs, _ = source_pairs(keys)
    rows = []
    for line in open(PROD / CORPORA[name]):
        r = json.loads(line)
        gi, pole = pairs[r["source"]] if "out" in r else (keys[r["source"]][0], keys[r["source"]][3])
        out_bit = (r["out"] if "out" in r else r["mask"]) & 1
        mask_bit = r["mask"] & 1
        rows.append(dict(
            text=r["text"], group=gi, pole=pole,
            out_bit=out_bit, mask_bit=mask_bit,
            # The decodable quantity: which of the two options the correct
            # answer names. A parser recovers bit = named_slot ^ pole, so
            # predicting slot is exactly the parser ceiling on the bit.
            slot_out=out_bit ^ pole, slot_mask=mask_bit ^ pole,
        ))
    return rows


def form_of(text, options):
    t = " " + text.casefold() + " "
    for i, opt in enumerate(options):
        t = t.replace(opt.casefold(), f" <{'AB'[i]}> ")
    t = COORD.sub(" <ROW> ", t)
    return re.sub(r"\s+", " ", t).strip()


def vectorizers(rows):
    texts = [r["text"] for r in rows]
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit_transform(texts)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(texts)
    return word, char


def cv_predict(X, y, rows, scheme, clf):
    if scheme == "random":
        splits = StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)
        return cross_val_predict(clf, X, y, cv=splits, n_jobs=-1)
    groups = np.array([r["group"] if scheme == "group-topic" else hash(r["text"]) for r in rows])
    splits = GroupKFold(5).split(X, y, groups)
    return cross_val_predict(clf, X, y, cv=splits, n_jobs=-1)


def evaluate(name, rows):
    word, char = vectorizers(rows)
    texts = np.array([r["text"] for r in rows])
    out = {"n": len(rows), "unique_texts": len(set(texts)), "targets": {}}
    for target in ("slot_out", "slot_mask"):
        y = np.array([r[target] for r in rows])
        diverge = np.array([r["out_bit"] != r["mask_bit"] for r in rows])
        base = max(y.mean(), 1 - y.mean())
        tgt = {"base_rate": float(base), "diverge_fraction": float(diverge.mean()), "estimators": {}}
        for est, X, clf in (
            ("word_lr", word, LogisticRegression(max_iter=2000, C=10)),
            ("char_lr", char, LogisticRegression(max_iter=2000, C=10)),
            ("word_gb", word.toarray(), HistGradientBoostingClassifier(max_iter=100)),
        ):
            res = {}
            for scheme in ("group-topic", "group-text", "random"):
                pred = cv_predict(X, y, rows, scheme, clf)
                res[scheme] = {
                    "acc": float((pred == y).mean()),
                    "acc_same": float((pred == y)[~diverge].mean()) if (~diverge).any() else None,
                    "acc_diff": float((pred == y)[diverge].mean()) if diverge.any() else None,
                }
            # permutation null on the loose split, word_lr only (shared across
            # estimators: the null depends on the split, not the learner)
            if est == "word_lr":
                null = np.empty(N_PERM)
                for i in range(N_PERM):
                    yp = RNG.permutation(y)
                    splits = StratifiedKFold(3, shuffle=True, random_state=i).split(X, yp)
                    null[i] = (cross_val_predict(clf, X, yp, cv=splits, n_jobs=-1) == yp).mean()
                res["null_p95"] = float(np.quantile(null, 0.95))
                res["null_max"] = float(null.max())
            tgt["estimators"][est] = res
        out["targets"][target] = tgt
    return out


def main():
    summary = {"corpora": {}}
    for name in CORPORA:
        rows = load(name)
        summary["corpora"][name] = evaluate(name, rows)
        for tgt_name, t in summary["corpora"][name]["targets"].items():
            acc = {e: r["random"]["acc"] for e, r in t["estimators"].items()}
            print(f"{name}/{tgt_name}: " + ", ".join(f"{k}={v:.3f}" for k, v in acc.items()))
        # Sanity: directed reports name the REALIZED slot ~89% of the time
        # (write6 audit consensus-correct); on realized divergence they fall
        # back to the instruction. slot_out is the operative target.
        sanity = min(r["random"]["acc"] for r in
                     summary["corpora"][name]["targets"]["slot_out"]["estimators"].values())
        if name == "rt_directed" and sanity < 0.80:
            print("SANITY FAIL: rt_directed below 0.99 under some estimator — "
                  "pipeline declared broken, other numbers uninterpreted",
                  file=sys.stderr)
            sys.exit(2)
    Path(__file__).with_name("summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote summary.json")


if __name__ == "__main__":
    main()
