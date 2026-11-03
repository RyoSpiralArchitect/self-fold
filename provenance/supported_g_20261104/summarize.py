"""Score both development routes from their native outputs and an independent Boolean ledger."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

CHECKS = (
    "native_parity",
    "content_discrimination",
    "meaning_swap_effect",
    "one_step_update",
    "readout_fidelity",
    "positional_controls",
)


def read(path):
    return json.loads(path.read_text())


def reference(keys):
    """Bit masks in four-world order, encoded independently of the execution predicate function."""
    mask = (0b0011, 0b1100, 0b0101, 0b1010)
    w = np.arange(4)[None, :]
    trigger = ((np.array([mask[k[1]] for k in keys])[:, None] >> w) & 1).astype(bool)
    veto = (
        (np.array([0 if k[2] == -1 else mask[k[2]] for k in keys])[:, None] >> w) & 1
    ).astype(bool)
    action = np.array([k[3] for k in keys])[:, None] ^ (trigger & ~veto)
    gold = action[:, :, None] ^ np.arange(2)
    return gold, trigger & veto, veto


def summarize(study, kind):
    raw = kind == "raw-bound"
    folder = study / kind
    terminal = read(folder / "terminal.json")
    if raw:
        source = [
            json.loads(line)
            for line in (folder / "sources.jsonl").read_text().splitlines()
        ]
        keys = [tuple(x["key"]) for x in source]
        ix = {k: i for i, k in enumerate(keys)}

        def blank():
            return (
                np.full((len(keys), 4, 2), -1, np.int8),
                np.full((len(keys), 4, 2), np.nan),
                np.zeros((len(keys), 4, 2), bool),
            )

        data = {}
        gold, affected, veto = reference(keys)
        expected_question = {}
        with gzip.open(folder / "decisions.jsonl.gz", "rt") as f:
            for r in map(json.loads, f):
                k = tuple(r[:6])
                tag, w, o, y, b, token, a0, a1, q = r[6:]
                i = ix[k]
                if tag not in data:
                    data[tag] = blank()
                pred, margin, present = data[tag]
                if present[i, w, o] or y != "AB"[gold[i, w, o]]:
                    raise ValueError("Duplicate cell or broken gold")
                if tag != "question_prefix":
                    if q != expected_question.setdefault((k[0], w, o), q):
                        raise ValueError("Changed question")
                pred[i, w, o] = "AB".index(b) if b in ("A", "B") else -1
                margin[i, w, o] = np.nan if a0 is None else a0 - a1
                present[i, w, o] = True
        data["updated"] = data["boundary"]
        for tag, (_, _, seen) in data.items():
            if tag.startswith(("D_", "ED_")):
                if seen.sum() != 128:
                    raise ValueError("Wrong free-text panel")
            elif not seen.all():
                raise ValueError("Missing raw decisions")
    else:
        with np.load(folder / "states.npz", allow_pickle=False) as f:
            keys = [tuple(map(int, k)) for k in f["keys"]]
            field_diag = dict(
                base=int((f["field_base"] == np.array(keys)[:, 1]).sum()),
                guard=int((f["field_guard"] == np.array(keys)[:, 2] + 1).sum())
                if "field_guard" in f
                else int((f["guard"] == np.array(keys)[:, 2] + 1).sum()),
                first_role=int((f["field_regular"] == np.array(keys)[:, 5]).sum()),
                n=len(keys),
                update_consumes_guard=bool(np.all(f["next_guard"] == 0)),
            )
            lookup = {k: i for i, k in enumerate(keys)}
            twins = [lookup[t, b ^ 1, v, p ^ 1, s, o] for t, b, v, p, s, o in keys]
            field_diag["naturally_bit_identical_twin_tables"] = int(
                np.all(f["table"] == f["table"][twins], axis=(1, 2)).sum()
            )
        gold, affected, veto = reference(keys)
        labels = read(folder / "native.json")["labels"]
        data = {}

        def decode(top, ab):
            return (
                np.where(top == labels[0], 0, np.where(top == labels[1], 1, -1)),
                ab[..., 0] - ab[..., 1],
                np.ones(top.shape, bool),
            )

        with (
            np.load(folder / "native.npz", allow_pickle=False) as f,
            np.load(folder / "dispatch.npz", allow_pickle=False) as d,
        ):
            for tag in d.files:
                data[tag] = decode(f["top"][d[tag]], f["ab"][d[tag]])
            if not np.array_equal(d["erased"], d["twin_erased"]):
                raise ValueError("Erased identical tables diverged")
        with np.load(folder / "literal.npz", allow_pickle=False) as f:
            for tag in ("D_initial", "D_updated"):
                data[tag] = decode(f["top"][f[tag]], f["ab"][f[tag]])
    topics = np.array([k[0] for k in keys])
    unique = np.unique(topics)
    boot = np.random.default_rng(4719).integers(0, len(unique), (10000, len(unique)))

    def stat(arr, include=None):
        a = np.asarray(arr)
        m = (
            np.ones(a.shape, bool)
            if include is None
            else np.broadcast_to(include, a.shape)
        )
        if not m.any():
            return dict(n=0, score=None, macro=None, ci95=None)
        group = [
            float(a[(topics == t) & m].mean())
            if a.ndim == 1
            else float(a[topics == t][m[topics == t]].mean())
            for t in unique
            if m[topics == t].any()
        ]
        g = np.asarray(group)
        draws = (
            boot
            if len(g) == len(unique)
            else np.random.default_rng(4719).integers(0, len(g), (10000, len(g)))
        )
        return dict(
            score=int(a[m].sum()),
            n=int(m.sum()),
            macro=float(g.mean()),
            ci95=np.quantile(g[draws].mean(1), [0.025, 0.975]).tolist(),
        )

    correct = {}
    for tag, (p, m, present) in data.items():
        target = 1 - gold if tag == "opposed" else gold
        correct[tag] = (p == target) & np.isfinite(m) & (m != 0) & present
    route_scores = {tag: stat(y, data[tag][2]) for tag, y in correct.items()}
    main = correct["updated"]
    baseline = correct["identity"]
    unrelated = correct["unrelated"]
    any_change = affected.any(1)
    stable = ~any_change
    full = main.all((1, 2))
    identity_full = baseline.all((1, 2))

    def delta(x, y, mask):
        return stat(x.astype(int) - y.astype(int), mask)

    nulls = ("empty", "zero", "random", "unrelated")
    best_null = max(nulls, key=lambda k: route_scores[k]["macro"])
    change_mask = any_change[:, None, None]
    stable_mask = stable[:, None, None]
    primary = dict(
        complete_change=stat(full, any_change),
        newly_complete=stat(full & ~identity_full, any_change),
        correct_change_policies=stat(main, change_mask),
        identity_change_policies=stat(baseline, change_mask),
        unrelated_change_policies=stat(unrelated, change_mask),
        gain_over_identity=delta(main, baseline, change_mask),
        gain_over_unrelated=delta(main, unrelated, change_mask),
        correct_changed_worlds=stat(main, affected[:, :, None]),
        identity_changed_worlds=stat(baseline, affected[:, :, None]),
        already_correct_spoiled=stat(baseline & ~main, ~affected[:, :, None]),
        stable_complete=stat(full, stable),
        stable_native=stat(main, stable_mask),
        stable_new_errors=stat(baseline & ~main, stable_mask),
    )
    mechanical_update = (
        primary["complete_change"]["macro"] >= 0.8
        and primary["newly_complete"]["macro"] >= 0.7
        and min(
            primary[k]["macro"] for k in ("gain_over_identity", "gain_over_unrelated")
        )
        >= 0.1
        and primary["stable_complete"]["macro"] >= 0.9
    )
    swapped = stat(
        main & correct["opposed"] & (data["updated"][1] * data["opposed"][1] < 0)
    )
    same = stat(main & correct["equivalent"])
    content_gap = delta(baseline, correct[best_null], None)
    subset = data["D_updated"][2]
    readout = dict(
        D=route_scores["D_updated"],
        ED=route_scores["ED_updated"],
        G_same_panel=stat(main, subset),
        ED_minus_G=delta(correct["ED_updated"], main, subset),
        initial_D=route_scores["D_initial"],
        initial_ED=route_scores["ED_initial"],
    )
    positions = {
        k: dict(native=route_scores[k], difference=delta(correct[k], main, None))
        for k in ("source_prefix", "reordered", "question_prefix")
        if k in data
    }
    parity = read(folder / "parity.json")
    parity_ok = (
        (
            all(r["native"] and r["neutral"] for r in parity.values())
            and terminal["isolated"]
        )
        if raw
        else (
            parity["native"]["exact"]
            and parity["lexical"]
            and parity["unchanged_reader"]
            and parity["state_isolation"]
        )
    )

    def entry(ok, **rest):
        return dict(mechanical="pass" if ok else "fail", **rest)

    checks = {
        "native_parity": entry(parity_ok, evidence=parity),
        "content_discrimination": entry(
            route_scores["identity"]["macro"] >= 0.8 and content_gap["macro"] >= 0.1,
            E=route_scores["identity"],
            strongest_fixed_null=best_null,
            gap=content_gap,
        ),
        "meaning_swap_effect": entry(
            swapped["macro"] >= 0.8 and same["macro"] >= 0.9,
            correct_opposed_reversal=swapped,
            correct_equivalent=same,
        ),
        "one_step_update": entry(
            mechanical_update, **primary, human_support="not_assessed"
        ),
        "readout_fidelity": entry(
            min(readout[k]["macro"] for k in ("D", "ED")) >= 0.8
            and readout["ED_minus_G"]["macro"] >= -0.1,
            **readout,
            human_additions="not_assessed",
        ),
        "positional_controls": entry(
            all(
                v["native"]["macro"] >= 0.75 and v["difference"]["macro"] >= -0.1
                for v in positions.values()
            ),
            allowed=positions,
            row_permutation=route_scores["permuted"],
            norm_only=route_scores["norm_only"],
        ),
    }
    result = dict(
        interface=kind,
        n_sources=len(keys),
        topics=unique.tolist(),
        checks=checks,
        routes=route_scores,
        formal_challenge="not_opened",
        human_review="skipped",
        pilot_status="NOT_QUALIFIED",
    )
    if raw:
        outputs = [
            json.loads(l) for l in (folder / "free_text.jsonl").read_text().splitlines()
        ]
        result["generations"] = {
            s: dict(
                n=sum(r["stage"] == s for r in outputs),
                stops=dict(Counter(r["stop"] for r in outputs if r["stage"] == s)),
                distinct=len({r["text"] for r in outputs if r["stage"] == s}),
                tokens=sum(r["token_count"] for r in outputs if r["stage"] == s),
                reencoding_scored=int(np.isfinite(data["ED_" + s][1]).sum()),
            )
            for s in ("initial", "updated")
        }
    else:
        result["source_fields"] = field_diag

        def twins(first, second, mask):
            p1, m1, _ = data[first]
            p2, m2, _ = data[second]
            paired = (p1 == gold) & (p2 == np.where(veto[:, :, None], 1 - gold, gold))
            paired &= np.where(veto[:, :, None], m1 * m2 < 0, m1 * m2 > 0)
            return stat(paired, mask[:, :, None])

        intact = twins("updated", "twin", veto)
        erased = twins("erased", "twin_erased", veto)
        keep = twins("updated", "twin", ~veto)
        result["exact_table_twins"] = dict(
            veto_intact=intact,
            veto_erased=erased,
            ordinary_preserved=keep,
            mechanical_pass=(
                intact["macro"] >= 0.8
                and intact["macro"] - erased["macro"] >= 0.1
                and keep["macro"] >= 0.9
            ),
        )
        result["analytic_gap"] = delta(main, correct["analytic"], None)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("study", type=Path)
    p.add_argument("route", choices=("structured", "raw-bound"))
    a = p.parse_args()
    print(
        json.dumps(summarize(a.study, a.route), sort_keys=True, separators=(",", ":"))
    )
