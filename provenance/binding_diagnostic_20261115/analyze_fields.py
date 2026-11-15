"""Field-confusion decomposition of the supported_g fresh-wording miss.

Experiment D (offline half; see PLAN.md). Decomposes the sealed field-reader
predictions in supported_g_20261104/{structured,readout}/states.npz into
per-field confusion matrices crossed with wording style, clause order, and
topic — no model, no generation.

Usage: python3 provenance/binding_diagnostic_20261115/analyze_fields.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIDES = ("left", "right")
VALUES = ("off", "on")


def lamp(i):
    return "none" if i == 0 else f"{SIDES[(i - 1) // 2]}-{VALUES[(i - 1) % 2]}"


def trigger_name(i):
    return f"{SIDES[i // 2]}-{VALUES[i % 2]}"


def analyze(route):
    f = np.load(ROOT / f"provenance/supported_g_20261104/{route}/states.npz")
    keys, fb, gd, fa = f["keys"], f["field_base"], f["guard"], f["field_actions"]
    tb, tg = keys[:, 1], keys[:, 2] + 1  # trigger id; guard = veto + 1
    style, order, topic = keys[:, 4], keys[:, 5], keys[:, 0]
    out = {"n": len(keys), "fields": {}}
    for name, pred, true, namer in (
        ("trigger", fb, tb, trigger_name),
        ("veto", gd, tg, lamp),
    ):
        err = pred != true
        conf = Counter(
            (namer(t), namer(p)) for t, p in zip(true[err], pred[err]))
        by = {}
        for s in (0, 1):
            for o in (0, 1):
                m = (style == s) & (order == o)
                by[f"style{s}_order{o}"] = [int(err[m].sum()), int(m.sum())]
        out["fields"][name] = dict(
            acc=float((~err).mean()), errors=int(err.sum()),
            confusions={f"{a}->{b}": c for (a, b), c in conf.items()},
            by_cell=by)
    # Action field pair (order-independent against the menu's two actions)
    tset = np.stack([2 * topic, 2 * topic + 1], 1)
    aerr = ~(np.sort(fa, 1) == tset).all(1)
    lamp_err = (fb != tb) | (gd != tg)
    out["fields"]["actions"] = dict(
        acc=float((~aerr).mean() * 1.0), errors=int(aerr.sum()),
        by_style={str(s): [int(aerr[style == s].sum()), int((style == s).sum())]
                  for s in (0, 1)},
        by_topic={str(t): [int(aerr[topic == t].sum()), int((topic == t).sum())]
                  for t in range(8)},
        overlap_with_lamp_err=int((aerr & lamp_err).sum()),
        action_only=int((aerr & ~lamp_err).sum()))
    # no-veto (stable) field reads
    stable = keys[:, 2] == -1
    out["stable"] = dict(
        n=int(stable.sum()),
        guard_err=int((gd[stable] != tg[stable]).sum()),
        trigger_err=int((fb[stable] != tb[stable]).sum()),
        action_err=int(aerr[stable].sum()))
    return out


def main():
    summary = {r: analyze(r) for r in ("structured", "readout")}
    out = Path(__file__).resolve().parent / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    for route, s in summary.items():
        print(f"== {route} n={s['n']}")
        for name, f_ in s["fields"].items():
            print(f"  {name}: acc={f_['acc']:.4f} errors={f_['errors']} "
                  f"confusions={f_.get('confusions')}")
        print(f"  stable (veto -1): {s['stable']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
