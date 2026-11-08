"""Writer/parser separation audit over saved report corpora.

Re-decodes every saved report under several decoders and a normalized
unique-form view, so "the writer never wrote the bit" separates from "the
parser could not read a written bit". Ground truth per record:

- rt_* styles: the realized out pattern's off/off bit (out & 1) — the report
  is about the actual post-update table.
- other styles: the requested mask's off/off bit (mask & 1).

Attribution per report (writer failure if no decoder finds the bit):
- all_missing: every decoder reports missing -> nothing written to read.
- all_wrong: every decoder binds the other option -> a written-but-wrong or
  systematically ambiguous report; still decoder-independent.
- split: decoders disagree -> parser-attributable (the bit was readable by
  at least one reasonable parser).

Unique-form view: normalize each text (option names -> <A>/<B>, coordinates
-> <ROW>, casefold) and count distinct forms among each outcome class. Many
distinct failing forms = a systematic writer gap; a few repeated failing
forms = parser blind spots. Writes no files; never overwrites summaries.

Usage:
  PYTHONPATH=src python3 provenance/message_boundary_20261106/audit_roundtrip.py [run ...]
"""
import json
import glob
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from selffold.lm_writer import (  # noqa: E402
    decode_bit, decode_state, decode_state_slot)

GROUPS = json.loads((ROOT / "data/within_topic_wording_20261103.json").read_text())["groups"]

COORD = re.compile(r"\b(?:off|on)\s*[/\-]\s*(?:off|on)\b", re.IGNORECASE)


def form_of(text, options):
    """Decoder-blind phrasing template: hides WHICH option/action was named."""
    t = " " + text.casefold() + " "
    for i, opt in enumerate(options):
        t = t.replace(opt.casefold(), f" <{'AB'[i]}> ")
    t = COORD.sub(" <ROW> ", t)
    return re.sub(r"\s+", " ", t).strip()


def decoders(options, pole):
    def rt(text):
        slot, miss = decode_state_slot(text, options)
        return slot ^ pole, miss
    return {
        "global": lambda t: decode_bit(t, options, pole),
        "rowbound": lambda t: decode_state(t, options, pole),
        "polefree": rt,
    }


def audit(run):
    prod = ROOT / f"provenance/message_boundary_20261106/{run}/produce"
    try:
        fit = json.loads((prod / "fit_keys.json").read_text())["keys"]
        test = json.loads((prod / "test_keys.json").read_text())["keys"]
    except Exception:
        return
    keys = fit + test
    for path in sorted(glob.glob(str(prod / "reports_*.jsonl"))):
        if Path(path).name.count(".") != 1:
            continue  # .generation.partial.jsonl and other auxiliaries
        m = re.fullmatch(r"reports_(.+)\.jsonl", Path(path).name)
        if not m:
            continue
        style = m.group(1)
        counts = Counter()
        forms = {k: Counter() for k in ("correct", "wrong", "missing")}
        by_key = Counter()
        for line in open(path):
            r = json.loads(line)
            k = keys[r["source"]] if r["source"] < len(keys) else None
            if k is None:
                # rt records are keyed by (group, pole) pair, not source index.
                opts, pole = GROUPS[r["group"]]["options"], r["pole"]
            else:
                opts, pole = GROUPS[k[0]]["options"], k[3]
            truth = (r["out"] if "out" in r else r["mask"]) & 1
            results = {name: fn(r["text"]) for name, fn in decoders(opts, pole).items()}
            bits = {n: b for n, (b, m) in results.items() if not m}
            if not bits:
                cls = "all_missing"
            elif len(set(bits.values())) == 1:
                cls = "correct" if next(iter(bits.values())) == truth else "all_wrong"
            elif any(b == truth for b in bits.values()):
                cls = "split_recoverable"
            else:
                cls = "all_wrong"
            counts[cls] += 1
            counts["n"] += 1
            f = form_of(r["text"], opts)
            per = {n: ("missing" if m else ("correct" if b == truth else "wrong"))
                   for n, (b, m) in results.items()}
            if cls == "all_missing":
                forms["missing"][f] += 1
            elif cls == "all_wrong":
                forms["wrong"][f] += 1
            else:
                for n, p in per.items():
                    forms[p][f] += 1
                by_key[cls] += 1
        n = counts["n"]
        print(f"{run} {style:20s} n={n} "
              f"all_missing={counts['all_missing']} ({counts['all_missing'] / n:.1%}) "
              f"all_wrong={counts['all_wrong']} ({counts['all_wrong'] / n:.1%}) "
              f"split_recoverable={counts['split_recoverable']} "
              f"correct={counts['correct']} ({counts['correct'] / n:.1%})")
        for cls in ("missing", "wrong"):
            c = forms[cls]
            top = ", ".join(f"{v}x {k[:60]!r}" for k, v in c.most_common(3))
            print(f"    {cls}: {len(c)} unique forms / {sum(c.values())} reports; top: {top}")


for run in sys.argv[1:] or ("write5",):
    audit(run)
