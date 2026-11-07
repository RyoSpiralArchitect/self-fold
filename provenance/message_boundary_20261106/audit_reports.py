"""Additive parser cross: re-decode saved reports under multiple decoders.

Reads reports_*.jsonl from write2/write3 produce dirs and prints per-report
bit correctness vs true mask bit0 under decode_bit (global) and decode_state
(rowbound_v1). Writes no files; historical summaries are never overwritten.

Usage: PYTHONPATH=src python3 provenance/message_boundary_20261106/audit_reports.py
"""
import json
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from selffold.lm_writer import decode_bit, decode_state  # noqa: E402

GROUPS = json.loads((ROOT / "data/within_topic_wording_20261103.json").read_text())["groups"]

for run in ("write2", "write3"):
    try:
        fit = json.loads((ROOT / f"provenance/message_boundary_20261106/{run}/produce/fit_keys.json").read_text())["keys"]
        test = json.loads((ROOT / f"provenance/message_boundary_20261106/{run}/produce/test_keys.json").read_text())["keys"]
    except Exception:
        continue
    keys = fit + test
    for path in sorted(glob.glob(str(ROOT / f"provenance/message_boundary_20261106/{run}/produce/reports_*.jsonl"))):
        style = path.split("reports_")[-1].split(".")[0]
        stats = {"global": [0, 0, 0], "rowbound": [0, 0, 0]}
        for line in open(path):
            r = json.loads(line)
            k = keys[r["source"]]
            opts, pole, tb = GROUPS[k[0]]["options"], k[3], r["mask"] & 1
            for tag, fn in (("global", decode_bit), ("rowbound", decode_state)):
                b, m = fn(r["text"], opts, pole)
                stats[tag][2 if m else (0 if b == tb else 1)] += 1
        print(f"{run} {style:20s} global={stats['global']} rowbound={stats['rowbound']}")
