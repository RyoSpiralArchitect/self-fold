"""Fresh-process verification of saved carried-reference results; no fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from selffold.reference_evaluation import gate_results, logical_hash, summarize_panels
from selffold.report_replay import read, restore_native_rows, verify_files
from fit_role_reference import reduce_pairs

ROOT = Path(__file__).resolve().parents[1]


def reduce(run):
    design = read(ROOT / "data/role_reference_20261110.json")
    cfg = read(ROOT / design["base_config"])
    manifest = read(ROOT / design["inputs_manifest"])
    verify_files(ROOT, manifest["files"])
    old = ROOT / manifest["archive"]
    keys = read(old / "test_keys.json")["keys"]
    _, top, labels = restore_native_rows(old / "test_native_hard", keys, cfg["native_reader_sha256"])
    entries = read(run / "trajectory_files.json")
    verify_files(run, {p: d["sha256"] for p, d in entries.items()})
    summaries = {}
    for path, record in entries.items():
        with np.load(run / path, allow_pickle=False) as data:
            if set(data.files) != set(record["arrays"]):
                raise ValueError("Array manifest changed")
            for key, expected in record["arrays"].items():
                value = data[key]
                if (list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]
                        or logical_hash(value) != expected["c_order_sha256"]):
                    raise ValueError(f"Logical array changed: {path}/{key}")
            if not bool(data["reference_carried_unchanged"]):
                raise ValueError("Reference carry was not verified")
            summaries[Path(path).stem] = summarize_panels(data, keys, cfg, top, labels)
    pairs = reduce_pairs(run, summaries, cfg, dict(keys=keys, top=top, labels=labels))
    gate = gate_results(design, cfg, summaries, pairs)
    if summaries != read(run / "summary.json") or pairs != read(run / "paired_reference.json") or gate != read(run / "gate.json"):
        raise ValueError("Fresh reduction differs")
    return dict(schema="carried_reference_reduction_v1", evaluations=len(summaries),
                panels=len(summaries)*3, paired_conditions=len(pairs), exact=True,
                logical_array_hashes_verified=True, generation_calls=0, new_native_calls=0,
                qualification_loaded=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    torch.set_default_device("cpu")
    torch.set_num_threads(1)
    print(json.dumps(reduce(args.run), sort_keys=True))
