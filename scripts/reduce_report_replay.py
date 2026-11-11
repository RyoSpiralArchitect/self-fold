"""Read-only fresh-process reduction of a sealed saved-report refit."""
import argparse
import json
from pathlib import Path

import numpy as np

from selffold.report_replay import read, reduce_compact, restore_native_rows, verify_files
from selffold.role_trials import donor_indices

ROOT = Path(__file__).resolve().parents[1]


def reduce(run):
    design = read(ROOT / "data/report_replay_20261110.json")
    cfg = read(ROOT / design["base_config"])
    manifest = read(ROOT / design["inputs_manifest"])
    verify_files(ROOT, manifest["files"])
    old = ROOT / manifest["archive"]
    keys = read(old / "test_keys.json")["keys"]
    _, top, labels = restore_native_rows(old / "test_native_hard", keys,
                                         cfg["native_reader_sha256"])
    result = {}
    for mode in ("archived", "refit"):
        files = read(run / f"{mode}_files.json")
        verify_files(run, files)
        expected = read(run / f"{mode}_summary.json")["arms"]
        count = 0
        for arm, seeds in expected.items():
            for seed, metrics in seeds.items():
                path = run / mode / f"{arm}_{seed}.npz"
                with np.load(path, allow_pickle=False) as f:
                    got = reduce_compact(f, keys, cfg["fit_masks"] + cfg["test_masks"], labels,
                                         top, donor_indices(keys), cfg["test_masks"], cfg["fit_topics"])
                if got != metrics:
                    raise ValueError(f"Saved reduction differs: {mode}/{arm}/{seed}")
                count += 1
        result[mode] = dict(exact=True, controllers=count)
    return dict(schema="report_replay_reduction_v1", comparisons=result,
                generation_calls=0, new_native_calls=0, qualification_loaded=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(reduce(args.run), sort_keys=True))
