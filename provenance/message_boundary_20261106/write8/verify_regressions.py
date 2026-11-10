"""Read-only mutation probes for the two PR #35 regressions.

Run each real fixture before changing behavior in memory, then require the
fixture to reject that substitution. No source files or experiment data change.
"""
import inspect
import json
import runpy
from pathlib import Path
from unittest.mock import patch

import torch

from selffold import mixture_boundary
from selffold.message_boundary import BoundaryMemory

ROOT = Path(__file__).resolve().parents[3]


def must_reject(test):
    try:
        test()
    except AssertionError:
        return True
    raise RuntimeError("Regression fixture accepted the deliberate defect")


def verify():
    torch.set_num_threads(1)
    fixtures = runpy.run_path(str(ROOT / "tests/test_roundtrip.py"))
    full_path = fixtures["test_perfect_text_feature_channel_completes_all_four_edits"]
    actual_out = fixtures["test_rollout_rt_uses_out_axis_not_requested_mask"]
    full_path()
    actual_out()

    def zero_start(self, batch, device=None):
        return torch.zeros(batch, self.feat_dim, device=device)

    with patch.object(BoundaryMemory, "initial_message", zero_start):
        start_rejected = must_reject(full_path)

    old = "rt_bits[cases, prior_p, cur, pattern_of(w2)]"
    source = inspect.getsource(mixture_boundary.rollout_rt)
    if source.count(old) != 1:
        raise RuntimeError("Actual-output lookup changed; review the mutation")
    namespace = dict(mixture_boundary.__dict__)
    exec(compile(source.replace(old, "rt_bits[cases, prior_p, cur, cur]"),
                 "<requested-output mutation>", "exec"), namespace)
    with patch.object(mixture_boundary, "rollout_rt", namespace["rollout_rt"]):
        out_rejected = must_reject(actual_out)
    return dict(schema="pr35_regression_mutations_v1", fixtures_pass_before_mutation=True,
                legacy_zero_feature_start_rejected=start_rejected,
                requested_output_lookup_rejected=out_rejected,
                source_files_modified=False, generation_calls=0, new_native_calls=0)


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
