#!/usr/bin/env python3
"""Fit an explicitly supplied calibration mapping. This does not pass the semantic gate."""
import argparse
from pathlib import Path
import numpy as np
import torch
from selffold.transport import fit_ridge
p=argparse.ArgumentParser()
p.add_argument("pairs", help="NPZ with x,y matched [n,d] calibration states, never test tasks")
p.add_argument("out")
p.add_argument("--ridge", type=float, default=1e-3)
a=p.parse_args()
if Path(a.out).exists():
    raise FileExistsError(a.out)
with np.load(a.pairs, allow_pickle=False) as z:
    t=fit_ridge(torch.from_numpy(z["x"].copy()),torch.from_numpy(z["y"].copy()),a.ridge)
np.savez_compressed(a.out, weight=t.weight.numpy(), bias=t.bias.numpy())
print("CALIBRATION_FIT_ONLY; semantic qualification remains pending")
