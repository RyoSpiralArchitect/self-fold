"""Explicit post-writer -> pre-cut transport. Identity is an unqualified smoke route."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from .records import file_sha

@dataclass
class Transport:
    weight: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    artifact_sha256: str = "identity_unqualified"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            return x.detach().clone()
        w, b = self.weight.to(x.device, torch.float32), self.bias.to(x.device, torch.float32)
        if w.shape != (x.shape[-1], x.shape[-1]) or b.shape != (x.shape[-1],):
            raise ValueError("transport dimension mismatch")
        y = (x.float() @ w.T + b).to(x.dtype)
        if not torch.isfinite(y).all():
            raise FloatingPointError("transport produced nonfinite state")
        return y

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {"weight", "bias"}:
                raise ValueError("transport NPZ must contain only weight and bias")
            w, b = torch.from_numpy(z["weight"].copy()), torch.from_numpy(z["bias"].copy())
        if w.ndim != 2 or w.shape[0] != w.shape[1] or b.shape != (w.shape[0],):
            raise ValueError("invalid transport shapes")
        if not torch.isfinite(w).all() or not torch.isfinite(b).all():
            raise ValueError("nonfinite transport")
        return cls(w, b, file_sha(path))


def fit_ridge(x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-3) -> Transport:
    """Paired calibration only. No claim that ridge fitting qualifies semantic recurrence."""
    if x.ndim != 2 or x.shape != y.shape or x.shape[0] < 2 or ridge <= 0:
        raise ValueError("need finite, matched [observations,d] arrays and positive ridge")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("nonfinite calibration")
    x, y = x.double().cpu(), y.double().cpu()
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x-xm, y-ym
    w_t = torch.linalg.solve(xc.T @ xc + ridge*torch.eye(x.shape[1], dtype=torch.float64), xc.T @ yc)
    return Transport(w_t.T.float(), (ym - xm @ w_t).float(), "fitted_unqualified")
