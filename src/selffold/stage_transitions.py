"""Reduced-rank ridge maps between whole question-free workspace states."""

from dataclasses import dataclass

import torch


@dataclass
class StageMap:
    center: torch.Tensor
    scale: torch.Tensor
    target: torch.Tensor
    down: torch.Tensor
    up: torch.Tensor

    def __call__(self, rows):
        x = rows.flatten(1).float()
        return self.target + ((x - self.center) / self.scale @ self.down) @ self.up

    def slots(self, rows):
        return self(rows).reshape_as(rows).to(rows.dtype)


def fit_stage_map(source, target, rank=32, ridge=0.001):
    if len(source) != len(target) or len(source) < 2 or rank < 1 or ridge <= 0:
        raise ValueError("Need aligned source and teacher banks")
    x = source.flatten(1).double().cpu()
    y = target.flatten(1).double().cpu()
    cx, cy = x.mean(0), y.mean(0)
    x = x - cx
    y = y - cy
    scale = x.square().mean().sqrt().clamp_min(1e-12)
    x = x / scale
    gram = x @ x.T / x.shape[1]
    reg = ridge * gram.diag().mean().clamp_min(1e-12)
    ev, u = torch.linalg.eigh(y @ y.T)
    threshold = max(1e-12, float(ev[-1]) * 1e-10)
    valid = torch.where(ev > threshold)[0][-rank:]
    if len(valid) == 0:
        raise ValueError("Constant teacher bank")
    up = (u[:, valid].T @ y) / ev[valid].sqrt()[:, None]
    coef = torch.linalg.solve(gram + reg * torch.eye(len(x), dtype=x.dtype), y @ up.T)
    down = x.T @ coef / x.shape[1]
    result = StageMap(*(t.float().contiguous() for t in (cx, scale, cy, down, up)))
    if not all(torch.isfinite(t).all() for t in vars(result).values()):
        raise ValueError("Nonfinite fitted map")
    return result


def target_preference(pred, reference, opposed):
    return (pred.float() - opposed.float()).flatten(1).square().mean(-1) - (
        pred.float() - reference.float()
    ).flatten(1).square().mean(-1)
