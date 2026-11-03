"""Compact within-domain readout with shared policy teachers."""

import json
from pathlib import Path

import numpy as np
import torch

from .codec_learning import EncoderBridge
from .records import digest, file_sha


class PolicyMixture(torch.nn.Module):
    """Map one E tensor to a mixture of lexical teachers without a style or topic ID."""

    def __init__(self, mean, scale, projection, bias, teachers):
        super().__init__()
        classes, slots, width = teachers.shape
        features = slots * width
        if (
            mean.shape != (features,)
            or scale.shape != mean.shape
            or projection.shape != (features, classes)
            or bias.shape != (classes,)
            or min(classes, slots, width) < 2
            or not torch.all(scale > 0)
        ):
            raise ValueError("Invalid compact policy geometry")
        for name, value in dict(
            mean=mean, scale=scale, projection=projection, bias=bias, teachers=teachers
        ).items():
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite policy buffer")
            self.register_buffer(name, value.detach().clone().float())
        self.calibration = torch.nn.Parameter(torch.zeros(classes, classes))
        self.renderer = EncoderBridge(width, min(4, width))

    def weights(self, values):
        if values.shape[-2:] != self.teachers.shape[-2:]:
            raise ValueError("E geometry differs from the trained map")
        flat = values.float().reshape(-1, self.mean.numel())
        scores = ((flat - self.mean) / self.scale) @ self.projection + self.bias
        return (scores + scores @ self.calibration).softmax(-1)

    def forward(self, values):
        lexical = (self.weights(values) @ self.teachers.flatten(1)).reshape(
            values.shape
        )
        return self.renderer(lexical.to(values.dtype))


def fit_policy_directions(states, labels, teachers):
    """Shrink within-policy variation using only training states, then keep class directions."""
    if states.ndim != 3 or teachers.ndim != 3 or states.shape[1:] != teachers.shape[1:]:
        raise ValueError("Expected paired source and teacher row geometries")
    x = states.detach().cpu().double().flatten(1)
    y = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    classes = len(teachers)
    if len(y) != len(x) or set(y.tolist()) != set(range(classes)):
        raise ValueError("Incomplete policy bank")
    counts = torch.bincount(y)
    if len(set(counts.tolist())) != 1 or counts.min() < 2:
        raise ValueError("Need a balanced fit with repeated source realizations")
    mean = x.mean(0)
    scale = x.std(0, correction=0)
    scale = scale.clamp_min(max(scale.mean().item() * 0.01, 1e-8))
    x = (x - mean) / scale
    centers = torch.stack([x[y == i].mean(0) for i in range(classes)])
    residuals = x - centers[y]
    shrinkage = max(residuals.square().mean().item(), 1e-8)
    gram = residuals @ residuals.T + len(x) * shrinkage * torch.eye(
        len(x), dtype=x.dtype
    )
    directions = centers.T - residuals.T @ torch.linalg.solve(
        gram, residuals @ centers.T
    )
    # Centering the class columns removes the common softmax coordinate.
    biases = -0.5 * (centers.T * directions).sum(0)
    directions = directions - directions.mean(1, keepdim=True)
    biases = biases - biases.mean()
    distances = centers @ directions + biases
    rivals = (
        distances.masked_fill(torch.eye(classes, dtype=torch.bool), -torch.inf)
        .max(1)
        .values
    )
    temperature = (distances.diag() - rivals).median().item() / 6.0
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Training class means do not define a positive scale")
    module = PolicyMixture(
        mean, scale, directions / temperature, biases / temperature, teachers.cpu()
    )
    variance = teachers.double().var(0, correction=0).mean().clamp_min(1e-8).item()
    return module, {
        "training_sources": len(x),
        "sources_per_policy": counts.tolist(),
        "policy_count": classes,
        "contrast_dimension_at_most": classes - 1,
        "shrinkage": shrinkage,
        "temperature": temperature,
        "teacher_variance": variance,
        "trainable_parameters": sum(p.numel() for p in module.parameters()),
    }


def save_policy_mixture(path, model, metadata):
    arrays = {
        name: value.detach().float().cpu().numpy()
        for name, value in model.state_dict().items()
    }
    metadata = metadata | {
        "kind": "within_topic_policy_mixture",
        "shape": list(model.teachers.shape),
    }
    arrays["metadata_utf8"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8
    )
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return file_sha(path)


def load_policy_mixture(path, host_identity, lexical_sha, device):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(archive["metadata_utf8"].tobytes())
        if (
            meta["kind"] != "within_topic_policy_mixture"
            or meta["base_host_sha256"] != digest(host_identity)
            or meta["lexical_sha256"] != lexical_sha
            or meta["shape"][1:] != [host_identity["slots"], host_identity["width"]]
        ):
            raise ValueError("Policy mixture binding mismatch")
        arrays = {
            k: torch.from_numpy(archive[k].copy())
            for k in archive.files
            if k != "metadata_utf8"
        }
    if any(not torch.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Nonfinite policy artifact")
    model = PolicyMixture(
        *(arrays[k] for k in ("mean", "scale", "projection", "bias", "teachers"))
    )
    if list(model.teachers.shape) != meta["shape"]:
        raise ValueError("Policy bank shape mismatch")
    model.load_state_dict(arrays, strict=True)
    return model.to(device).requires_grad_(False).eval(), meta, file_sha(path)
