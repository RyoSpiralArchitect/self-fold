"""Question-independent maps from E rows into a frozen lexical interface."""

import json
from pathlib import Path

import numpy as np
import torch

from .codec_learning import EncoderBridge
from .records import digest, file_sha


class SourceLexicalMapper(torch.nn.Module):
    """Train-only centered ridge map with one zero-initialized rank-16 residual.

    A call receives source rows alone. Questions and source identifiers have no
    representation in this interface. Stored training anchors implement the
    fitted linear ridge map, rather than a lookup by case identifier.
    """

    def __init__(
        self,
        source_mean,
        source_scale,
        anchors,
        coefficients,
        teacher_mean,
        *,
        slots,
        width,
        rank=16,
    ):
        super().__init__()
        self.slots, self.width, self.rank = slots, width, rank
        features = slots * width
        buffers = {
            "source_mean": source_mean,
            "source_scale": source_scale,
            "anchors": anchors,
            "coefficients": coefficients,
            "teacher_mean": teacher_mean,
        }
        for name, values in buffers.items():
            self.register_buffer(name, values.detach().clone().float())
        if (
            source_mean.shape != (features,)
            or source_scale.shape != (features,)
            or teacher_mean.shape != (features,)
            or anchors.ndim != 2
            or anchors.shape[1] != features
            or coefficients.shape != anchors.shape
            or not 1 <= rank <= min(features, anchors.shape[0])
        ):
            raise ValueError("Invalid source mapper geometry")
        if any(not torch.isfinite(x).all() for x in buffers.values()) or not torch.all(
            source_scale > 0
        ):
            raise ValueError("Invalid source mapper statistics")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(47)
            self.down = torch.nn.Parameter(torch.randn(features, rank) / features**0.5)
            self.up = torch.nn.Parameter(torch.zeros(rank, features))

    def forward(self, values):
        if values.shape[-2:] != (self.slots, self.width) or not torch.is_floating_point(
            values
        ):
            raise ValueError("Expected source rows with the fitted geometry")
        x = (
            values.float().reshape(-1, self.slots * self.width) - self.source_mean
        ) / self.source_scale
        kernel = x @ self.anchors.T / self.anchors.shape[1]
        result = (
            self.teacher_mean + kernel @ self.coefficients + (x @ self.down) @ self.up
        )
        return result.reshape(values.shape).to(values.dtype)


def fit_source_ridge(sources, teachers, *, ridge=0.001, rank=16):
    if (
        sources.shape != teachers.shape
        or sources.ndim != 3
        or sources.shape[0] < rank
        or ridge <= 0
    ):
        raise ValueError("Need paired source/teacher rows and positive ridge")
    x = sources.detach().cpu().double().flatten(1)
    y = teachers.detach().cpu().double().flatten(1)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Nonfinite fit data")
    mean = x.mean(0)
    scale = x.std(0, correction=0)
    floor = max(float(scale.mean()) * 0.01, 1e-8)
    scale = scale.clamp_min(floor)
    x = (x - mean) / scale
    teacher_mean = y.mean(0)
    gram = x @ x.T / x.shape[1]
    coefficients = torch.linalg.solve(
        gram + ridge * torch.eye(len(x), dtype=torch.float64), y - teacher_mean
    )
    mapper = SourceLexicalMapper(
        mean,
        scale,
        x,
        coefficients,
        teacher_mean,
        slots=sources.shape[1],
        width=sources.shape[2],
        rank=rank,
    )
    teacher_variance = (y - teacher_mean).square().mean().clamp_min(1e-8).item()
    return mapper, {
        "ridge": ridge,
        "source_scale_floor": floor,
        "teacher_variance": teacher_variance,
        "training_sources": len(x),
        "fit_dtype": "float64",
        "inference_dtype": "float32",
    }


def load_lexical_interface(path, host_identity, *, device):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"down", "up", "metadata_utf8"}:
            raise ValueError("Unexpected lexical artifact fields")
        metadata = json.loads(archive["metadata_utf8"].tobytes())
        down, up = archive["down"].copy(), archive["up"].copy()
    if (
        metadata["kind"] != "lexical_slot_low_rank_residual"
        or metadata["base_host_sha256"] != digest(host_identity)
        or metadata["capture_geometry"] != "matched_recipient_length_lexical_slots"
    ):
        raise ValueError("Lexical artifact host or geometry mismatch")
    if (
        down.ndim != 2
        or down.shape[0] != host_identity["width"]
        or up.shape != down.T.shape
    ):
        raise ValueError("Lexical artifact shape mismatch")
    if not np.isfinite(down).all() or not np.isfinite(up).all():
        raise ValueError("Nonfinite lexical artifact")
    module = EncoderBridge(down.shape[0], down.shape[1]).to(device)
    with torch.no_grad():
        module.down.copy_(torch.from_numpy(down).to(device))
        module.up.copy_(torch.from_numpy(up).to(device))
    return module.requires_grad_(False).eval(), metadata, file_sha(path)


def save_source_mapper(path, module, *, metadata):
    path = Path(path)
    bound = {
        **metadata,
        "kind": "centered_E_to_lexical_ridge_residual",
        "slots": module.slots,
        "width": module.width,
        "rank": module.rank,
    }
    arrays = {
        name: value.detach().float().cpu().numpy()
        for name, value in module.state_dict().items()
    }
    arrays["metadata_utf8"] = np.frombuffer(
        json.dumps(bound, sort_keys=True).encode(), dtype=np.uint8
    )
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return file_sha(path)


def load_source_mapper(path, *, host_identity, lexical_sha256, device):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_utf8"].tobytes())
        fields = {
            "source_mean",
            "source_scale",
            "anchors",
            "coefficients",
            "teacher_mean",
            "down",
            "up",
        }
        if set(archive.files) != fields | {"metadata_utf8"}:
            raise ValueError("Unexpected source mapper fields")
        if (
            metadata["kind"] != "centered_E_to_lexical_ridge_residual"
            or metadata["base_host_sha256"] != digest(host_identity)
            or metadata["lexical_sha256"] != lexical_sha256
        ):
            raise ValueError("Source mapper binding mismatch")
        arrays = {key: torch.from_numpy(archive[key].copy()) for key in fields}
    if any(not torch.isfinite(x).all() for x in arrays.values()):
        raise ValueError("Nonfinite source mapper")
    module = SourceLexicalMapper(
        *(
            arrays[k]
            for k in (
                "source_mean",
                "source_scale",
                "anchors",
                "coefficients",
                "teacher_mean",
            )
        ),
        slots=metadata["slots"],
        width=metadata["width"],
        rank=metadata["rank"],
    )
    module.load_state_dict(arrays, strict=True)
    return module.to(device).requires_grad_(False).eval(), metadata, file_sha(path)
