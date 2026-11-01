"""Explicit state ownership, safe fingerprints, and immutable output directories."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import subprocess
import torch


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class State:
    """One text OR one [K,d] tensor; never a transcript or persistent KV cache."""
    text: str | None = None
    slots: torch.Tensor | None = None

    def __post_init__(self):
        if (self.text is None) == (self.slots is None):
            raise ValueError("A state must contain exactly one of text or slots")
        if self.text is not None and not self.text.strip():
            raise ValueError("Empty text is not a valid recurrence state")
        if self.slots is not None:
            if self.slots.ndim != 2 or min(self.slots.shape) < 1:
                raise ValueError("Latent state must have shape [K,d]")
            if not self.slots.is_floating_point() or not torch.isfinite(self.slots).all():
                raise ValueError("Latent state must be finite floating-point values")
            object.__setattr__(self, "slots", self.slots.detach().clone())

    def clone(self) -> State:
        return State(text=self.text, slots=self.slots)

    def sha(self) -> str:
        if self.text is not None:
            return digest({"type": "text", "value": self.text})
        a = self.slots.detach().cpu().contiguous()
        h = hashlib.sha256(canonical({"shape": list(a.shape), "dtype": str(a.dtype)}).encode())
        h.update(a.view(torch.uint8).numpy().tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class TextOutput:
    text: str
    stop: str
    token_count: int
    prompt_tokens: int | None = None
    effective_max_new_tokens: int | None = None
    stop_token_id: int | None = None

    def receipt(self) -> dict:
        return {**asdict(self), "censored": self.stop in {"length", "context_limit"}}

    def as_state(self) -> State:
        if not self.text.strip():
            raise ValueError("Decoder produced empty text; record failure, do not invent a replacement")
        return State(text=self.text)


class RunStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        # Atomic mkdir makes retries and overwriting a prior trial explicit errors.
        self.path.mkdir(parents=True, exist_ok=False)

    def write(self, name: str, value: Any):
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        tmp = target.with_suffix(target.suffix + ".partial")
        tmp.write_text(canonical(value) + "\n", encoding="utf8")
        os.replace(tmp, target)

    def append(self, name: str, value: Any):
        with (self.path / name).open("a", encoding="utf8") as f:
            f.write(canonical(value) + "\n")
            f.flush()

    def save_state(self, name: str, state: State):
        if state.text is not None:
            self.write(name + ".json", {"text": state.text, "sha256": state.sha()})
        else:
            import numpy as np
            p = self.path / (name + ".npz")
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                raise FileExistsError(p)
            # Float32 storage exactly represents BF16/FP16 values; original dtype is recorded.
            storage = state.slots.cpu() if state.slots.dtype == torch.float64 else state.slots.float().cpu()
            np.savez_compressed(p, slots=storage.numpy())
            self.write(name + ".json", {"sha256": state.sha(), "dtype": str(state.slots.dtype),
                                        "file_sha256": file_sha(p), "shape": list(state.slots.shape)})


def code_manifest(root: Path) -> dict:
    items = {}
    for folder in ("src", "configs", "data", "scripts", "tests", "docs", "paper"):
        for p in sorted((root / folder).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".yaml", ".json", ".jsonl", ".sh", ".md", ".tex", ".bib"}:
                items[str(p.relative_to(root))] = file_sha(p)
    for name in ("pyproject.toml", "README.md", "README.ja.md"):
        if (root / name).is_file():
            items[name] = file_sha(root / name)
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"files": items, "source_sha256": digest(items), "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain")}


def load_state(base: Path, *, device="cpu") -> State:
    """Restore an owned state from safe NPZ + JSON and verify bytes and original dtype."""
    import numpy as np
    base = Path(base)
    meta = json.loads(base.with_suffix(".json").read_text())
    if "text" in meta:
        state = State(text=meta["text"])
    else:
        path = base.with_suffix(".npz")
        if file_sha(path) != meta["file_sha256"]:
            raise ValueError("State file SHA mismatch")
        dtypes = {str(d): d for d in (torch.float16, torch.bfloat16, torch.float32, torch.float64)}
        if meta["dtype"] not in dtypes:
            raise ValueError("Unsupported stored dtype")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"slots"}:
                raise ValueError("Unexpected state archive fields")
            value = torch.from_numpy(archive["slots"].copy()).to(dtype=dtypes[meta["dtype"]], device=device)
        if list(value.shape) != meta["shape"]:
            raise ValueError("State shape mismatch")
        state = State(slots=value)
    if state.sha() != meta["sha256"]:
        raise ValueError("Restored state fingerprint mismatch")
    return state
