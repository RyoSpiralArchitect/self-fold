"""A scoped native-output recipient for frozen description carriers."""
from __future__ import annotations
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import torch
from .records import digest, file_sha


class _LowRankProjection(torch.nn.Module):
    def __init__(self, inputs, outputs, rank):
        super().__init__()
        self.down = torch.nn.Parameter(torch.randn(inputs, rank) / inputs**.5)
        self.up = torch.nn.Parameter(torch.zeros(rank, outputs))

    def forward(self, values):
        return values.float() @ self.down @ self.up


class RecipientAdapter(torch.nn.Module):
    """FP32 q/v factors enabled only inside an explicit task context.

    The backbone remains outside this module's parameter tree. Creating the
    adapter installs inactive projection hooks. close() removes those hooks.
    """
    def __init__(self, host, *, rank=16, last_layers=4, seed=47, scale=1.):
        super().__init__()
        if getattr(host, "_recipient_adapter", None) is not None:
            raise ValueError("A recipient adapter is already installed")
        if any(p.requires_grad for p in host.model.parameters()):
            raise ValueError("The recipient requires a frozen backbone")
        if not 1 <= last_layers <= host.depth - host.cut or rank < 1:
            raise ValueError("Recipient projections must be at or after the injection cut")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid recipient scale")
        self.host = host
        self.base_host_sha256 = digest(host.identity)
        self.rank, self.last_layers, self.seed, self.scale = rank, last_layers, seed, float(scale)
        self.factors = torch.nn.ModuleList()
        self.targets = []
        self._handles = []
        self._active = False
        self._closed = False
        projections = []
        for layer_index in range(host.depth-last_layers, host.depth):
            attention = host.layers[layer_index].self_attn
            for name in ("q_proj", "v_proj"):
                projection = getattr(attention, name, None)
                if not isinstance(projection, torch.nn.Linear) or rank > min(projection.in_features, projection.out_features):
                    raise ValueError("Unsupported recipient projection or rank")
                projections.append(projection)
                self.targets.append({"layer": layer_index, "name": name,
                                     "inputs": projection.in_features, "outputs": projection.out_features})
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for target in self.targets:
                self.factors.append(_LowRankProjection(target["inputs"], target["outputs"], rank))
        self.to(device=host.device, dtype=torch.float32)
        try:
            for index, projection in enumerate(projections):
                def add_delta(module, arguments, output, index=index):
                    if not self._active:
                        return output
                    delta = self.factors[index](arguments[0]) * self.scale
                    return output + delta.to(output.dtype)
                self._handles.append(projection.register_forward_hook(add_delta))
        except Exception:
            self.close()
            raise
        host._recipient_adapter = self

    @property
    def architecture(self):
        return {"rank": self.rank, "last_layers": self.last_layers, "seed": self.seed,
                "scale": self.scale, "targets": self.targets, "parameter_dtype": "float32",
                "scope": "explicit_task_context_only"}

    @contextmanager
    def active(self):
        if self._closed:
            raise ValueError("The recipient adapter is closed")
        previous = self._active
        self._active = True
        try:
            yield self
        finally:
            self._active = previous

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = False
        self._closed = True
        if getattr(self.host, "_recipient_adapter", None) is self:
            self.host._recipient_adapter = None


def batched_recipient_logits(host, adapter, input_ids, attention_mask, slots, input_positions, score_positions):
    """Return native [batch, vocabulary] logits at causal prediction positions.

    score_positions points to the token immediately preceding the answer.
    Caller gradients through slots are retained for a later update experiment.
    Padded positions count toward the processed-token meter.
    """
    if adapter.host is not host or adapter._closed:
        raise ValueError("Recipient adapter belongs to another or closed host")
    if any(p.requires_grad for p in host.model.parameters()):
        raise ValueError("Backbone parameters must remain frozen")
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=host.device)
    mask = torch.as_tensor(attention_mask, dtype=torch.long, device=host.device)
    score_at = torch.as_tensor(score_positions, dtype=torch.long, device=host.device)
    if ids.ndim != 2 or mask.shape != ids.shape or not 0 < ids.shape[1] <= host.max_context:
        raise ValueError("Invalid batched recipient context")
    batch, tokens = ids.shape
    if batch < 1 or score_at.shape != (batch,) or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Invalid batched recipient masks or score positions")
    if torch.any(score_at < 0) or torch.any(score_at >= tokens):
        raise ValueError("Recipient score position outside the context")
    row_ids = torch.arange(batch, device=host.device)
    if not torch.all(mask[row_ids, score_at] == 1):
        raise ValueError("Cannot score a masked token")
    positions = None
    if slots is not None:
        if slots.shape != (batch, host.slots, host.width) or not torch.is_floating_point(slots) or not torch.isfinite(slots).all():
            raise ValueError("Invalid batched source slots")
        positions = torch.as_tensor(input_positions, dtype=torch.long, device=host.device)
        if positions.shape != (batch, host.slots) or torch.any(positions < 0) or torch.any(positions >= tokens):
            raise ValueError("Invalid batched slot positions")
        if any(len(set(row.tolist())) != host.slots for row in positions):
            raise ValueError("Duplicate recipient slot position")
        if not torch.all(mask[row_ids[:, None], positions] == 1):
            raise ValueError("Cannot inject into masked tokens")
        if torch.any(positions > score_at[:, None]):
            raise ValueError("Source slots must precede the scored answer")
    elif input_positions is not None and torch.as_tensor(input_positions).numel():
        raise ValueError("Slot positions require source slots")
    host._budget(batch*tokens)
    hook = None
    try:
        if slots is not None:
            def inject(module, arguments, keywords):
                hidden = arguments[0] if arguments else keywords["hidden_states"]
                hidden = hidden.clone()
                hidden[row_ids[:, None], positions, :] = slots.to(device=hidden.device, dtype=hidden.dtype)
                if arguments:
                    return (hidden, *arguments[1:]), keywords
                return arguments, {**keywords, "hidden_states": hidden}
            hook = host.layers[host.cut].register_forward_pre_hook(inject, with_kwargs=True)
        with adapter.active():
            output = host.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
        if getattr(output, "past_key_values", None) is not None:
            raise RuntimeError("Recipient returned a forbidden KV cache")
        logits = output.logits[row_ids, score_at, :].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite recipient logits")
        host.meter.forwards += 1
        host.meter.processed_tokens += batch*tokens
        return logits
    finally:
        if hook is not None:
            hook.remove()


def recipient_loss(host, adapter, input_ids, attention_mask, slots, input_positions, score_positions, gold_ids):
    if torch.is_inference_mode_enabled():
        raise ValueError("Recipient learning needs an enabled gradient path")
    logits = batched_recipient_logits(host, adapter, input_ids, attention_mask, slots, input_positions, score_positions)
    gold = torch.as_tensor(gold_ids, dtype=torch.long, device=host.device)
    if gold.shape != (logits.shape[0],) or torch.any(gold < 0) or torch.any(gold >= logits.shape[-1]):
        raise ValueError("Invalid recipient target labels")
    return torch.nn.functional.cross_entropy(logits, gold)


def save_recipient(path, adapter, host_identity, *, dataset_sha256, source_sha256, training):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if adapter._closed or digest(host_identity) != adapter.base_host_sha256:
        raise ValueError("Recipient base host binding differs")
    for value in (dataset_sha256, source_sha256):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Need exact recipient data/source bindings")
    metadata = {"kind": "scoped_recipient_qv_lora", "base_host_sha256": adapter.base_host_sha256,
                "architecture": adapter.architecture, "dataset_sha256": dataset_sha256,
                "source_sha256": source_sha256, "training": training}
    arrays = {name: parameter.detach().float().cpu().numpy() for name, parameter in adapter.named_parameters()}
    arrays["metadata_utf8"] = np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return file_sha(path)


def load_recipient(path, host, *, trainable=False):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_utf8"].tobytes())
        if metadata["kind"] != "scoped_recipient_qv_lora" or metadata["base_host_sha256"] != digest(host.identity):
            raise ValueError("Recipient artifact belongs to another host")
        architecture = metadata["architecture"]
        adapter = RecipientAdapter(host, rank=architecture["rank"], last_layers=architecture["last_layers"],
                                   seed=architecture["seed"], scale=architecture["scale"])
        try:
            if adapter.architecture != architecture:
                raise ValueError("Recipient architecture differs")
            parameters = dict(adapter.named_parameters())
            if set(archive.files) != set(parameters) | {"metadata_utf8"}:
                raise ValueError("Unexpected recipient artifact arrays")
            with torch.no_grad():
                for name, parameter in parameters.items():
                    value = archive[name]
                    if value.shape != tuple(parameter.shape) or value.dtype != np.float32 or not np.isfinite(value).all():
                        raise ValueError("Invalid recipient artifact parameter")
                    parameter.copy_(torch.from_numpy(value.copy()).to(host.device))
            adapter.requires_grad_(trainable)
            adapter.eval()
            return adapter, metadata, file_sha(Path(path))
        except Exception:
            adapter.close()
            raise
