"""Scoped source encoders and native recipients for description carriers."""
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
    """FP32 q/v factors enabled inside the declared source or task context.

    The backbone remains outside this module's parameter tree. Creating the
    adapter installs inactive projection hooks. close() removes those hooks.
    """
    def __init__(self, host, *, rank=16, last_layers=4, seed=47, scale=1.,
                 layer_indices=None, scope="task"):
        super().__init__()
        if scope not in {"task", "source_encoding"}:
            raise ValueError("Unknown adapter scope")
        registry = getattr(host, "_scoped_adapters", {})
        if scope in registry or (scope == "task" and getattr(host, "_recipient_adapter", None) is not None):
            raise ValueError("An adapter for this scope is already installed")
        if any(p.requires_grad for p in host.model.parameters()):
            raise ValueError("The recipient requires a frozen backbone")
        if not 1 <= last_layers <= host.depth or rank < 1:
            raise ValueError("Invalid adapter layer count or rank")
        selected = list(range(host.depth-last_layers, host.depth)) if layer_indices is None else list(layer_indices)
        if len(selected) != last_layers or selected != sorted(set(selected)) or any(not isinstance(i, int) or i < 0 or i >= host.depth for i in selected):
            raise ValueError("Invalid explicit adapter layers")
        if scope == "task" and any(i < host.cut for i in selected):
            raise ValueError("Recipient projections must be at or after the injection cut")
        if scope == "source_encoding" and any(i >= host.cut for i in selected):
            raise ValueError("Source encoder projections must precede the injection cut")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid recipient scale")
        self.host = host
        self.base_host_sha256 = digest(host.identity)
        self.rank, self.last_layers, self.seed, self.scale = rank, last_layers, seed, float(scale)
        self.scope = scope
        self.layer_indices = None if layer_indices is None else selected
        self.factors = torch.nn.ModuleList()
        self.targets = []
        self._handles = []
        self._active = False
        self._closed = False
        projections = []
        for layer_index in selected:
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
        host._scoped_adapters = registry
        registry[scope] = self
        if scope == "task":
            host._recipient_adapter = self

    @property
    def architecture(self):
        architecture = {"rank": self.rank, "last_layers": self.last_layers, "seed": self.seed,
                        "scale": self.scale, "targets": self.targets, "parameter_dtype": "float32",
                        "scope": "explicit_task_context_only"}
        if self.layer_indices is not None or self.scope != "task":
            architecture.update(layer_indices=[t["layer"] for t in self.targets[::2]],
                                adapter_scope=self.scope,
                                scope="explicit_" + self.scope + "_context_only")
        return architecture

    @contextmanager
    def active(self):
        if self._closed:
            raise ValueError("The recipient adapter is closed")
        if any(other is not self and other._active for other in getattr(self.host, "_scoped_adapters", {}).values()):
            raise ValueError("Another adapter scope is active")
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
        registry = getattr(self.host, "_scoped_adapters", {})
        if registry.get(self.scope) is self:
            del registry[self.scope]


def batched_recipient_logits(host, adapter, input_ids, attention_mask, slots, input_positions, score_positions):
    """Return native [batch, vocabulary] logits at causal prediction positions.

    score_positions points to the token immediately preceding the answer.
    Caller gradients through slots are retained for a later update experiment.
    Padded positions count toward the processed-token meter.
    """
    if adapter.host is not host or adapter._closed or adapter.scope != "task":
        raise ValueError("Recipient adapter belongs to another host or scope, or is closed")
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
    kind = "scoped_recipient_qv_lora" if adapter.scope == "task" else "scoped_source_encoder_qv_lora"
    metadata = {"kind": kind, "base_host_sha256": adapter.base_host_sha256,
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
        if metadata["kind"] not in {"scoped_recipient_qv_lora", "scoped_source_encoder_qv_lora"} or metadata["base_host_sha256"] != digest(host.identity):
            raise ValueError("Recipient artifact belongs to another host")
        architecture = metadata["architecture"]
        adapter = RecipientAdapter(host, rank=architecture["rank"], last_layers=architecture["last_layers"],
                                   seed=architecture["seed"], scale=architecture["scale"],
                                   layer_indices=architecture.get("layer_indices"),
                                   scope=architecture.get("adapter_scope", "task"))
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


class _SourceCaptureComplete(Exception):
    """Internal signal to finish the source prefix before decoder cut execution."""


def differentiable_source_slots(host, encoder_adapter, text, *, forward_length=256):
    """Encode one current source, stopping at the pre-cut state with gradients intact.

    This interface accepts no task, answer label, trajectory, or target state.
    Prefix execution costs are counted as one forward and forward_length input
    tokens, with the number of executed decoder layers recorded in its layout.
    """
    from .carrier_probe import description_layout
    if encoder_adapter.host is not host or encoder_adapter._closed or encoder_adapter.scope != "source_encoding":
        raise ValueError("Need a source-encoding adapter for this host")
    if host.model.training or any(p.requires_grad for p in host.model.parameters()):
        raise ValueError("Need an eval-mode frozen source backbone")
    if getattr(host.model, "is_gradient_checkpointing", False):
        raise ValueError("Scoped source encoding requires gradient checkpointing disabled")
    ids, positions, layout = description_layout(host, text, "workspace", forward_length=forward_length)
    host._budget(len(ids))
    captured = None
    def capture(module, arguments, keywords):
        nonlocal captured
        hidden = arguments[0] if arguments else keywords["hidden_states"]
        captured = hidden[:, positions, :].clone()
        raise _SourceCaptureComplete()
    hook = host.layers[host.cut].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with encoder_adapter.active():
            inputs = torch.tensor([ids], device=host.device)
            try:
                host.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, return_dict=True)
            except _SourceCaptureComplete:
                pass
        if captured is None or captured.shape != (1, host.slots, host.width):
            raise ValueError("Source prefix did not reach its capture boundary")
        if not torch.isfinite(captured).all():
            raise FloatingPointError("Nonfinite source prefix state")
        host.meter.forwards += 1
        host.meter.processed_tokens += len(ids)
        return captured[0], {**layout, "source_prefix_layers_executed": host.cut,
                             "native_output_head_executed": False}
    finally:
        hook.remove()
