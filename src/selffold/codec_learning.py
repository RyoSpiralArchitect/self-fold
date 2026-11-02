"""A differentiable encoder bridge, isolated from the inference-only host."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from . import prompts
from .records import State, digest, file_sha


class EncoderBridge(torch.nn.Module):
    def __init__(self, width, rank=16):
        super().__init__()
        if not 1 <= rank <= width:
            raise ValueError("Invalid bridge rank")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(47)
            self.down = torch.nn.Parameter(torch.randn(width, rank) / width**.5)
            self.up = torch.nn.Parameter(torch.zeros(rank, width))

    def forward(self, values):
        return (values.float() + values.float() @ self.down @ self.up).to(values.dtype)


def codec_loss(host, bridge, source_slots, target, *, end_token_id):
    target_ids = host.tokenizer.encode(target, add_special_tokens=False) + [end_token_id]
    loss, _ = codec_objectives(host, bridge, source_slots, target_ids)
    return loss


def codec_objectives(host, bridge, source_slots, target_ids, *, contrast=None):
    """Source NLL and optional correct-minus-foil logit at a causal target position."""
    if torch.is_inference_mode_enabled() or any(p.requires_grad for p in host.model.parameters()):
        raise ValueError("Need a frozen backbone and an enabled gradient path")
    ids, locations = host.render(prompts.read_prompt())
    if not target_ids or target_ids[-1] not in host.eos_ids or len(ids) + len(target_ids) > host.max_context:
        raise ValueError("Invalid target EOS or oversized target")
    if tuple(source_slots.shape) != (host.slots, host.width):
        raise ValueError("Encoder source shape mismatch")
    full = ids + target_ids
    host._budget(len(full))
    adapted = bridge(source_slots.detach().clone())
    positions = locations[prompts.INPUT_MARKER]

    def inject(module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        hidden = hidden.clone()
        hidden[:, positions, :] = adapted[None].to(hidden.dtype)
        return ((hidden, *args[1:]), kwargs) if args else (args, {**kwargs, "hidden_states": hidden})

    hook = host.layers[host.cut].register_forward_pre_hook(inject, with_kwargs=True)
    try:
        x = torch.tensor([full], device=host.device)
        with torch.enable_grad():
            output = host.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True)
            logits = output.logits[:, len(ids)-1:len(full)-1, :].float()
            labels = torch.tensor([target_ids], device=host.device)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            margin = None
            if contrast is not None:
                index, foil = contrast
                if not 0 <= index < len(target_ids) - 1 or target_ids[index] == foil:
                    raise ValueError("Invalid source contrast")
                margin = logits[0, index, target_ids[index]] - logits[0, index, foil]
        if not torch.isfinite(loss) or (margin is not None and not torch.isfinite(margin)):
            raise FloatingPointError("Nonfinite codec loss")
        host.meter.forwards += 1
        host.meter.processed_tokens += len(full)
        return loss, margin
    finally:
        hook.remove()


def save_bridge(path, bridge, host_identity, *, dataset_sha256, source_sha256, training, capture_route="workspace"):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    for value in (dataset_sha256, source_sha256):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Need exact data/source SHA256 bindings")
    if capture_route not in {"workspace", "native_tail"}:
        raise ValueError("Unknown description-only capture route")
    metadata = {"kind": "encoder_low_rank_residual", "base_host_sha256": digest(host_identity),
                "capture_route": capture_route, "forward_length": 256,
                "dataset_sha256": dataset_sha256, "source_sha256": source_sha256, "training": training}
    with path.open("xb") as file:
        np.savez_compressed(file, down=bridge.down.detach().float().cpu().numpy(),
                            up=bridge.up.detach().float().cpu().numpy(),
                            metadata_utf8=np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8))
    return file_sha(path)


def load_bridge(path, host_identity, *, device):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"down", "up", "metadata_utf8"}:
            raise ValueError("Unexpected encoder artifact fields")
        metadata = json.loads(archive["metadata_utf8"].tobytes())
        down, up = archive["down"].copy(), archive["up"].copy()
    if metadata["kind"] != "encoder_low_rank_residual" or metadata["base_host_sha256"] != digest(host_identity):
        raise ValueError("Encoder artifact belongs to another host")
    if metadata.get("capture_route") not in {"workspace", "native_tail"} or metadata.get("forward_length") != 256:
        raise ValueError("Unknown encoder capture geometry")
    if down.ndim != 2 or up.shape != (down.shape[1], down.shape[0]) or down.shape[0] != host_identity["width"]:
        raise ValueError("Encoder artifact shape mismatch")
    if not np.isfinite(down).all() or not np.isfinite(up).all():
        raise ValueError("Nonfinite encoder artifact")
    bridge = EncoderBridge(*down.shape).to(device)
    with torch.no_grad():
        bridge.down.copy_(torch.from_numpy(down).to(device))
        bridge.up.copy_(torch.from_numpy(up).to(device))
    bridge.requires_grad_(False).eval()
    return bridge, metadata, file_sha(path)


def install_bridge(host, path):
    if getattr(host, "encoder_bridge", None) is not None:
        raise ValueError("An encoder bridge is already installed")
    bridge, metadata, sha = load_bridge(path, host.identity, device=host.device)
    def encode(text):
        from .carrier_probe import capture_description
        state, _ = capture_description(host, text, metadata["capture_route"], forward_length=metadata["forward_length"])
        with torch.no_grad():
            return State(slots=bridge(state.slots))

    host.encode = encode
    host.encoder_bridge = bridge
    host.identity.update(encoder_bridge_sha256=sha, encoder_bridge_kind=metadata["kind"],
                         encoder_capture_route=metadata["capture_route"], encoder_forward_length=256)
    return metadata
