"""Score an anonymous numeric packet in a fresh process without loading sources or answer keys."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_conditional_curriculum import parser
from selffold.cli import make_host
from selffold.records import RunStore, digest, file_sha
from selffold.recipient_learning import load_recipient
from selffold.state_only import OPERATIONS, PolicyUpdate
from state_only_native import NativeChoice, fingerprint


@torch.no_grad()
def decide(args, store):
    packet = args.packet
    binding = json.loads((packet / "binding.json").read_text())
    expected = {"menus.json", "reader.npz", "update.npz", "inputs.npz"}
    if set(binding["files"]) != expected or {
        p.name for p in packet.iterdir()
    } != expected | {"binding.json"}:
        raise ValueError("Unexpected packet member")
    if any(file_sha(packet / name) != sha for name, sha in binding["files"].items()):
        raise ValueError("Packet identity failed")
    with np.load(packet / "inputs.npz", allow_pickle=False) as f:
        if set(f.files) != {"requests", "states"}:
            raise ValueError("Unexpected packet field")
        requests = f["requests"]
        states = torch.from_numpy(f["states"])
    menus = json.loads((packet / "menus.json").read_text())
    if (
        requests.ndim != 2
        or requests.shape[1] != 6
        or requests.dtype != np.int32
        or not len(requests)
    ):
        raise ValueError("Invalid requests")
    if (
        states.ndim != 3
        or states.shape[1:] != (4, 16)
        or not torch.isfinite(states).all()
    ):
        raise ValueError("Invalid memory shape or values")
    if torch.any(states < 0) or not torch.allclose(
        states.sum(-1), torch.ones(states.shape[:2]), atol=1e-5
    ):
        raise ValueError("Expected distributions over actions")
    limits = [len(states), len(OPERATIONS), 4, len(menus), 2, 2]
    if np.any(requests < 0) or np.any(requests >= limits):
        raise ValueError("Address outside packet")
    host, ident = make_host(args)
    if (
        digest(host.identity) != binding["host"]
        or ident["checkpoint_sha256"] != CHECKPOINT_SHA
    ):
        raise ValueError("Fresh native host differs")
    backbone = parameter_sha(host.model)
    native = NativeChoice(host, menus)
    if native.binding != binding["native"]:
        raise ValueError("Question or lexical bank mismatch")
    update = PolicyUpdate().requires_grad_(False)
    with np.load(packet / "update.npz", allow_pickle=False) as f:
        update.load_state_dict({"routing": torch.from_numpy(f["routing"])})
    # No test-world selector runs until the entire updated policy table is materialized.
    revised = torch.stack([update(states, o) for o in range(len(OPERATIONS))])
    if not torch.isfinite(revised).all():
        raise ValueError("Invalid revised state")
    unique, calls, indices = {}, [], []
    for state, op, world, menu, order, bypass in requests.tolist():
        table = states[state] if bypass else revised[op, state]
        row = table[world]
        key = (fingerprint(row), menu, order)
        if key not in unique:
            unique[key] = len(calls)
            calls.append((row, menu, order))
        indices.append(unique[key])
    reader, _, _ = load_recipient(packet / "reader.npz", host)
    predictions, margins = [], []
    try:
        for start in range(0, len(calls), 16):
            batch = calls[start : start + 16]
            rows, topics, orders = zip(*batch)
            logits = native.logits(reader, torch.stack(rows), topics, orders)
            predictions.extend(logits.argmax(-1).cpu().tolist())
            margins.extend(logits[:, native.labels].cpu().tolist())
    finally:
        reader.close()
    np.savez_compressed(
        store.path / "answers.npz",
        dispatch=np.asarray(indices, dtype=np.int32),
        top_token=np.asarray(predictions, dtype=np.int32),
        ab_logits=np.asarray(margins, dtype=np.float32),
    )
    store.write("calls.json", list(unique))
    store.write(
        "binding.json",
        dict(
            packet_sha256=file_sha(packet / "binding.json"),
            labels=native.labels,
            native_prompts_sha256=binding["native"]["prompts_sha256"],
            answers_sha256=file_sha(store.path / "answers.npz"),
        ),
    )
    if backbone != parameter_sha(host.model):
        raise RuntimeError("Backbone changed")
    return dict(
        status="BLIND_DECISION_COMPLETE",
        logical=len(requests),
        native_inputs=len(calls),
        memory_count=len(states),
        backbone_sha256=backbone,
        compute=host.meter.snapshot(),
    )


if __name__ == "__main__":
    p = parser()
    p.add_argument("--packet", required=True, type=Path)
    args = p.parse_args()
    store, start = RunStore(args.out), time.monotonic()
    try:
        result = decide(args, store)
    except Exception as error:
        store.write("terminal.json", dict(status="TECHNICAL_STOP", error=str(error)))
        raise
    store.write(
        "terminal.json", result | dict(elapsed_seconds=time.monotonic() - start)
    )
    print(json.dumps(result))
