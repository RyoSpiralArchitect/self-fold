"""Score content-addressed soft rows with the public frozen Gemma reader."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_conditional_curriculum import parser
from selffold.cli import ROOT, make_host
from selffold.records import RunStore, digest, file_sha
from selffold.recipient_learning import load_recipient
from state_only_native import NativeChoice, fingerprint

PARENT = ROOT / "provenance/state_only_20261104/run/produce"
READER = PARENT / "packet/reader.npz"
READER_SHA = "427e138c8693e60ebf28074836a8452dc6c7da93e9bab1c876c03b33dc0b8909"


@torch.no_grad()
def score_rows(host, menus, rows, topics, out):
    if file_sha(READER) != READER_SHA or rows.shape != (len(topics), 2, 16):
        raise ValueError("Unexpected native interface")
    if (
        not torch.isfinite(rows).all()
        or torch.any(rows < 0)
        or not torch.allclose(rows.sum(-1), torch.ones(rows.shape[:2]), atol=1e-5)
    ):
        raise ValueError("Invalid soft action rows")
    native = NativeChoice(host, menus)
    keys, calls = {}, []
    dispatch = np.empty((len(rows), 2, 2), np.int32)
    for i, topic in enumerate(topics):
        for r in (0, 1):
            for order in (0, 1):
                key = (fingerprint(rows[i, r]), int(topic), order)
                if key not in keys:
                    keys[key] = len(calls)
                    calls.append((rows[i, r], int(topic), order))
                dispatch[i, r, order] = keys[key]
    reader, _, sha = load_recipient(READER, host)
    top, ab = [], []
    try:
        for i in range(0, len(calls), 16):
            values, menu, order = zip(*calls[i : i + 16])
            logits = native.logits(reader, torch.stack(values), menu, order)
            top.extend(logits.argmax(-1).cpu().tolist())
            ab.extend(logits[:, native.labels].cpu().tolist())
    finally:
        reader.close()
    top = np.asarray(top, np.int32)
    np.savez_compressed(
        out.path / "scores.npz",
        dispatch=dispatch,
        top=top,
        ab=np.asarray(ab, np.float32),
    )
    out.write("calls.json", list(keys))
    out.write(
        "native.json",
        dict(
            reader_sha256=sha,
            labels=native.labels,
            prompts=native.binding["prompts_sha256"],
            unique=len(calls),
            scores_sha256=file_sha(out.path / "scores.npz"),
        ),
    )
    return top[dispatch], np.asarray(native.labels)


def run(args, store):
    p = args.packet
    if {x.name for x in p.iterdir()} != {"binding.json", "menus.json", "rows.npz"}:
        raise ValueError("Unexpected field in blind native packet")
    binding = json.loads((p / "binding.json").read_text())
    if any(file_sha(p / k) != binding[k] for k in ("rows.npz", "menus.json")):
        raise ValueError("Packet mismatch")
    with np.load(p / "rows.npz", allow_pickle=False) as f:
        if set(f.files) != {"rows", "menu"}:
            raise ValueError("Unexpected row metadata")
        rows, topics = torch.from_numpy(f["rows"]), f["menu"]
    host, ident = make_host(args)
    if (
        ident["checkpoint_sha256"] != CHECKPOINT_SHA
        or digest(host.identity) != binding["host"]
    ):
        raise ValueError("Frozen host differed")
    before = parameter_sha(host.model)
    score_rows(host, json.loads((p / "menus.json").read_text()), rows, topics, store)
    if before != parameter_sha(host.model):
        raise ValueError("Backbone changed")
    return dict(
        status="NATIVE_PACKET_SCORED",
        backbone=before,
        packet=file_sha(p / "binding.json"),
        compute=host.meter.snapshot(),
    )


if __name__ == "__main__":
    p = parser()
    p.add_argument("--packet", required=True, type=Path)
    args = p.parse_args()
    store = RunStore(args.out)
    start = time.monotonic()
    try:
        end = run(args, store)
    except Exception as e:
        store.write("terminal.json", dict(status="TECHNICAL_STOP", error=str(e)))
        raise
    store.write("terminal.json", end | dict(elapsed_seconds=time.monotonic() - start))
