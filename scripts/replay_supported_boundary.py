"""Fresh process: check physical Gemma queries and a fixed native G slice without fitting."""

from functools import partial
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from calibrate_boundary_readout import SPEC, stems
from fit_conditional_curriculum import parser
from fit_continuity_recipient import model_scores
from fit_stage_grammar import ACTIONS, read
from raw_supported_boundary import CONFIG, READER as RAW_READER, new_host
from role_memory_native import READER
from structured_supported_boundary import LITERAL, no_injection
from selffold.carrier_probe import capture_description
from selffold.host import SlotHost
from selffold.records import RunStore, file_sha
from selffold.recipient_learning import load_recipient
from selffold.supported_boundary import Case, questions
from selffold.transport import Transport
from state_only_native import NativeChoice


@torch.no_grad()
def numeric(host, menus, folder):
    native = NativeChoice(host, menus)
    reader, _, _ = load_recipient(READER, host)
    try:
        with np.load(folder / "native.npz", allow_pickle=False) as f:
            topics = f["topics"]
            rows = torch.from_numpy(f["rows"])
            missing = f["missing"]
            expected, ab = f["top"], f["ab"]
        jobs = [(i, o) for i in range(len(rows)) if not missing[i] for o in (0, 1)]
        for j in range(0, len(jobs), 8):
            ix, order = zip(*jobs[j : j + 8], strict=True)
            ids = list(ix)
            o = list(order)
            logits = native.logits(reader, rows[ids], topics[ids], o)
            if not np.array_equal(
                logits.argmax(-1).cpu().numpy(), expected[ids, o]
            ) or not np.array_equal(
                logits.log_softmax(-1)[:, native.labels].cpu().numpy(), ab[ids, o]
            ):
                raise ValueError("Structured physical query differs")
        for i in np.flatnonzero(missing):
            for o in (0, 1):
                top, scores = no_injection(host, reader, native, topics[i], o)
                if top != expected[i, o] or not np.array_equal(scores, ab[i, o]):
                    raise ValueError("Empty query differs")
        return expected.size
    finally:
        reader.close()


@torch.no_grad()
def literal(host, menus, folder, path, build, suffix=""):
    reader, _, _ = load_recipient(path, host)
    try:
        with np.load(folder / f"literal{suffix}.npz", allow_pickle=False) as f:
            top, ab = f["top"], f["ab"]
        with gzip.open(folder / f"rendered{suffix}.jsonl.gz", "rt") as stream:
            texts = list(map(json.loads, stream))
        if len(texts) != len(top):
            raise ValueError("Literal row count differs")
        for i, x in enumerate(texts):
            jobs = build(Case(x["topic"], 0, -1, 0, 0, 0), menus, text=x["text"])
            out = model_scores(host, reader, jobs, {"current": None}, "literal_base")
            if not np.array_equal(
                [s["greedy_token"] for s in out], top[i].flat
            ) or not np.array_equal(
                [[s["A"], s["B"]] for s in out], ab[i].reshape(8, 2)
            ):
                raise ValueError("Literal native query differs")
        return top.size
    finally:
        reader.close()


@torch.no_grad()
def raw_slice(task, menus, folder):
    cfg = read(CONFIG)
    tr = Transport.load(folder / "transport.npz")
    tr.weight = tr.weight.to(task.device)
    tr.bias = tr.bias.to(task.device)
    h = SlotHost(
        task.model,
        task.tokenizer,
        slots=task.slots,
        cut=task.cut,
        writer=task.writer,
        transport=tr,
        max_context=512,
        max_new_tokens=None,
    )
    h.meter = task.meter
    selected = [
        Case(t, i % 2, v, (i + j) % 2, j, i % 2)
        for i, t in enumerate(cfg["raw_topics"])
        for j, v in enumerate((-1, 3))
    ]
    saved = {}
    keyset = {c.key for c in selected}
    with gzip.open(folder / "decisions.jsonl.gz", "rt") as stream:
        for r in map(json.loads, stream):
            if tuple(r[:6]) in keyset and r[6] in ("boundary", "opposed", "identity"):
                saved[tuple(r[:9])] = r
    reader, _, _ = load_recipient(RAW_READER, task)
    count = len(saved)
    try:
        for c in selected:
            jobs = questions(c, menus)
            z = capture_description(
                h, c.source(menus, cfg["fresh"]), "workspace", forward_length=256
            )[0]
            oz = capture_description(
                h,
                c.donor("opposed").source(menus, cfg["fresh"]),
                "workspace",
                forward_length=256,
            )[0]
            states = dict(
                identity=z,
                boundary=h.update_latent(z, "boundary"),
                opposed=h.update_latent(oz, "boundary"),
            )
            for tag, st in states.items():
                out = model_scores(task, reader, jobs, {"current": st.slots}, "matched")
                for t, ans in zip(jobs, out, strict=True):
                    row = saved.pop((*c.key, tag, t["world"], t["label_order"]))
                    if row[10:14] != [
                        ans["greedy_label"],
                        ans["greedy_token"],
                        ans["A"],
                        ans["B"],
                    ]:
                        raise ValueError("Reconstructed raw G decision differs")
    finally:
        reader.close()
    if saved or count != 192:
        raise ValueError("Incomplete raw replay")
    return count


def run(args):
    store = RunStore(args.out)
    host = new_host(args, context=1024)
    menus = [g["options"] for g in read(ACTIONS)["groups"]]
    s = args.structured
    rep = args.readout
    fresh = partial(stems, template=read(SPEC)["fresh_question"])
    result = dict(
        primary_rows=numeric(host, menus, s),
        replication_rows=numeric(host, menus, rep),
        primary_literal=literal(host, menus, s, LITERAL, questions),
        replication_literal=literal(
            host, menus, rep, rep / "literal_reader.npz", fresh
        ),
        replication_parent_literal=literal(host, menus, rep, LITERAL, fresh, "_parent"),
        raw_state_decisions=raw_slice(host, menus, args.raw),
    )
    store.write(
        "receipt.json",
        dict(
            exact=result,
            raw_transport=file_sha(args.raw / "transport.npz"),
            update=file_sha(s / "heads.npz"),
            literal=file_sha(rep / "literal_reader.npz"),
            compute=host.meter.snapshot(),
        ),
    )


if __name__ == "__main__":
    p = parser()
    for name in ("raw", "structured", "readout"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())
