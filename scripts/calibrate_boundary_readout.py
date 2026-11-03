"""Adapt only the literal table readout, retaining the frozen one-step update."""

from functools import partial
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from fit_codec import parameter_sha
from fit_conditional_curriculum import parser
from fit_continuity_recipient import batch_inputs
from fit_stage_grammar import ACTIONS, read
from raw_supported_boundary import CONFIG, new_host
from structured_supported_boundary import OLD, LITERAL, evaluate_panel, load_heads
from selffold.cli import ROOT
from selffold.records import RunStore, code_manifest, file_sha, digest
from selffold.recipient_learning import (
    batched_recipient_logits,
    load_recipient,
    save_recipient,
)
from selffold.role_memory import RoleMemory
from selffold.supported_boundary import Case, VALUES, questions, render_cases

SPEC = ROOT / "data/supported_g_readout_20261104.json"


def stems(case, menus, text, template):
    jobs = questions(case, menus, text=text)
    for t in jobs:
        w = t["world"]
        o = t["label_order"]
        a, b = menus[case.topic][:: 1 if o == 0 else -1]
        t["question"] = (
            template.format(left=VALUES[w // 2], right=VALUES[w % 2])
            + f"\nA = {a}. B = {b}. Respond with only A or B."
        )
    return jobs


def fit(host, spec, menus, store, seal):
    units = list(product(spec["fit_topics"], range(16)))
    rng = torch.Generator().manual_seed(spec["seed"])
    schedule = [
        units[i]
        for _ in range(3)
        for i in torch.randperm(len(units), generator=rng).tolist()
    ]
    store.write("schedule.json", schedule)
    reader, _, parent = load_recipient(LITERAL, host, trainable=True)
    if parent != "fa1258b01b165959c346e9f409d0cad43543e1d48bd0c48a9b8917cb08c0ec59":
        raise ValueError("Wrong literal starting point")
    labels = [host.tokenizer.encode(c, add_special_tokens=False)[0] for c in ("A", "B")]
    opt = torch.optim.AdamW(reader.parameters(), lr=0.0002, weight_decay=0)
    path = store.path / "literal_reader.npz"
    prompt_hashes = set()
    try:
        for step, (topic, mask) in enumerate(schedule):
            gold = [(mask >> w) & 1 for w in range(4)]
            lines = render_cases([menus[topic][a] for a in gold]).splitlines()
            opt.zero_grad(set_to_none=True)
            total = 0
            for order in (0, 1):
                text = "\n".join(lines[:: 1 if order == 0 else -1])
                jobs = stems(
                    Case(topic, 0, -1, 0, 0, 0),
                    menus,
                    text,
                    spec["training_questions"][(step + order) % 2],
                )
                for task in jobs:
                    prompt_hashes.add(digest([text, task["question"]]))
                for start in (0, 4):
                    part = jobs[start : start + 4]
                    inputs = batch_inputs(host, part, {"current": None}, "literal_base")
                    logits = batched_recipient_logits(host, reader, *inputs)
                    target = torch.tensor(
                        [gold[j["world"]] ^ j["label_order"] for j in part],
                        device=host.device,
                    )
                    pair = logits[:, labels]
                    inds = torch.arange(len(part), device=host.device)
                    loss = (
                        F.cross_entropy(
                            logits, torch.tensor(labels, device=host.device)[target]
                        )
                        + (1 - pair[inds, target] + pair[inds, 1 - target])
                        .clamp_min(0)
                        .mean()
                    ) / 4
                    loss.backward()
                    total += float(loss.detach())
            grad = torch.nn.utils.clip_grad_norm_(
                reader.parameters(), 1, error_if_nonfinite=True
            )
            opt.step()
            if step % 32 == 31:
                store.append(
                    "training.jsonl",
                    dict(step=step + 1, loss=total, gradient=float(grad)),
                )
        sha = save_recipient(
            path,
            reader,
            host.identity,
            dataset_sha256=file_sha(SPEC),
            source_sha256=seal["source_sha256"],
            training=dict(parent=parent, steps=192),
        )
    finally:
        reader.close()
    store.write("fit_prompts.json", sorted(prompt_hashes))
    store.write(
        "fit_closed.json",
        dict(
            weights=sha,
            parent=parent,
            source_and_update=spec["source_and_update_sha256"],
            updates=len(schedule),
            target_labels=16 * len(schedule),
            fresh_captures=0,
            fit_topics=spec["fit_topics"],
        ),
    )
    return path, prompt_hashes


def run(args, store):
    spec = read(SPEC)
    cfg = read(CONFIG)
    cfg["fresh"] = spec["fresh"]
    menus = [g["options"] for g in read(ACTIONS)["groups"]]
    primary = args.primary
    saved = primary / "heads.npz"
    if file_sha(saved) != spec["source_and_update_sha256"] or read(
        primary / "fit_closed.json"
    )["weights"] != file_sha(saved):
        raise ValueError("Wrong frozen structured candidate")
    seal = code_manifest(ROOT)
    if seal["git_status"]:
        raise ValueError("Dirty execution tree")
    store.write("source.json", seal)
    host = new_host(args, context=1024)
    before = parameter_sha(host.model)
    parity = host.native_parity()
    if not parity["exact"]:
        raise ValueError("Native parity failure")
    actor = RoleMemory(17).requires_grad_(False)
    if (
        file_sha(OLD / "fits.npz")
        != read(primary / "fit_closed.json")["one_bit_source"]
    ):
        raise ValueError("Wrong role bit")
    with np.load(OLD / "fits.npz", allow_pickle=False) as z:
        actor.load_state_dict(
            {k: torch.from_numpy(z["reward_17_" + k]) for k in actor.state_dict()}
        )
    heads, gates = load_heads(saved)
    with (
        np.load(primary / "dispatch.npz", allow_pickle=False) as ix,
        np.load(primary / "native.npz", allow_pickle=False) as outputs,
    ):
        unrelated = torch.from_numpy(outputs["rows"][ix["unrelated"][0]]).unsqueeze(0)
    literal_path, fitted_prompts = fit(host, spec, menus, store, seal)
    fresh_stem = partial(stems, template=spec["fresh_question"])
    result = evaluate_panel(
        host,
        cfg,
        menus,
        actor,
        heads,
        gates,
        unrelated,
        store,
        saved,
        seal,
        before,
        parity,
        literal_path=literal_path,
        build_questions=fresh_stem,
        data_path=SPEC,
    )
    import gzip, json

    fresh_prompts = set()
    with gzip.open(store.path / "rendered.jsonl.gz", "rt") as stream:
        for row in map(json.loads, stream):
            c = Case(row["topic"], 0, -1, 0, 0, 0)
            fresh_prompts.update(
                digest([row["text"], t["question"]])
                for t in fresh_stem(c, menus, row["text"])
            )
    overlap = len(fitted_prompts & fresh_prompts)
    if overlap:
        raise ValueError("Fresh literal test prompt occurs in training")
    return result | dict(
        status="FROZEN_G_READOUT_REPLICATION_COMPLETE",
        fit_test_literal_overlap=overlap,
        distinct_fresh_literal_inputs=len(fresh_prompts),
        G_updates=0,
    )


if __name__ == "__main__":
    p = parser()
    p.add_argument("--primary", type=Path, required=True)
    args = p.parse_args()
    store = RunStore(args.out)
    start = time.monotonic()
    try:
        result = run(args, store)
    except Exception as e:
        store.write("terminal.json", dict(status="TECHNICAL_STOP", error=str(e)))
        raise
    store.write(
        "terminal.json", result | dict(elapsed_seconds=time.monotonic() - start)
    )
