"""Compare grammar coverage and component allocation from the same literal parts checkpoint."""

import gzip
import json
from itertools import islice

import torch
from conditional_curriculum_learning import weighted_native_loss
from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_conditional_curriculum import main, make_schedule
from fit_two_flags import BASE, DATA as PREVIOUS, ACTIONS, native_inputs
from selffold.cli import ROOT, make_host
from selffold.conditional_curriculum import make_cases
from selffold.records import code_manifest, digest, file_sha
from selffold.recipient_learning import (
    batched_recipient_logits,
    load_recipient,
    save_recipient,
)
from selffold.stage_grammar import (
    ARMS,
    PANELS,
    case_for,
    fixture,
    panel_jobs,
    training_pair,
)

STUDY = ROOT / "provenance/stage_grammar_20261104"
DESIGN = ROOT / "data/stage_grammar_20261104.json"
PARENT = ROOT / "provenance/two_flags_20261104/run/parts.npz"


def read(path):
    return json.loads(path.read_text())


def choose_logits(host, reader, inputs, width, legacy):
    ids, mask, _, _, positions = inputs
    pad = host.tokenizer.pad_token_id
    ids = [x + [pad] * (width - len(x)) for x in ids]
    mask = [x + [0] * (width - len(x)) for x in mask]
    if legacy:
        return batched_recipient_logits(host, reader, ids, mask, None, None, positions)
    ids = torch.tensor(ids, device=host.device)
    mask = torch.tensor(mask, device=host.device)
    pos, inverse = torch.unique(
        torch.tensor(positions, device=host.device), return_inverse=True
    )
    host._budget(ids.numel())
    with reader.active():
        result = host.model(
            input_ids=ids,
            attention_mask=mask,
            logits_to_keep=pos,
            use_cache=False,
            return_dict=True,
        )
    if result.past_key_values is not None:
        raise ValueError("Unexpected KV cache")
    logits = result.logits[torch.arange(len(ids), device=host.device), inverse].float()
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite native scores")
    host.meter.forwards += 1
    host.meter.processed_tokens += ids.numel()
    return logits


def batches(units, step, look, groups, expanded, base, previous, design):
    """Build all three renderings before padding; the semantic units and label order are identical."""
    tasks = {a: [] for a in ARMS}
    counts = {a: {} for a in ARMS}
    for arm in ARMS:
        for slot, unit in enumerate(units):
            for pole in (0, 1):
                c = case_for(look, unit, pole, step, slot, arm, base, design)
                for w in range(4):
                    kind, pair = training_pair(
                        c,
                        groups[c["group"]],
                        expanded,
                        previous,
                        design,
                        step,
                        slot,
                        pole,
                        w,
                        arm,
                    )
                    tasks[arm].extend(pair)
                    counts[arm][kind] = counts[arm].get(kind, 0) + len(pair)
    return tasks, counts


def train(
    host,
    arm,
    groups,
    expanded,
    base,
    previous,
    design,
    schedule,
    store,
    seal,
    labels,
    legacy,
):
    data = list(make_cases(list(groups.values()), expanded, "fit"))
    look = {
        (
            c["group"],
            c["stage"],
            c["flag_pair"],
            c["style"],
            c["source_order"],
            c["pole"],
        ): c
        for c in data
    }
    reader, _, parent = load_recipient(PARENT, host, trainable=True)
    if parent != design["parent_sha256"]:
        raise ValueError("Unexpected initial adapter")
    start_sha = parameter_sha(reader)
    opt = torch.optim.AdamW(reader.parameters(), lr=0.0001, weight_decay=0)
    try:
        for step, unit in enumerate(schedule):
            ts, counts = batches(
                unit, step, look, groups, expanded, base, previous, design
            )
            opt.zero_grad(set_to_none=True)
            total = 0.0
            used = 0
            before = host.meter.snapshot()
            for i in range(0, 32, 4):
                bs = {a: native_inputs(host, t[i : i + 4]) for a, t in ts.items()}
                width = max(len(x[0][0]) for x in bs.values())
                chosen = ts[arm][i : i + 4]
                inputs = bs[arm]
                used += sum(map(sum, inputs[1]))
                logits = choose_logits(host, reader, inputs, width, legacy)
                target = torch.tensor(
                    [labels[t["gold"]] for t in chosen], device=host.device
                )
                loss = weighted_native_loss(
                    logits,
                    target,
                    [(n // 2) % 4 for n in range(i, i + 4)],
                    margin=2.0,
                    label_ids=list(labels.values()),
                )
                marg = logits[:, labels["A"]] - logits[:, labels["B"]]
                bias = marg.reshape(-1, 2).mean(1)
                loss = (
                    loss
                    + 0.1
                    * torch.nn.functional.smooth_l1_loss(
                        bias, torch.zeros_like(bias), reduction="sum"
                    )
                    / 16
                )
                loss.backward()
                total += loss.item()
            grad = torch.nn.utils.clip_grad_norm_(
                reader.parameters(), 1.0, error_if_nonfinite=True
            )
            if any(p.grad is not None for p in host.model.parameters()):
                raise ValueError("Backbone gradient")
            opt.step()
            after = host.meter.snapshot()
            store.append(
                "training.jsonl",
                dict(
                    arm=arm,
                    step=step + 1,
                    counts=counts[arm],
                    loss=total,
                    grad=float(grad),
                    bindings=digest(
                        [(t["source"], t["question"], t["gold"]) for t in ts[arm]]
                    ),
                    padded=after["processed_tokens"] - before["processed_tokens"],
                    unmasked=used,
                    forwards=after["forwards"] - before["forwards"],
                ),
            )
        check = native_inputs(host, ts[arm][:4])
        width = len(check[0][0])
        with torch.no_grad():
            original = choose_logits(host, reader, check, width, legacy)
        sha = save_recipient(
            store.path / (arm + ".npz"),
            reader,
            host.identity,
            dataset_sha256=file_sha(DESIGN),
            source_sha256=seal["source_sha256"],
            training=dict(parent=parent, arm=arm, steps=len(schedule)),
        )
    finally:
        reader.close()
    restored, _, _ = load_recipient(store.path / (arm + ".npz"), host)
    try:
        with torch.no_grad():
            if not torch.equal(
                original, choose_logits(host, restored, check, width, legacy)
            ):
                raise ValueError("Saved adapter replay failed")
    finally:
        restored.close()
    return dict(arm=arm, initial_parameters=start_sha, weights=sha, replayed=True)


@torch.no_grad()
def evaluate(host, arm, groups, base, prior, design, labels, legacy, store):
    reader, _, _ = load_recipient(
        PARENT if arm == "parent" else store.path / (arm + ".npz"), host
    )
    try:
        with gzip.open(store.path / (arm + ".jsonl.gz"), "wt", encoding="utf8") as out:
            for panel in PANELS:
                if arm == "parent" and panel not in ("fresh", "previous"):
                    continue
                f = fixture(base, prior, design, "train" if panel == "fit" else panel)
                words = (
                    prior["fresh_probes"]
                    if panel == "previous"
                    else design["fresh_probes"]
                )
                jobs = iter(panel_jobs(groups, f, words, panel))
                while batch := list(islice(jobs, 8)):
                    ts = [r[-1] for r in batch]
                    inputs = native_inputs(host, ts)
                    logits = choose_logits(
                        host, reader, inputs, len(inputs[0][0]), legacy
                    )
                    ab = logits.log_softmax(-1)[:, list(labels.values())].cpu().tolist()
                    for record, win, prob in zip(
                        batch, logits.argmax(-1).tolist(), ab, strict=True
                    ):
                        kind, key, index, order, t = record
                        row = [
                            panel,
                            kind,
                            key,
                            index,
                            order,
                            t["gold"],
                            win,
                            *prob,
                            digest(t["source"]),
                            digest(t["question"]),
                        ]
                        out.write(json.dumps(row, separators=(",", ":")) + "\n")
    finally:
        reader.close()


def run(args, store):
    base, prior, d, actions = map(read, (BASE, PREVIOUS, DESIGN, ACTIONS))
    groups = {g["id"]: g for g in actions["groups"]}
    seal = code_manifest(ROOT)
    if (
        seal["git_status"]
        or not seal["git_commit"]
        or file_sha(PARENT) != d["parent_sha256"]
    ):
        raise ValueError("Clean source and bound parent required")
    store.write("source.json", seal)
    store.write(
        "request.json",
        dict(
            parent=file_sha(PARENT),
            data=file_sha(DESIGN),
            plan=file_sha(STUDY / "PLAN.md"),
            scorer=file_sha(STUDY / "summarize.py"),
        ),
    )
    host, ident = make_host(args)
    host.max_forwards = 26000
    host.max_processed_tokens = 24_000_000
    if (
        ident["checkpoint_sha256"] != CHECKPOINT_SHA
        or not host.native_parity()["exact"]
    ):
        raise ValueError("Pinned native host required")
    before = parameter_sha(host.model)
    store.write("host.json", host.identity)
    labels = {
        v: host.tokenizer.encode(v, add_special_tokens=False)[0] for v in ("A", "B")
    }
    store.write("labels.json", labels)
    expanded = fixture(base, prior, d, "train")
    fit = list(make_cases(list(groups.values()), base, "fit"))
    schedule = make_schedule(
        fit,
        (["source", "cases", "source", "cases", "rule"] * (d["steps"] // 5)),
        d["seed"],
    )
    store.write("schedule.json", schedule)
    task, _, _ = load_recipient(PARENT, host)
    try:
        with torch.no_grad():
            sample = list(
                islice(panel_jobs(groups, base, prior["fit_probes"], "fit"), 4)
            )
            inputs = native_inputs(host, [r[-1] for r in sample])
            normal = choose_logits(host, task, inputs, len(inputs[0][0]), True)
            fast = choose_logits(host, task, inputs, len(inputs[0][0]), False)
            legacy = not torch.equal(normal, fast)
            store.write(
                "position_parity.json",
                dict(
                    exact=not legacy,
                    full_logits_fallback=legacy,
                    max_abs=float((normal - fast).abs().max()),
                ),
            )
    finally:
        task.close()
    fits = [
        train(
            host,
            a,
            groups,
            expanded,
            base,
            prior,
            d,
            schedule,
            store,
            seal,
            labels,
            legacy,
        )
        for a in ARMS
    ]
    store.write("fit_closed.json", dict(candidates=fits, fresh_scores=0))
    for arm in ("parent", *ARMS):
        evaluate(host, arm, groups, base, prior, d, labels, legacy, store)
    if before != parameter_sha(host.model):
        raise ValueError("Backbone mutated")
    return dict(
        status="LITERAL_STAGE_RUN_COMPLETE",
        backbone=before,
        compute=host.meter.snapshot(),
    )


if __name__ == "__main__":
    main(run)
