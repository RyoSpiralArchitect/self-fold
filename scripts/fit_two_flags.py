"""Train two equal-budget literal readers and test a one-bit intermediate decision."""

import json
from pathlib import Path

import torch

from conditional_curriculum_learning import gradient_norm, weighted_native_loss
from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_conditional_curriculum import main, make_schedule
from fit_continuity_recipient import batch_inputs, model_scores
from selffold.cli import ROOT, make_host
from selffold.conditional_curriculum import STAGES, make_cases, tasks_for
from selffold.recipient_learning import (
    batched_recipient_logits,
    load_recipient,
    save_recipient,
)
from selffold.records import code_manifest, digest, file_sha
from selffold.two_flags import (
    ALL_NEW,
    ARM_NAMES,
    PANELS,
    branch_probe,
    panel_fixture,
    training_pairs,
    world_probe,
)

STUDY = ROOT / "provenance/two_flags_20261104"
BASE = ROOT / "data/conditional_margin_20261103.json"
DATA = ROOT / "data/two_flags_20261104.json"
ACTIONS = ROOT / "data/within_topic_wording_20261103.json"
PARENT = ROOT / "provenance/conditional_margin_20261103/run/warm_task.npz"
PARENT_SHA = "91b7e96a22c43d6c9e9b626b8d4c3131a2d9a563bc7a5191f9a9517062727845"
CASE_FIELDS = ("key", "group", "stage", "style", "flag_pair", "pole", "source_order")


def native_inputs(host, tasks):
    return batch_inputs(
        host, tasks, dict.fromkeys(t["key"] for t in tasks), "literal_base"
    )


def equal_width_inputs(host, direct, parts, arm):
    left, right = native_inputs(host, direct), native_inputs(host, parts)
    width = max(len(left[0][0]), len(right[0][0]))
    ids, mask, _, _, at = left if arm == "direct" else right
    ids = [r + [host.tokenizer.pad_token_id] * (width - len(r)) for r in ids]
    mask = [r + [0] * (width - len(r)) for r in mask]
    return (ids, mask, None, None, at), sum(map(sum, mask))


def train_arm(
    host, arm, fit, groups, fixture, design, schedule, store, seal, label_ids
):
    lookup = {
        (
            c["group"],
            c["stage"],
            c["flag_pair"],
            c["style"],
            c["source_order"],
            c["pole"],
        ): c
        for c in fit
    }
    reader, _, sha = load_recipient(PARENT, host, trainable=True)
    parent_parameters = json.loads(
        (PARENT.parent / "warm_parameters.json").read_text()
    )["sha256"]
    try:
        if sha != PARENT_SHA or parameter_sha(reader) != parent_parameters:
            raise ValueError("Unexpected initial task adapter")
        store.append(
            "starts.jsonl",
            dict(arm=arm, artifact_sha256=sha, parameters_sha256=parent_parameters),
        )
        optimizer = torch.optim.AdamW(reader.parameters(), lr=0.0001, weight_decay=0.0)
        for step, units in enumerate(schedule):
            optimizer.zero_grad(set_to_none=True)
            before = host.meter.snapshot()
            total_loss, unmasked, counts, bindings = 0.0, 0, {}, []
            for u, unit in enumerate(units):
                for pole in (0, 1):
                    c = lookup[(*unit, pole)]
                    pairs = [
                        training_pairs(
                            c, groups[c["group"]], fixture, design, step, u, w
                        )
                        for w in range(4)
                    ]
                    for start in (0, 2):
                        chunk = pairs[start : start + 2]
                        direct = [t for _, a, _ in chunk for t in a]
                        parts = [t for _, _, b in chunk for t in b]
                        batch = direct if arm == "direct" else parts
                        inputs, useful = equal_width_inputs(host, direct, parts, arm)
                        unmasked += useful
                        for w, (kind, _, _) in enumerate(chunk, start):
                            name = "action" if arm == "direct" else kind
                            counts[name] = counts.get(name, 0) + 2
                            bindings.append(
                                (
                                    c["key"],
                                    w,
                                    name,
                                    [
                                        (t["source"], t["question"], t["gold"])
                                        for t in batch[
                                            (w - start) * 2 : (w - start) * 2 + 2
                                        ]
                                    ],
                                )
                            )
                        logits = batched_recipient_logits(host, reader, *inputs)
                        targets = torch.tensor(
                            [label_ids[t["gold"]] for t in batch], device=host.device
                        )
                        worlds = [start, start, start + 1, start + 1]
                        loss = weighted_native_loss(
                            logits,
                            targets,
                            worlds,
                            margin=2.0,
                            label_ids=list(label_ids.values()),
                        )
                        margins = logits[:, label_ids["A"]] - logits[:, label_ids["B"]]
                        bias = margins.reshape(-1, 2).mean(1)
                        symmetry = torch.nn.functional.smooth_l1_loss(
                            bias, torch.zeros_like(bias), reduction="sum"
                        )
                        loss = loss + 0.1 * symmetry / 16
                        loss.backward()
                        total_loss += loss.item()
            norm = gradient_norm(reader)
            torch.nn.utils.clip_grad_norm_(
                reader.parameters(), 1.0, error_if_nonfinite=True
            )
            if any(p.grad is not None for p in host.model.parameters()):
                raise ValueError("Backbone received gradients")
            optimizer.step()
            after = host.meter.snapshot()
            store.append(
                "training.jsonl",
                dict(
                    arm=arm,
                    step=step + 1,
                    units=units,
                    counts=counts,
                    loss=total_loss,
                    gradient_norm=norm,
                    bindings_sha256=digest(bindings),
                    unmasked_tokens=unmasked,
                    padded_tokens=after["processed_tokens"]
                    - before["processed_tokens"],
                    forwards=after["forwards"] - before["forwards"],
                ),
            )
        c = fit[0]
        check = [
            t
            for world in (0, 1)
            for t in tasks_for(c, groups[c["group"]], fixture, world)
        ]
        with torch.no_grad():
            logits_before = batched_recipient_logits(
                host, reader, *native_inputs(host, check)
            ).clone()
        path = store.path / f"{arm}.npz"
        saved = save_recipient(
            path,
            reader,
            host.identity,
            dataset_sha256=file_sha(DATA),
            source_sha256=seal["source_sha256"],
            training=dict(
                arm=arm,
                updates=len(schedule),
                parent_sha256=PARENT_SHA,
                schedule_sha256=digest(schedule),
                base_fixture_sha256=file_sha(BASE),
            ),
        )
    finally:
        reader.close()
    restored, _, _ = load_recipient(path, host)
    try:
        with torch.no_grad():
            exact = torch.equal(
                logits_before,
                batched_recipient_logits(host, restored, *native_inputs(host, check)),
            )
        if not exact:
            raise ValueError("Saved task head does not replay native fitting logits")
        store.append(
            "fits.jsonl", dict(arm=arm, artifact_sha256=saved, native_logits_exact=True)
        )
    finally:
        restored.close()


@torch.no_grad()
def score_batch(host, reader, tasks, tags, store, filename):
    scores = model_scores(
        host, reader, tasks, dict.fromkeys(t["key"] for t in tasks), "literal_base"
    )
    for task, score in zip(tasks, scores, strict=True):
        store.append(
            filename,
            tags
            | dict(
                world=task["world"],
                label_order=task["label_order"],
                gold=task["gold"],
                source_sha256=digest(task["source"]),
                question_sha256=digest(task["question"]),
            )
            | score,
        )


def evaluate(host, arm, groups, base, design, store):
    reader, _, _ = load_recipient(store.path / f"{arm}.npz", host)
    try:
        for panel in ("fit", *PANELS):
            f = base if panel == "fit" else panel_fixture(base, design, panel)
            split = "fit" if panel == "fit" else "fresh"
            cases = list(make_cases(list(groups.values()), f, split))
            for c in cases:
                tags = dict(arm=arm, panel=panel) | {key: c[key] for key in CASE_FIELDS}
                group = groups[c["group"]]
                ts = [t for w in range(4) for t in tasks_for(c, group, f, w)]
                for i in (0, 4):
                    score_batch(
                        host, reader, ts[i : i + 4], tags, store, "choices.jsonl"
                    )
                if panel == ALL_NEW:
                    for bit in (False, True):
                        pair = branch_probe(c, group, f, design["fresh_probes"], bit)
                        score_batch(
                            host,
                            reader,
                            pair,
                            tags | dict(bit=bit),
                            store,
                            "branch_choices.jsonl",
                        )
            if panel in ("fit", ALL_NEW):
                words = design["fit_probes" if panel == "fit" else "fresh_probes"]
                for pair_id, flags in enumerate(f[split]["flag_pairs"]):
                    for kind, index in (("flag", 0), ("flag", 1), ("and", 0)):
                        for w in range(4):
                            ts = world_probe(flags, w, f, words, kind, index)
                            score_batch(
                                host,
                                reader,
                                ts,
                                dict(
                                    arm=arm,
                                    panel=panel,
                                    flag_pair=pair_id,
                                    kind=kind,
                                    flag_index=index,
                                ),
                                store,
                                "world_answers.jsonl",
                            )
    finally:
        reader.close()


def run(args, store):
    seal = code_manifest(ROOT)
    base, design, actions = (json.loads(p.read_text()) for p in (BASE, DATA, ACTIONS))
    if (
        seal["git_status"] != ""
        or not seal["git_commit"]
        or file_sha(PARENT) != PARENT_SHA
    ):
        raise ValueError("Clean source and the registered parent required")
    if (
        file_sha(BASE) != design["base_fixture_sha256"]
        or file_sha(ACTIONS) != base["action_fixture_sha256"]
        or design["arm_names"] != ARM_NAMES
    ):
        raise ValueError("Input fixtures differ")
    groups = {g["id"]: g for g in actions["groups"]}
    fit = list(make_cases(list(groups.values()), base, "fit"))
    schedule = make_schedule(fit, [STAGES[i % 3] for i in range(384)], 411)
    store.write("source.json", seal)
    store.write("schedule.json", schedule)
    store.write(
        "request.json",
        dict(
            plan_sha256=file_sha(STUDY / "PLAN.md"),
            scorer_sha256=file_sha(STUDY / "summarize.py"),
            fixture_sha256=file_sha(DATA),
            base_sha256=file_sha(BASE),
            actions_sha256=file_sha(ACTIONS),
            parent_sha256=PARENT_SHA,
            schedule_sha256=digest(schedule),
            seed=411,
            steps=384,
            microbatch=4,
            effective_batch=32,
            learning_rate=0.0001,
        ),
    )
    host, checkpoint = make_host(args)
    host.max_forwards, host.max_processed_tokens = 30000, 20000000
    parity = host.native_parity()
    store.write("host.json", host.identity)
    store.write("native_parity.json", parity)
    if (
        not parity["exact"]
        or checkpoint["checkpoint_sha256"] != CHECKPOINT_SHA
        or any(p.requires_grad for p in host.model.parameters())
    ):
        raise ValueError("Pinned frozen native host required")
    native = [host.tokenizer.encode(k, add_special_tokens=False) for k in ("A", "B")]
    if any(len(ids) != 1 for ids in native) or native[0] == native[1]:
        raise ValueError("A/B must be distinct single native tokens")
    labels = dict(zip(("A", "B"), (ids[0] for ids in native)))
    store.write("labels.json", labels)
    backbone = parameter_sha(host.model)
    for arm in ARM_NAMES:
        train_arm(host, arm, fit, groups, base, design, schedule, store, seal, labels)
    store.write("fit_closed.json", dict(both_endpoints_saved=True, fresh_scores=0))
    for arm in ARM_NAMES:
        evaluate(host, arm, groups, base, design, store)
    after = parameter_sha(host.model)
    if backbone != after:
        raise ValueError("Frozen backbone changed")
    return dict(
        status="COMPLETED_TWO_FLAGS",
        backbone_before=backbone,
        backbone_after=after,
        compute=host.meter.snapshot(),
        G_trained=False,
        qualification_content_loaded=False,
    )


if __name__ == "__main__":
    main(run)
