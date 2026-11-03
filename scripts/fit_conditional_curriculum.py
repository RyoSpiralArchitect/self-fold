"""Warm a native literal reader, then compare frozen and learned source interfaces."""

import argparse
import importlib.util
import json
from pathlib import Path
import time

import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_continuity_recipient import batch_inputs, model_scores
from conditional_curriculum_learning import train_phase, native_source_slots
from selffold.cli import ROOT, make_host
from selffold.conditional_curriculum import STAGES, make_cases, tasks_for
from selffold.recipient_learning import (
    RecipientAdapter,
    batched_recipient_logits,
    save_recipient,
    load_recipient,
)
from selffold.records import RunStore, State, code_manifest, digest, file_sha

DATA = ROOT / "data/conditional_curriculum_20261103.json"
ACTIONS = ROOT / "data/within_topic_wording_20261103.json"
STUDY = ROOT / "provenance/conditional_curriculum_20261103"


def make_schedule(cases, stages, seed):
    pools = {
        s: sorted(
            {
                (c["group"], c["stage"], c["flag_pair"], c["style"], c["source_order"])
                for c in cases
                if c["stage"] == s
            }
        )
        for s in STAGES
    }
    gen = torch.Generator().manual_seed(seed)
    queues = {s: [] for s in STAGES}
    out = []
    for stage in stages:
        if len(queues[stage]) < 2:
            if queues[stage]:
                raise ValueError("Odd schedule pool")
            queues[stage] = [
                pools[stage][i]
                for i in torch.randperm(
                    len(pools[stage]), generator=gen, device="cpu"
                ).tolist()
            ]
        out.append([queues[stage].pop(), queues[stage].pop()])
    return out


def tasks(case, groups, fixture):
    return [
        t
        for world in range(4)
        for t in tasks_for(case, groups[case["group"]], fixture, world)
    ]


def encode_bank(host, cases, store, name, encoder=None):
    bank = {}
    with torch.no_grad():
        for c in cases:
            z = (
                host.encode(c["source"]).slots
                if encoder is None
                else native_source_slots(host, encoder, c["source"])[0]
            )
            bank[c["key"]] = z.detach().clone()
            store.append(
                "captures.jsonl",
                {
                    "bank": name,
                    "key": c["key"],
                    "source_sha256": digest(c["source"]),
                    "state_sha256": State(slots=z).sha(),
                },
            )
    return bank


@torch.no_grad()
def evaluate(
    host,
    adapter,
    cases,
    groups,
    fixture,
    store,
    arm,
    endpoint,
    split,
    routes,
    states=None,
    means=None,
):
    by = {
        (
            c["group"],
            c["stage"],
            c["flag_pair"],
            c["style"],
            c["source_order"],
            c["pole"],
        ): c
        for c in cases
    }
    cache = {}
    for c in cases:
        ts = tasks(c, groups, fixture)
        for route in routes:
            if route == "literal":
                z = None
            elif route == "matched":
                z = states[c["key"]]
            elif route == "opposed":
                opposite = by[
                    c["group"],
                    c["stage"],
                    c["flag_pair"],
                    c["style"],
                    c["source_order"],
                    1 - c["pole"],
                ]
                z = states[opposite["key"]]
            elif route == "zero":
                z = torch.zeros_like(states[c["key"]])
            elif route == "mean":
                z = means[c["stage"]]
            else:
                raise ValueError("Unknown route")
            sha = digest(c["source"]) if z is None else State(slots=z).sha()
            for start in (0, 4):
                batch = ts[start : start + 4]
                cache_key = (route, sha, tuple(t["question"] for t in batch))
                if route in ("zero", "mean") and cache_key in cache:
                    values = cache[cache_key]
                else:
                    values = model_scores(
                        host,
                        adapter,
                        batch,
                        {c["key"]: z},
                        "literal_base" if route == "literal" else "matched",
                    )
                    if route in ("zero", "mean"):
                        cache[cache_key] = values
                for t, s in zip(batch, values, strict=True):
                    store.append(
                        "decisions.jsonl",
                        {
                            k: c[k]
                            for k in (
                                "key",
                                "group",
                                "stage",
                                "pole",
                                "source_order",
                                "flag_pair",
                                "style",
                            )
                        }
                        | {
                            "arm": arm,
                            "endpoint": endpoint,
                            "split": split,
                            "route": route,
                            "world": t["world"],
                            "label_order": t["label_order"],
                            "gold": t["gold"],
                            "source_sha256": digest(c["source"]),
                            "question_sha256": digest(t["question"]),
                            "injected_sha256": sha,
                        }
                        | s,
                    )


def save_checked(
    host, adapter, encoder, case, groups, fixture, store, name, seal, dataset_path=DATA
):
    ts = tasks(case, groups, fixture)[:4]
    with torch.no_grad():
        z = (
            None
            if name == "warm"
            else (
                host.encode(case["source"]).slots
                if encoder is None
                else native_source_slots(host, encoder, case["source"])[0]
            )
        )
        args = batch_inputs(
            host, ts, {case["key"]: z}, "literal_base" if name == "warm" else "matched"
        )
        expected = batched_recipient_logits(host, adapter, *args).clone()
    paths = [(name + "_task.npz", adapter)]
    if encoder is not None:
        paths.append((name + "_source.npz", encoder))
    shas = {}
    for filename, module in paths:
        shas[filename] = save_recipient(
            store.path / filename,
            module,
            host.identity,
            dataset_sha256=file_sha(dataset_path),
            source_sha256=seal["source_sha256"],
            training={
                "phase": name,
                "fixed_endpoint": True,
                "schedule_file_sha256": file_sha(store.path / "schedule.json"),
                "task_parameters_sha256": parameter_sha(adapter),
                "source_parameters_sha256": (
                    parameter_sha(encoder) if encoder is not None else None
                ),
            },
        )
    adapter.close()
    if encoder is not None:
        encoder.close()
    restored, _, _ = load_recipient(store.path / (name + "_task.npz"), host)
    source = None
    if encoder is not None:
        source, _, _ = load_recipient(store.path / (name + "_source.npz"), host)
    with torch.no_grad():
        actual_z = (
            None
            if name == "warm"
            else (
                host.encode(case["source"]).slots
                if source is None
                else native_source_slots(host, source, case["source"])[0]
            )
        )
        actual = batched_recipient_logits(
            host,
            restored,
            *batch_inputs(
                host,
                ts,
                {case["key"]: actual_z},
                "literal_base" if name == "warm" else "matched",
            ),
        )
        if not torch.equal(expected, actual) or (
            z is not None and not torch.equal(z, actual_z)
        ):
            raise ValueError("Curriculum serialization replay differs")
    store.append(
        "fits.jsonl",
        {
            "phase": name,
            "artifacts": shas,
            "rows_exact": True,
            "native_logits_exact": True,
        },
    )
    return restored, source


def run(
    args,
    store,
    *,
    data=DATA,
    study=STUDY,
    warm_start=None,
    margin=0.0,
    warm_stages=None,
    completion_status="COMPLETED_CONDITIONAL_CURRICULUM",
):
    seal = code_manifest(ROOT)
    if not seal["git_commit"] or seal["git_status"] != "":
        raise ValueError("Clean sealed source required")
    fixture = json.loads(data.read_text())
    reducer_spec = importlib.util.spec_from_file_location(
        "active_curriculum_scores", study / "summarize.py"
    )
    reducer = importlib.util.module_from_spec(reducer_spec)
    reducer_spec.loader.exec_module(reducer)
    groups = {g["id"]: g for g in json.loads(ACTIONS.read_text())["groups"]}
    if file_sha(ACTIONS) != fixture["action_fixture_sha256"]:
        raise ValueError("Canonical actions differ")
    fit = list(make_cases(list(groups.values()), fixture, "fit"))
    fresh = list(make_cases(list(groups.values()), fixture, "fresh"))
    if len(fit) != 480 or len(fresh) != 160:
        raise ValueError("Fixture counts differ")
    warm_schedule = make_schedule(
        fit,
        (
            warm_stages
            if warm_stages is not None
            else ["cases"] * 64 + ["rule"] * 64 + [STAGES[i % 3] for i in range(256)]
        ),
        47,
    )
    slot_schedule = make_schedule(fit, [STAGES[i % 3] for i in range(256)], 947)
    store.write("source.json", seal)
    store.write("schedule.json", {"warm": warm_schedule, "adaptation": slot_schedule})
    store.write(
        "request.json",
        {
            "fixture_sha256": file_sha(data),
            "plan_sha256": file_sha(study / "PLAN.md"),
            "reducer_sha256": file_sha(study / "summarize.py"),
            "schedule_sha256": digest(
                {"warm": warm_schedule, "adaptation": slot_schedule}
            ),
            "warm_updates": 384,
            "adaptation_updates": 256,
            "arms": ["frozen", "joint"],
            "margin": margin,
            "warm_start_sha256": file_sha(warm_start) if warm_start else None,
            "optimizer_reset": True,
        },
    )
    host, checkpoint = make_host(args)
    host.max_forwards = 40000
    host.max_processed_tokens = 30_000_000
    store.write("host.json", host.identity)
    store.write("native_parity.json", parity := host.native_parity())
    if not parity["exact"] or checkpoint["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Model binding differs")
    backbone = parameter_sha(host.model)
    if warm_start is None:
        task = RecipientAdapter(host, rank=16, last_layers=4, seed=47)
    else:
        task, _, _ = load_recipient(warm_start, host, trainable=True)
        store.write(
            "parent_initialization.json",
            {
                "artifact_sha256": file_sha(warm_start),
                "parameter_sha256": parameter_sha(task),
                "parent_literal_gate": False,
            },
        )
    try:
        evaluate(
            host,
            task,
            fit,
            groups,
            fixture,
            store,
            "warm",
            "initial",
            "fit",
            ("literal",),
        )
        train_phase(
            host,
            task,
            None,
            fit,
            groups,
            fixture,
            warm_schedule,
            store,
            "warm",
            literal=True,
            margin=margin,
        )
        task, _ = save_checked(
            host, task, None, fit[0], groups, fixture, store, "warm", seal, data
        )
        evaluate(
            host,
            task,
            fit,
            groups,
            fixture,
            store,
            "warm",
            "final",
            "fit",
            ("literal",),
        )
        store.write(
            "warm_parameters.json",
            {
                "sha256": parameter_sha(task),
                "artifact_sha256": file_sha(store.path / "warm_task.npz"),
            },
        )
        gate = reducer.warm_gate(reducer.rows(store.path / "decisions.jsonl"))
        store.write("warm_gate.json", gate)
    finally:
        task.close()
    if gate["passed"]:
        raw = encode_bank(host, fit, store, "frozen_fit")
        for arm in ("frozen", "joint"):
            task, _, _ = load_recipient(
                store.path / "warm_task.npz", host, trainable=True
            )
            encoder = None
            if (
                parameter_sha(task)
                != json.loads((store.path / "warm_parameters.json").read_text())[
                    "sha256"
                ]
            ):
                raise ValueError("Branch did not restore exact warmed task weights")
            if arm == "joint":
                encoder = RecipientAdapter(
                    host,
                    rank=16,
                    last_layers=4,
                    seed=947,
                    scope="source_encoding",
                    layer_indices=list(range(host.cut - 4, host.cut)),
                )
                with torch.no_grad():
                    for stage in STAGES:
                        c = next(c for c in fit if c["stage"] == stage)
                        z, _ = native_source_slots(host, encoder, c["source"])
                        if not torch.equal(z, raw[c["key"]]):
                            raise ValueError(
                                "Zero source adapter differs from original E"
                            )
                store.write(
                    "source_parity.json",
                    {
                        "stages": list(STAGES),
                        "native_length_rows_exact": True,
                        "before_source_updates": 0,
                    },
                )
            try:
                store.append(
                    "branch_initializations.jsonl",
                    {
                        "arm": arm,
                        "task_parameter_sha256": parameter_sha(task),
                        "warm_artifact_sha256": file_sha(store.path / "warm_task.npz"),
                    },
                )
                train_phase(
                    host,
                    task,
                    encoder,
                    fit,
                    groups,
                    fixture,
                    slot_schedule,
                    store,
                    arm,
                    raw_cache=raw,
                    margin=margin,
                )
                task, encoder = save_checked(
                    host, task, encoder, fit[0], groups, fixture, store, arm, seal, data
                )
                bank = (
                    raw
                    if encoder is None
                    else encode_bank(host, fit, store, arm + "_fit", encoder)
                )
                evaluate(
                    host,
                    task,
                    fit,
                    groups,
                    fixture,
                    store,
                    arm,
                    "final",
                    "fit",
                    ("matched",),
                    bank,
                )
            finally:
                task.close()
                if encoder is not None:
                    encoder.close()
        store.write(
            "branches_closed.json",
            {"fresh_decisions": 0, "both_fixed_endpoints_saved": True},
        )
    task, _, _ = load_recipient(store.path / "warm_task.npz", host)
    try:
        evaluate(
            host,
            task,
            fresh,
            groups,
            fixture,
            store,
            "warm",
            "final",
            "fresh",
            ("literal",),
        )
    finally:
        task.close()
    if gate["passed"]:
        for arm in ("frozen", "joint"):
            task, _, _ = load_recipient(store.path / (arm + "_task.npz"), host)
            encoder = None
            if arm == "joint":
                encoder, _, _ = load_recipient(store.path / (arm + "_source.npz"), host)
            try:
                bank = encode_bank(host, fresh, store, arm + "_fresh", encoder)
                fitting = (
                    raw
                    if arm == "frozen"
                    else encode_bank(host, fit, store, "joint_fit_mean", encoder)
                )
                means = {
                    s: torch.stack([fitting[c["key"]] for c in fit if c["stage"] == s])
                    .float()
                    .mean(0)
                    .to(host.dtype)
                    for s in STAGES
                }
                evaluate(
                    host,
                    task,
                    fresh,
                    groups,
                    fixture,
                    store,
                    arm,
                    "final",
                    "fresh",
                    ("matched", "opposed", "zero", "mean"),
                    bank,
                    means,
                )
            finally:
                task.close()
                if encoder is not None:
                    encoder.close()
    after = parameter_sha(host.model)
    if after != backbone:
        raise ValueError("Backbone parameters changed")
    return {
        "status": completion_status,
        "literal_gate": gate["passed"],
        "backbone_before": backbone,
        "backbone_after": after,
        "compute": host.meter.snapshot(),
        "G_trained": False,
        "qualification_content_loaded": False,
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(
        backend="hf",
        cpu_threads=1,
        device="cuda:0",
        dtype="bfloat16",
        slots=8,
        cut=None,
        writer=None,
        max_context=1024,
        max_new_tokens=None,
        until_eos=True,
        transport=None,
        encoder_bridge=None,
        reviewed_local_code=None,
        wall_seconds=2400,
        profile=ROOT / "configs/models/gemma2b.yaml",
    )
    return p


def main(run_fn=run):
    a = parser().parse_args()
    store, start = RunStore(a.out), time.monotonic()
    try:
        r = run_fn(a, store)
    except Exception as e:
        store.write(
            "terminal.json",
            {
                "status": "TECHNICAL_STOP",
                "type": type(e).__name__,
                "error": str(e),
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        raise
    store.write("terminal.json", r | {"elapsed_seconds": time.monotonic() - start})
    print(json.dumps(r))


if __name__ == "__main__":
    main()
