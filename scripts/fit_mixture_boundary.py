"""Mixture-sweep boundary experiment: genuinely mixed intermediate states.

Development follow-up to provenance/message_boundary_20261106. Predictions
preregistered in its RESULTS.md "Next measurement" section.
"""

import hashlib
import json

import numpy as np
import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_conditional_curriculum import main
from fit_role_memory import bank
from role_memory_native import PARENT, READER, score_rows
from selffold.cli import ROOT, make_host
from selffold.mixture_boundary import (
    BoundaryMemory,
    enumerate_w,
    init_weights,
    mix_members,
    rollout_w,
    train_batch,
    truth_bits,
    twins_w,
)
from selffold.records import RunStore, code_manifest, digest, file_sha
from selffold.role_trials import donor_indices
from selffold.state_only import ClauseReader

DIR = ROOT / "provenance/message_boundary_20261106"
DATA = ROOT / "data/mixture_boundary_20261106.json"
OLD = ROOT / "data/state_only_20261104.json"
ACTIONS = ROOT / "data/within_topic_wording_20261103.json"


def arm_name(arm):
    return f"{arm['form']}_{arm['channel']}_w{arm['width']}_b{int(arm['beta'] * 100)}"


def members_for(form, rows, beta):
    mixed = mix_members(rows, beta)
    if (mixed[:, 0].argmax(-1) == mixed[:, 1].argmax(-1)).any():
        raise ValueError("Member actions collide under mixing")
    if form == "hard":
        return torch.nn.functional.one_hot(mixed.argmax(-1), 16).float()
    return mixed


def score_variants(host, menus, members, w_values, topics, store):
    """Score every realizable member-mixture row exactly, content-addressed."""
    if members.shape[-1] != 16 or (members < 0).any():
        raise ValueError("Unexpected member rows")
    realized = []
    for w in w_values:
        realized.append(w * members[:, 1:2] + (1 - w) * members[:, 0:1])
    rows = torch.cat(realized, 1)  # (n, W, 16)
    dummy = torch.full((len(members), rows.shape[1], 16), 1 / 16)
    pairs = torch.stack([rows, dummy], 2).reshape(-1, 2, 16)
    tpc = [t for t in topics for _ in w_values]
    top, letters = score_rows(host, menus, pairs, tpc, store)
    return top[:, 0].reshape(len(members), len(w_values), 2), letters


def make_reward(case_ix, keys, labels, top, w_values):
    """reward(descriptor, mask) -> (batch, worlds, orders) correctness.

    descriptor: (batch, 4) realized w (soft) or member index (hard).
    mask: (batch,) instruction mask; gold letters derive from it per world.
    """
    def reward(descriptor, mask):
        mask_np = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
        pole = np.array([keys[i][3] for i in case_ix])[:, None, None]
        world_bits = (mask_np[:, None, None] >> np.arange(4)[None, :, None]) & 1
        truth = labels[pole ^ world_bits ^ np.arange(2)[None, None, :]]
        if descriptor.dtype in (torch.float16, torch.float32, torch.float64):
            d = descriptor.detach().cpu().numpy()
            idx = np.argmin(np.abs(np.asarray(w_values)[None] - d.ravel()[:, None]), -1).reshape(d.shape)
            if np.abs(np.asarray(w_values)[idx] - d).max() > 1e-5:
                raise ValueError("Realized weight outside the scored variant set")
        else:
            idx = descriptor.long().cpu().numpy()
        got = top[case_ix[:, None], idx]
        return torch.from_numpy((got == truth).astype(np.float32))

    return reward


def fit_arm(arm, seed, members, keys, labels, top, w_values, cfg, store):
    actor = BoundaryMemory(seed, arm["width"])
    rng = torch.Generator().manual_seed(1000 + seed)
    start = parameter_sha(actor)
    opt = torch.optim.Adam(actor.parameters(), lr=cfg["learning_rate"])
    masks = torch.tensor(cfg["fit_masks"])
    schedule = hashlib.sha256()
    n = len(members)
    for step_i in range(cfg["updates"]):
        ix = torch.randint(n, (cfg["batch"],), generator=rng)
        pattern = masks[torch.randint(len(masks), (len(ix), 2), generator=rng)]
        noise = torch.rand(len(ix), 2, 2 + arm["width"], generator=rng)
        for v in (ix, pattern, noise):
            schedule.update(v.numpy().tobytes())
        rw = make_reward(ix.numpy(), keys, labels, top, w_values)
        loss, observed = train_batch(
            actor, members[ix], rw, pattern, noise,
            arm["form"], arm["channel"], cfg["exploration_floor"], arm["beta"],
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        store.append(
            "training.jsonl",
            dict(arm=arm_name(arm), seed=seed, step=step_i + 1,
                 loss=float(loss.detach()), first_reward=observed[0],
                 second_reward=observed[1]),
        )
    return actor, dict(arm=arm_name(arm), seed=seed, start=start,
                       finish=parameter_sha(actor), schedule=schedule.hexdigest())


def run(args, store):
    cfg, design, actions = [json.loads(p.read_text()) for p in (DATA, OLD, ACTIONS)]
    seal = code_manifest(ROOT)
    if (seal["git_status"] or not seal["git_commit"]
            or file_sha(PARENT / "compiler.npz") != cfg["source_compiler_sha256"]
            or file_sha(READER) != cfg["native_reader_sha256"]):
        raise ValueError("Clean source and bound parent required")
    store.write("source.json", seal)
    store.write("request.json", dict(
        design=file_sha(DATA), parent_design=file_sha(OLD), actions=file_sha(ACTIONS),
        plan=file_sha(DIR / "PLAN.md"), boundary_plan=file_sha(ROOT / "docs/G_MESSAGE_BOUNDARY_PLAN.md"),
        prior_results=file_sha(DIR / "RESULTS.md")))
    host, identity = make_host(args)
    if identity["checkpoint_sha256"] != CHECKPOINT_SHA or not host.native_parity()["exact"]:
        raise ValueError("Pinned native host required")
    # Development scoring of exact realized rows needs more forwards than the
    # 20k engineering default; recorded in host.json identity separately.
    host.max_forwards = 400_000
    host.max_processed_tokens = 400_000_000
    host.wall_seconds = 21_600
    before = parameter_sha(host.model)
    store.write("host.json", host.identity)
    groups = actions["groups"]
    menus = [g["options"] for g in groups]
    compiler = ClauseReader(host.width * 2, 16).requires_grad_(False)
    with np.load(PARENT / "compiler.npz", allow_pickle=False) as f:
        compiler.load_state_dict({k: torch.from_numpy(f[k]) for k in f.files})
    keys, hashes, fitrows = bank(host, compiler, design, groups, cfg, "fit")
    store.write("fit_keys.json", dict(keys=keys, source_sha256=hashes))
    # Score the needed member rows once; everything else derives from W.
    native = RunStore(store.path / "fit_native_soft")
    topics = [k[0] for k in keys]
    base_top, labels = score_rows(host, menus, fitrows, topics, native)
    betas = sorted({a["beta"] for a in cfg["arms"]})
    variant_cache = {}
    fits, params = [], {}
    for beta in betas:
        for form in ("soft", "hard"):
            members = members_for(form, fitrows, beta)
            w_values = [0.0, 1.0] if form == "hard" else enumerate_w(beta, depth=2)
            nv = RunStore(store.path / f"fit_variants_{form}_b{int(beta * 100)}")
            vt, _ = score_variants(host, menus, members, w_values, topics, nv)
            variant_cache[(form, beta)] = (members, w_values, vt)
    for seed in cfg["seeds"]:
        for arm in cfg["arms"]:
            members, w_values, vt = variant_cache[(arm["form"], arm["beta"])]
            actor, receipt = fit_arm(
                arm, seed, members, keys, labels, vt, w_values, cfg, store)
            fits.append(receipt | dict(arm))
            for k, v in actor.state_dict().items():
                params[f"{arm_name(arm)}_{seed}_{k}"] = v.numpy()
    np.savez_compressed(store.path / "fits.npz", **params)
    store.write("fit_closed.json", dict(
        fits=fits, weights=file_sha(store.path / "fits.npz"), fresh_captures=0,
        training_episodes_per_arm_seed=cfg["updates"] * cfg["batch"]))
    keys, hashes, rows = bank(host, compiler, design, groups, cfg, "primary")
    store.write("test_keys.json", dict(keys=keys, source_sha256=hashes))
    eval_masks = cfg["fit_masks"] + cfg["test_masks"]
    donors = donor_indices(keys)
    np.savez_compressed(store.path / "join.npz", donors=donors,
                        eval_masks=np.array(eval_masks, np.int32))
    output = {}
    realized = {}
    with np.load(store.path / "fits.npz", allow_pickle=False) as f:
        for fit in fits:
            seed, name = fit["seed"], fit["arm"]
            arm = next(a for a in cfg["arms"] if arm_name(a) == name)
            actor = BoundaryMemory(seed, arm["width"]).requires_grad_(False)
            prefix = f"{name}_{seed}_"
            actor.load_state_dict(
                {k: torch.from_numpy(f[prefix + k]) for k in actor.state_dict()})
            if parameter_sha(actor) != fit["finish"]:
                raise ValueError("Fitted endpoint replay failed")
            members = members_for(arm["form"], rows, arm["beta"])
            rolled = rollout_w(actor, arm["form"], arm["channel"], members,
                               keys, eval_masks, arm["beta"])
            output.setdefault("eval_paths", rolled.pop("paths"))
            rolled["twins"] = twins_w(actor, arm["form"], arm["channel"],
                                      members, eval_masks, arm["beta"])
            tag = (arm["form"], arm["beta"])
            bucket = realized.setdefault(tag, set())
            for k in ("depths", "second", "fourth", "twins"):
                bucket.update(np.unique(rolled[k].astype(np.float32)))
            output.update({prefix + k: v for k, v in rolled.items()})
    # Score only the row vectors actually realized, exactly and content-addressed.
    test_variants = {}
    for (form, beta), used in sorted(realized.items()):
        members = members_for(form, rows, beta)
        w_values = [0.0, 1.0] if form == "hard" else sorted(float(w) for w in used)
        nv = RunStore(store.path / f"test_variants_{form}_b{int(beta * 100)}")
        vt, _ = score_variants(host, menus, members, w_values,
                               [k[0] for k in keys], nv)
        np.savez_compressed(
            nv.path / "variants.npz",
            w=np.asarray(w_values, np.float32), top=vt)
        test_variants[(form, beta)] = len(w_values)
    np.savez_compressed(store.path / "trajectories.npz", **output)
    store.write("rollout_closed.json", dict(
        paths=file_sha(store.path / "trajectories.npz"), scored_fresh_native=0))
    if parameter_sha(host.model) != before or any(
            p.grad is not None for p in host.model.parameters()):
        raise ValueError("Backbone mutated")
    return dict(status="MIXTURE_BOUNDARY_CLOSED", fit_sources=len(fitrows),
                eval_sources=len(rows),
                arms=[arm_name(a) for a in cfg["arms"]], backbone=before,
                qualification_loaded=False, compute=host.meter.snapshot())


if __name__ == "__main__":
    main(run)
