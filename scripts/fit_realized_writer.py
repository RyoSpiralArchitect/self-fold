"""Realized-state writer test (registered S-WRITE conditions), on-policy.

The writer reports on the controller's OWN realized transition, generated
lazily inside the rollout (ReportBank memoizes per realized triple and
persists every record). The decoder asserts off/off -> member1 with no role
mapping handed in; member1 is the action the fixed initial table assigns to
on/on. Controllers: shared supplied-fitted pointer for every report channel
(the registered single-pointer regime), plus supplied/reward/none ceilings
with their own actors.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_conditional_curriculum import main
from fit_role_memory import bank
from fit_mixture_boundary import members_for
from fit_lm_writer import make_reward
from role_memory_native import PARENT, READER, score_rows
from selffold.cli import ROOT, make_host
from selffold.lm_writer_reports import generate_kv, source_pairs
from selffold.mixture_boundary import (BoundaryMemory, rollout_w, train_batch,
                                       twins_w)
from selffold.realized_writer import (INITIAL_PRIOR as INITIAL_CELL,
                                      ReportBank, rollout_realized)
from selffold.records import RunStore, code_manifest, file_sha
from selffold.role_trials import donor_indices
from selffold.state_only import ClauseReader

DATA = ROOT / os.environ.get("LM_WRITER_CFG", "data/lm_writer_20261106.json")
OLD = ROOT / "data/state_only_20261104.json"
ACTIONS = ROOT / "data/within_topic_wording_20261103.json"
STYLES = ("rw_directed", "rw_natural", "rw_bare")
CEILINGS = ("supplied", "reward", "none")


def bits(mask):
    return (np.asarray(mask)[..., None] >> np.arange(4)) & 1


def expected(keys, masks, labels):
    polarity = np.array([k[3] for k in keys]).reshape(
        (-1,) + (1,) * (np.ndim(masks) + 2))
    return labels[polarity ^ bits(masks)[..., None] ^ np.arange(2)]


def select_top(top, widx, case_ids):
    ids = np.asarray(case_ids).reshape(-1, *([1] * (widx.ndim - 1)))
    return top[ids, widx]



def score(rolled, twins, keys, donors, eval_masks, top, labels):
    paths = rolled["paths"]
    path_masks = np.asarray(eval_masks)[paths]
    hits = []
    for d in range(4):
        widx = rolled["depths"][:, :, d].astype(np.int64)
        hits.append(select_top(top, widx, donors[0])
                    == expected(keys, path_masks[:, d], labels))
    hits = np.stack(hits, 2)
    complete = hits.all((2, 3, 4))
    tgot = select_top(top, twins.astype(np.int64), np.arange(len(keys)))
    base = expected(keys, np.asarray(eval_masks), labels)
    truth0 = np.broadcast_to(
        base[:, None], (len(keys), twins.shape[1], len(eval_masks), 4, 2))
    twin_out = {}
    for ci, cond in enumerate(("intact", "erased")):
        hit0 = tgot[:, :, :, 0, ci] == truth0
        hit1 = tgot[:, :, :, 1, ci] == truth0[..., ::-1]
        twin_out[cond] = [int((hit0 & hit1).sum()), int(hit0.size)]
    return dict(
        native_per_depth=[[int(hits[:, :, d].sum()), int(hits[:, :, d].size)]
                          for d in range(4)],
        all_four_correct=[int(complete.sum()), int(complete.size)],
        twins_both_histories=twin_out)


def run(args, store):
    cfg, design, actions = [json.loads(p.read_text()) for p in (DATA, OLD, ACTIONS)]
    seal = code_manifest(ROOT)
    if (seal["git_status"] or not seal["git_commit"]
            or file_sha(PARENT / "compiler.npz") != cfg["source_compiler_sha256"]
            or file_sha(READER) != cfg["native_reader_sha256"]):
        raise ValueError("Clean source and bound parent required")
    store.write("source.json", seal)
    store.write("request.json", dict(
        design=file_sha(DATA), parent_design=file_sha(OLD),
        actions=file_sha(ACTIONS),
        plan=file_sha(ROOT / "provenance/realized_writer_20261115/PLAN.md"),
        registered_plan=file_sha(ROOT / "docs/STATE_WRITER_PLAN.md")))
    host, identity = make_host(args)
    if identity["checkpoint_sha256"] != CHECKPOINT_SHA or not host.native_parity()["exact"]:
        raise ValueError("Pinned native host required")
    host.max_forwards = 1_500_000
    host.max_processed_tokens = 400_000_000
    host.wall_seconds = 21_600
    host.max_new_tokens = 48
    writer_identity = dict(host.identity, max_new_tokens=host.max_new_tokens,
                           max_context=host.max_context,
                           generation_policy="kv_capped_argmax")
    store.write("host.json", writer_identity)
    before = parameter_sha(host.model)
    menus = [g["options"] for g in actions["groups"]]
    compiler = ClauseReader(host.width * 2, 16).requires_grad_(False)
    with np.load(PARENT / "compiler.npz", allow_pickle=False) as f:
        compiler.load_state_dict({k: torch.from_numpy(f[k]) for k in f.files})
    fit_keys, fit_hashes, fitrows = bank(host, compiler, design, actions["groups"], cfg, "fit")
    keys, hashes, rows = bank(host, compiler, design, actions["groups"], cfg, "primary")
    all_keys = fit_keys + keys
    n_fit, n_all = len(fit_keys), len(all_keys)
    members_all = torch.cat([members_for("hard", fitrows, 1.0),
                             members_for("hard", rows, 1.0)])
    pairs, index = source_pairs(all_keys)
    store.write("fit_keys.json", dict(keys=fit_keys, source_sha256=fit_hashes))
    store.write("test_keys.json", dict(keys=keys, source_sha256=hashes))
    # Frozen native choice scores (fit bank for reward fitting, test bank for eval).
    native = RunStore(store.path / "fit_native_hard")
    fit_top, labels = score_rows(host, menus, members_all[:n_fit],
                                 [k[0] for k in fit_keys], native)
    nv = RunStore(store.path / "test_native_hard")
    test_top, _ = score_rows(host, menus, members_all[n_fit:],
                             [k[0] for k in keys], nv)
    (store.path / "test_variants_hard_b100").mkdir(exist_ok=True)
    np.savez_compressed(store.path / "test_variants_hard_b100/variants.npz",
                        w=np.asarray([0.0, 1.0], np.float32), top=test_top)
    donors = donor_indices(keys)
    eval_masks = cfg["fit_masks"] + cfg["test_masks"]
    np.savez_compressed(store.path / "join.npz", donors=donors,
                        eval_masks=np.array(eval_masks, np.int32))
    # Fit ceiling controllers (no reports needed).
    fits, params = [], {}
    masks_t = torch.tensor(cfg["fit_masks"])
    for seed in cfg["seeds"]:
        for channel in CEILINGS:
            actor = BoundaryMemory(seed, 1)
            rng = torch.Generator().manual_seed(1000 + seed)
            start = parameter_sha(actor)
            opt = torch.optim.Adam(actor.parameters(), lr=cfg["learning_rate"])
            sched = hashlib.sha256()
            for step_i in range(cfg["updates"]):
                ix = torch.randint(n_fit, (cfg["batch"],), generator=rng)
                pattern = masks_t[torch.randint(len(masks_t), (len(ix), 2), generator=rng)]
                noise = torch.rand(len(ix), 2, 3, generator=rng)
                for v in (ix, pattern, noise):
                    sched.update(v.numpy().tobytes())
                rw = make_reward(ix.numpy(), fit_keys, labels, fit_top)
                loss, observed = train_batch(
                    actor, members_all[ix], rw, pattern, noise,
                    "hard", channel, cfg["exploration_floor"], 1.0)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                store.append("training.jsonl", dict(
                    arm=channel, seed=seed, step=step_i + 1,
                    loss=float(loss.detach()), first_reward=observed[0],
                    second_reward=observed[1]))
            name = f"hard_{channel}_w1"
            fits.append(dict(arm=name, seed=seed, start=start,
                             finish=parameter_sha(actor), schedule=sched.hexdigest()))
            for k, v in actor.state_dict().items():
                params[f"{name}_{seed}_{k}"] = v.numpy()
    np.savez_compressed(store.path / "fits.npz", **params)
    store.write("fit_closed.json", dict(fits=fits, weights=file_sha(store.path / "fits.npz")))
    # On-policy rollouts: the writer generates inside the loop, memoized.
    report_dir = Path(os.environ.get("RW_REPORTS_DIR", store.path))
    banks = {s: ReportBank(host, s, pairs, actions["groups"], report_dir,
                           generate=lambda p: generate_kv(host, p))
             for s in STYLES}
    test_pairs = index[n_fit:]
    pair_of = {i - n_fit: int(test_pairs[i - n_fit]) for i in range(n_fit, n_all)}
    # Prefill the twin-history cells twins_w will read: (prior=8, mask=m, out=m).
    for bank_ in banks.values():
        for p in set(int(x) for x in test_pairs):
            for m in eval_masks:
                bank_.bit(p, INITIAL_CELL, m, m)
    output, summary = {}, {"arms": {}}
    test_members = members_all[n_fit:]
    with np.load(store.path / "fits.npz", allow_pickle=False) as f:
        for seed in cfg["seeds"]:
            supplied = BoundaryMemory(seed, 1).requires_grad_(False)
            sp = f"hard_supplied_w1_{seed}_"
            supplied.load_state_dict(
                {k: torch.from_numpy(f[sp + k]) for k in supplied.state_dict()})
            for style, bank_ in banks.items():
                name = f"{style}_shared_{seed}"
                rolled = rollout_realized(
                    supplied, "hard", test_members, keys, eval_masks,
                    bank_.bit, pair_of)
                matrix = bank_.matrix(len(pairs))[test_pairs]
                twins = twins_w(supplied, "hard", "lm", test_members,
                                eval_masks, 1.0, rt_bits=matrix)
                output[name + "depths"] = rolled["depths"]
                output[name + "written"] = rolled["written"]
                output[name + "prior_mismatch"] = rolled["prior_mismatch"]
                summary["arms"][name] = score(rolled, twins, keys, donors,
                                              eval_masks, test_top, labels)
                store.write(f"arm_{name}.json", summary["arms"][name])
            for channel in CEILINGS:
                actor = BoundaryMemory(seed, 1).requires_grad_(False)
                prefix = f"hard_{channel}_w1_{seed}_"
                actor.load_state_dict(
                    {k: torch.from_numpy(f[prefix + k]) for k in actor.state_dict()})
                rolled = rollout_w(actor, "hard", channel, test_members,
                                   keys, eval_masks, 1.0)
                twins = twins_w(actor, "hard", channel, test_members,
                                eval_masks, 1.0)
                name = f"{channel}_{seed}"
                output[name + "depths"] = rolled["depths"]
                output[name + "written"] = rolled["written"]
                output[name + "prior_mismatch"] = rolled["prior_mismatch"]
                summary["arms"][name] = score(rolled, twins, keys, donors,
                                              eval_masks, test_top, labels)
                store.write(f"arm_{name}.json", summary["arms"][name])
    store.write("summary.json", summary)
    output["eval_paths"] = rolled["paths"]
    np.savez_compressed(store.path / "trajectories.npz", **output)
    store.write("rollout_closed.json", dict(
        schema="realized_writer_rollout_v1",
        paths=file_sha(store.path / "trajectories.npz"),
        written_semantics="rt_member1_bits_v1"))
    if parameter_sha(host.model) != before or any(
            p.grad is not None for p in host.model.parameters()):
        raise ValueError("Backbone mutated")
    gen = {s: len(b.cache) for s, b in banks.items()}
    return dict(status="REALIZED_WRITER_CLOSED", reports_generated=gen,
                backbone=before, qualification_loaded=False,
                compute=host.meter.snapshot())


if __name__ == "__main__":
    main(run)
