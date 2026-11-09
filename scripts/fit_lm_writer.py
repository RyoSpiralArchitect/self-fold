"""W-RITE supplied-assignment report channel, not autonomous state verbalization.

The LM sees ideal assignments rendered from source options/polarity and a mask.
A source-aware lexical decoder maps its report to a bit. Each channel trains its
own controller; differences are not a fixed-G L/B causal comparison. New runs
use bound complete caches and decoder v3; archived write/write2 remain unchanged.
"""

import hashlib
import os
import json
from pathlib import Path

import numpy as np
import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_conditional_curriculum import main
from fit_role_memory import bank
from role_memory_native import PARENT, READER, score_rows
from selffold.cli import ROOT, make_host
from selffold.lm_writer_reports import (build_report_matrix, build_rt_matrix,
                                        save_report_matrices)
from fit_mixture_boundary import members_for
from selffold.mixture_boundary import (
    BoundaryMemory, rollout_rt, rollout_w, train_batch, twins_w,
)
from selffold.records import RunStore, code_manifest, file_sha
from selffold.role_trials import donor_indices
from selffold.state_only import ClauseReader

DIR = ROOT / "provenance/message_boundary_20261106"
DATA = ROOT / os.environ.get("LM_WRITER_CFG", "data/lm_writer_20261106.json")
OLD = ROOT / "data/state_only_20261104.json"
ACTIONS = ROOT / "data/within_topic_wording_20261103.json"


def arm_name(arm):
    return f"{arm['form']}_{arm['channel']}_w{arm['width']}"


def make_reward(case_ix, keys, labels, top):
    def reward(descriptor, mask):
        mask_np = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
        pole = np.array([keys[i][3] for i in case_ix])[:, None, None]
        world_bits = (mask_np[:, None, None] >> np.arange(4)[None, :, None]) & 1
        truth = labels[pole ^ world_bits ^ np.arange(2)[None, None, :]]
        got = top[case_ix[:, None], descriptor.long().cpu().numpy()]
        return torch.from_numpy((got == truth).astype(np.float32))
    return reward


CACHE = ROOT / "local/lm_reports_v3"


def build_lm_matrix(host, cfg, all_keys, all_poles, groups, store, binding, n_fit):
    return build_report_matrix(host, cfg, all_keys, all_poles, groups,
                               store.path, CACHE, binding, n_fit=n_fit)


def run(args, store):
    cfg, design, actions = [json.loads(p.read_text()) for p in (DATA, OLD, ACTIONS)]
    for arm in cfg["arms"]:
        if arm["form"] != "hard" or arm["width"] != 1 or arm["beta"] != 1.0:
            raise ValueError("W-RITE currently requires hard, width=1, beta=1 arms")
    seal = code_manifest(ROOT)
    if (seal["git_status"] or not seal["git_commit"]
            or file_sha(PARENT / "compiler.npz") != cfg["source_compiler_sha256"]
            or file_sha(READER) != cfg["native_reader_sha256"]):
        raise ValueError("Clean source and bound parent required")
    store.write("source.json", seal)
    store.write("request.json", dict(
        design=file_sha(DATA), parent_design=file_sha(OLD), actions=file_sha(ACTIONS),
        prior_results=file_sha(DIR / "RESULTS.md"),
        reading=file_sha(ROOT / "docs/SELF_REFERENCE_READING.md")))
    host, identity = make_host(args)
    if identity["checkpoint_sha256"] != CHECKPOINT_SHA or not host.native_parity()["exact"]:
        raise ValueError("Pinned native host required")
    host.max_forwards = 1_500_000
    host.max_processed_tokens = 400_000_000
    host.wall_seconds = 21_600
    host.max_new_tokens = 48
    before = parameter_sha(host.model)
    # make_host captured its identity before the local 48-token cap was set.
    writer_identity = dict(host.identity, max_new_tokens=host.max_new_tokens,
                           max_context=host.max_context, generation_policy="capped")
    store.write("host.json", writer_identity)
    writer_binding = dict(host=writer_identity, backbone_sha256=before,
                          source_sha256=seal["source_sha256"],
                          source_commit=seal["git_commit"])
    menus = [g["options"] for g in actions["groups"]]
    compiler = ClauseReader(host.width * 2, 16).requires_grad_(False)
    with np.load(PARENT / "compiler.npz", allow_pickle=False) as f:
        compiler.load_state_dict({k: torch.from_numpy(f[k]) for k in f.files})
    fit_keys, fit_hashes, fitrows = bank(host, compiler, design, actions["groups"], cfg, "fit")
    keys, hashes, rows = bank(host, compiler, design, actions["groups"], cfg, "primary")
    all_keys = fit_keys + keys
    all_poles = np.array([k[3] for k in all_keys])
    n_all, n_fit = len(all_keys), len(fit_keys)
    if n_fit != len(fitrows) or not n_fit or not keys:
        raise ValueError("Source-bank keys and rows disagree")
    members_all = torch.cat([members_for("hard", fitrows, 1.0),
                             members_for("hard", rows, 1.0)])
    bits, miss, offoff, generated = build_lm_matrix(
        host, cfg, all_keys, all_poles, actions["groups"], store, writer_binding,
        n_fit=n_fit)
    # RoundTrip reports are keyed by the realized (prior, mask, out) and
    # generated over the full feasible bound, for fit and test sources alike.
    rt, rt_miss, rt_offoff, rt_gen, rt_counts = build_rt_matrix(
        host, cfg, all_keys, all_poles, actions["groups"], store.path, CACHE,
        writer_binding)
    for dst, src in ((bits, rt), (miss, rt_miss),
                     (offoff, rt_offoff), (generated, rt_gen)):
        dst.update(src)
    save_report_matrices(store.path, cfg["fit_masks"] + cfg["test_masks"],
                         bits, miss, offoff, generated,
                         record_counts=rt_counts)
    store.write("fit_keys.json", dict(keys=fit_keys, source_sha256=fit_hashes))
    store.write("test_keys.json", dict(keys=keys, source_sha256=hashes))
    # Hard members only: realized rows are the pure member actions.
    native = RunStore(store.path / "fit_native_hard")
    fit_top, labels = score_rows(host, menus, members_all[:n_fit], [k[0] for k in fit_keys], native)
    nv = RunStore(store.path / "test_native_hard")
    test_top, _ = score_rows(host, menus, members_all[n_fit:], [k[0] for k in keys], nv)
    eval_masks = cfg["fit_masks"] + cfg["test_masks"]
    donors = donor_indices(keys)
    np.savez_compressed(store.path / "join.npz", donors=donors,
                        eval_masks=np.array(eval_masks, np.int32))
    fits, params = [], {}
    for seed in cfg["seeds"]:
        for arm in cfg["arms"]:
            actor = BoundaryMemory(seed, arm["width"])
            rng = torch.Generator().manual_seed(1000 + seed)
            start = parameter_sha(actor)
            opt = torch.optim.Adam(actor.parameters(), lr=cfg["learning_rate"])
            masks_t = torch.tensor(cfg["fit_masks"])
            channel = "lm" if arm["channel"].startswith("lm_") else arm["channel"]
            style = arm["channel"][3:] if channel == "lm" else None
            sched = hashlib.sha256()
            for step_i in range(cfg["updates"]):
                ix = torch.randint(n_fit, (cfg["batch"],), generator=rng)
                pattern = masks_t[torch.randint(len(masks_t), (len(ix), 2), generator=rng)]
                noise = torch.rand(len(ix), 2, 2 + arm["width"], generator=rng)
                for v in (ix, pattern, noise):
                    sched.update(v.numpy().tobytes())
                rw = make_reward(ix.numpy(), fit_keys, labels, fit_top)
                rtb = rt[style] if style in rt else None
                if channel == "lm" and rtb is None:
                    m = bits[style]
                    if m.dim() == 3:
                        # Prior-aware reports: step0 prior = initial layout (8),
                        # step1 prior = the mask applied at step0.
                        lm_bits = torch.stack([
                            m[ix, pattern[:, 0].new_full((len(ix),), 8), pattern[:, 0]],
                            m[ix, pattern[:, 0], pattern[:, 1]],
                        ], 1)
                    else:
                        lm_bits = m[ix[:, None], pattern]
                else:
                    lm_bits = None
                loss, observed = train_batch(
                    actor, members_all[ix], rw, pattern, noise,
                    arm["form"], channel, cfg["exploration_floor"], 1.0,
                    lm_bits=lm_bits, rt_bits=rtb,
                    cases=ix if rtb is not None else None)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                store.append("training.jsonl", dict(
                    arm=arm_name(arm), seed=seed, step=step_i + 1,
                    loss=float(loss.detach()), first_reward=observed[0],
                    second_reward=observed[1]))
            fits.append(dict(arm=arm_name(arm), seed=seed, start=start,
                             finish=parameter_sha(actor), schedule=sched.hexdigest()) | dict(arm))
            for k, v in actor.state_dict().items():
                params[f"{arm_name(arm)}_{seed}_{k}"] = v.numpy()
    np.savez_compressed(store.path / "fits.npz", **params)
    store.write("fit_closed.json", dict(fits=fits, weights=file_sha(store.path / "fits.npz")))
    output = {}
    test_idx = torch.arange(n_fit, n_all)
    with np.load(store.path / "fits.npz", allow_pickle=False) as f:
        for fit in fits:
            seed, name = fit["seed"], fit["arm"]
            arm = next(a for a in cfg["arms"] if arm_name(a) == name)
            channel = "lm" if arm["channel"].startswith("lm_") else arm["channel"]
            style = arm["channel"][3:] if channel == "lm" else None
            actor = BoundaryMemory(seed, arm["width"]).requires_grad_(False)
            prefix = f"{name}_{seed}_"
            actor.load_state_dict({k: torch.from_numpy(f[prefix + k]) for k in actor.state_dict()})
            rtb = rt[style][test_idx] if style in rt else None
            lm_matrix = bits[style][test_idx] if channel == "lm" and rtb is None else None
            if rtb is not None:
                rolled = rollout_rt(actor, arm["form"], members_all[n_fit:],
                                    keys, eval_masks, rt_bits=rtb)
            else:
                rolled = rollout_w(actor, arm["form"], channel, members_all[n_fit:],
                                   keys, eval_masks, 1.0, lm_matrix=lm_matrix)
            output.setdefault("eval_paths", rolled.pop("paths"))
            rolled["twins"] = twins_w(actor, arm["form"], channel,
                                      members_all[n_fit:], eval_masks, 1.0,
                                      lm_matrix=lm_matrix, rt_bits=rtb)
            output.update({prefix + k: v for k, v in rolled.items()})
        # Shared-pointer block: every LM channel rolled through ONE
        # supplied-fitted controller per seed — isolates channel quality from
        # per-channel refits.
        lm_styles = [a["channel"][3:] for a in cfg["arms"] if a["channel"].startswith("lm_")]
        for seed in cfg["seeds"]:
            actor = BoundaryMemory(seed, 1).requires_grad_(False)
            sp = f"hard_supplied_w1_{seed}_"
            actor.load_state_dict({k: torch.from_numpy(f[sp + k]) for k in actor.state_dict()})
            for style in lm_styles:
                prefix = f"shared_supplied_lm_{style}_{seed}_"
                if style in rt:
                    rolled = rollout_rt(actor, "hard", members_all[n_fit:],
                                        keys, eval_masks,
                                        rt_bits=rt[style][test_idx])
                    rolled.pop("paths", None)
                    rolled["twins"] = twins_w(actor, "hard", "lm", members_all[n_fit:],
                                              eval_masks, 1.0,
                                              rt_bits=rt[style][test_idx])
                else:
                    rolled = rollout_w(actor, "hard", "lm", members_all[n_fit:],
                                       keys, eval_masks, 1.0,
                                       lm_matrix=bits[style][test_idx])
                    rolled.pop("paths", None)
                    rolled["twins"] = twins_w(actor, "hard", "lm", members_all[n_fit:],
                                              eval_masks, 1.0,
                                              lm_matrix=bits[style][test_idx])
                output.update({prefix + k: v for k, v in rolled.items()})
    np.savez_compressed(store.path / "trajectories.npz", **output)
    # Hard-member variant table for the summarizer.
    (store.path / "test_variants_hard_b100").mkdir(exist_ok=True)
    np.savez_compressed(store.path / "test_variants_hard_b100" / "variants.npz",
                        w=np.asarray([0.0, 1.0], np.float32), top=test_top)
    store.write("rollout_closed.json", dict(
        schema="lm_writer_rollout_v2", paths=file_sha(store.path / "trajectories.npz"),
        written_semantics="carried_message_bits_v2",
        twins_axes=["source", "prior", "current", "history", "erasure", "world"]))
    if parameter_sha(host.model) != before or any(
            p.grad is not None for p in host.model.parameters()):
        raise ValueError("Backbone mutated")
    return dict(status="LM_WRITER_CLOSED", fit_sources=n_fit, eval_sources=len(rows),
                arms=[arm_name(a) for a in cfg["arms"]], backbone=before,
                qualification_loaded=False, compute=host.meter.snapshot())


if __name__ == "__main__":
    main(run)
