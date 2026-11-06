"""W-RITE experiment: does the model's own verbalization write the role bit?

See docs/SELF_REFERENCE_READING.md. A frozen Gemma writer emits self-reports
per (source, mask, style); a fixed decoder extracts the off/off bit; the bit
enters the channel as `lm`. Arms test directed (ceiling-ish) vs natural
(spontaneous self-reference) reports against supplied/none/reward controls.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_conditional_curriculum import main
from fit_role_memory import bank
from role_memory_native import PARENT, READER, score_rows
from selffold.cli import ROOT, make_host
from selffold.lm_writer import STYLES, decode_bit, mentions_offoff, prompt_for
from fit_mixture_boundary import members_for
from selffold.mixture_boundary import (
    BoundaryMemory, rollout_w, train_batch, twins_w,
)
from selffold.records import RunStore, code_manifest, digest, file_sha
from selffold.role_trials import donor_indices
from selffold.state_only import ClauseReader

DIR = ROOT / "provenance/message_boundary_20261106"
OUT_TAG = "write"
DATA = ROOT / "data/lm_writer_20261106.json"
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


CACHE = Path.home() / "work/self-fold/local/lm_reports_v2"


def build_lm_matrix(host, cfg, all_keys, all_poles, groups, store):
    """Generate and decode self-reports; returns (bits, missing, offoff).

    Raw texts are cached outside the evidence tree under local/ (ignored);
    reruns with the same writer checkpoint reuse them.
    """
    masks = cfg["fit_masks"] + cfg["test_masks"]
    n = len(all_keys)
    bits = {s: torch.zeros(n, 16, dtype=torch.long) for s in STYLES}
    miss = {s: np.zeros((n, 16)) for s in bits}
    offoff = {s: np.zeros((n, 16)) for s in bits}
    for style in bits:
        cache = CACHE / f"{style}.jsonl"
        if cache.exists():
            rows_ = [json.loads(x) for x in cache.read_text().splitlines()]
            for r in rows_:
                options = groups[all_keys[r["source"]][0]]["options"]
                b, missing = decode_bit(r["text"], options, all_poles[r["source"]])
                bits[style][r["source"], r["mask"]] = b
                miss[style][r["source"], r["mask"]] = missing
                offoff[style][r["source"], r["mask"]] = mentions_offoff(r["text"])
            (store.path / f"reports_{style}.jsonl").write_text(cache.read_text())
            continue
        cache.parent.mkdir(parents=True, exist_ok=True)
        log = store.path / f"reports_{style}.jsonl"
        with log.open("w") as fh, cache.open("w") as cf:
            for i in range(n):
                topic, pole = all_keys[i][0], all_poles[i]
                options = groups[topic]["options"]
                for m in masks:
                    out = host.generate(prompt_for(style, options, pole, m))
                    b, missing = decode_bit(out.text, options, pole)
                    bits[style][i, m] = b
                    miss[style][i, m] = missing
                    offoff[style][i, m] = mentions_offoff(out.text)
                    line = json.dumps(dict(
                        source=i, mask=m, style=style, text=out.text,
                        bit=b, missing=missing)) + "\n"
                    fh.write(line)
                    cf.write(line)
    return bits, miss, offoff


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
        prior_results=file_sha(DIR / "RESULTS.md"),
        reading=file_sha(ROOT / "docs/SELF_REFERENCE_READING.md")))
    host, identity = make_host(args)
    if identity["checkpoint_sha256"] != CHECKPOINT_SHA or not host.native_parity()["exact"]:
        raise ValueError("Pinned native host required")
    host.max_forwards = 400_000
    host.max_processed_tokens = 400_000_000
    host.wall_seconds = 21_600
    host.max_new_tokens = 48
    before = parameter_sha(host.model)
    store.write("host.json", host.identity)
    menus = [g["options"] for g in actions["groups"]]
    compiler = ClauseReader(host.width * 2, 16).requires_grad_(False)
    with np.load(PARENT / "compiler.npz", allow_pickle=False) as f:
        compiler.load_state_dict({k: torch.from_numpy(f[k]) for k in f.files})
    fit_keys, fit_hashes, fitrows = bank(host, compiler, design, actions["groups"], cfg, "fit")
    keys, hashes, rows = bank(host, compiler, design, actions["groups"], cfg, "primary")
    all_keys = fit_keys + keys
    all_poles = np.array([k[3] for k in all_keys])
    n_all = len(all_keys)
    members_all = torch.cat([members_for("hard", fitrows, 1.0),
                             members_for("hard", rows, 1.0)])
    bits, miss, offoff = build_lm_matrix(
        host, cfg, all_keys, all_poles, actions["groups"], store)
    np.savez_compressed(store.path / "lm_matrix.npz",
                        directed=bits["directed"], natural=bits["natural"])
    store.write("lm_matrix.json", dict(
        missing={s: {m: float(miss[s][:, m].mean()) for m in cfg["fit_masks"] + cfg["test_masks"]} for s in miss},
        offoff_rate={s: float(offoff[s].mean()) for s in offoff}))
    store.write("fit_keys.json", dict(keys=fit_keys, source_sha256=fit_hashes))
    store.write("test_keys.json", dict(keys=keys, source_sha256=hashes))
    # Hard members only: realized rows are the pure member actions.
    native = RunStore(store.path / "fit_native_hard")
    fit_top, labels = score_rows(host, menus, members_all[:144], [k[0] for k in fit_keys], native)
    nv = RunStore(store.path / "test_native_hard")
    test_top, _ = score_rows(host, menus, members_all[144:], [k[0] for k in keys], nv)
    w_values = [0.0, 1.0]
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
                ix = torch.randint(144, (cfg["batch"],), generator=rng)
                pattern = masks_t[torch.randint(len(masks_t), (len(ix), 2), generator=rng)]
                noise = torch.rand(len(ix), 2, 2 + arm["width"], generator=rng)
                for v in (ix, pattern, noise):
                    sched.update(v.numpy().tobytes())
                rw = make_reward(ix.numpy(), fit_keys, labels, fit_top)
                lm_bits = (bits[style][ix[:, None], pattern]
                           if channel == "lm" else None)
                loss, observed = train_batch(
                    actor, members_all[ix], rw, pattern, noise,
                    arm["form"], channel, cfg["exploration_floor"], 1.0,
                    lm_bits=lm_bits)
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
    test_idx = torch.arange(144, n_all)
    with np.load(store.path / "fits.npz", allow_pickle=False) as f:
        for fit in fits:
            seed, name = fit["seed"], fit["arm"]
            arm = next(a for a in cfg["arms"] if arm_name(a) == name)
            channel = "lm" if arm["channel"].startswith("lm_") else arm["channel"]
            style = arm["channel"][3:] if channel == "lm" else None
            actor = BoundaryMemory(seed, arm["width"]).requires_grad_(False)
            prefix = f"{name}_{seed}_"
            actor.load_state_dict({k: torch.from_numpy(f[prefix + k]) for k in actor.state_dict()})
            lm_matrix = bits[style][test_idx] if channel == "lm" else None
            rolled = rollout_w(actor, arm["form"], channel, members_all[144:],
                               keys, eval_masks, 1.0, lm_matrix=lm_matrix)
            output.setdefault("eval_paths", rolled.pop("paths"))
            rolled["twins"] = twins_w(actor, arm["form"], channel,
                                      members_all[144:], eval_masks, 1.0,
                                      lm_matrix=lm_matrix)
            output.update({prefix + k: v for k, v in rolled.items()})
    np.savez_compressed(store.path / "trajectories.npz", **output)
    # Hard-member variant table for the summarizer.
    (store.path / "test_variants_hard_b100").mkdir(exist_ok=True)
    np.savez_compressed(store.path / "test_variants_hard_b100" / "variants.npz",
                        w=np.asarray([0.0, 1.0], np.float32), top=test_top)
    store.write("rollout_closed.json", dict(paths=file_sha(store.path / "trajectories.npz")))
    if parameter_sha(host.model) != before or any(
            p.grad is not None for p in host.model.parameters()):
        raise ValueError("Backbone mutated")
    return dict(status="LM_WRITER_CLOSED", fit_sources=144, eval_sources=len(rows),
                arms=[arm_name(a) for a in cfg["arms"]], backbone=before,
                qualification_loaded=False, compute=host.meter.snapshot())


if __name__ == "__main__":
    main(run)
