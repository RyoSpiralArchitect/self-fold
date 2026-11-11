"""Refit the report-channel controllers using write7's saved evidence only.

Execution contract: this worker does not enforce the design's 3600-second
wall limit internally. On Furnace/Linux, launch it under GNU timeout so the
deadline covers this process and its evaluation workers:

  timeout --kill-after=5s 3600s env PYTHONPATH=src OPENBLAS_NUM_THREADS=1 \\
    OMP_NUM_THREADS=1 python3 scripts/replay_lm_writer.py --out NEW_DIR --workers 2

A timeout is an incomplete technical stop, never permission to resume into
the same output directory. The extra five seconds are termination grace,
not an additional scientific computation budget.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import time

import numpy as np
import torch

from fit_codec import parameter_sha
from selffold.message_boundary import BoundaryMemory
from selffold.mixture_boundary import rollout_rt, rollout_w, train_batch, twins_w
from selffold.records import RunStore, code_manifest, file_sha
from selffold.report_replay import (compact_rollout, load_archive, read,
                                  reduce_compact, reward_function, verify_files)
from selffold.role_trials import donor_indices

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "data/report_replay_20261110.json"
STATE = None


class ArchivedZeroStart(BoundaryMemory):
    """Retain the executed write7 initialization for archived-fit replay only."""
    def initial_message(self, batch, device=None):
        if self.feat_dim:
            return torch.zeros(batch, self.feat_dim, device=device)
        return super().initial_message(batch, device)


def arm_name(arm):
    return f"{arm['form']}_{arm['channel']}_w{arm['width']}"


def channel_args(arm, archive, ix=None):
    ch = arm["channel"]
    text = ch.startswith("lmtext_")
    rt = ch.startswith("lm_rt_")
    channel = "lmtext" if text else "lm" if ch.startswith("lm_") else ch
    feats = archive["feats"][ch[7:]] if text else None
    bits = archive["bits"][ch[3:]] if rt else None
    matrix = archive["bits"][ch[3:]] if channel == "lm" and not rt else None
    if ix is not None:
        feats = feats[ix] if feats is not None else None
        bits = bits[ix] if bits is not None else None
        matrix = matrix[ix] if matrix is not None else None
    return channel, feats, bits, matrix


def fit_all(cfg, archive, store):
    fits, params = [], {}
    old_fits = {(f["arm"], f["seed"]): f for f in
                read(archive["archive"] / "fit_closed.json")["fits"]}
    n_fit = len(archive["fit_keys"])
    for seed in cfg["seeds"]:
        for arm in cfg["arms"]:
            name = arm_name(arm)
            channel, features, rt, matrix = channel_args(arm, archive)
            actor = BoundaryMemory(seed, arm["width"], feat_dim=32 if features is not None else 0)
            start = parameter_sha(actor)
            rng = torch.Generator().manual_seed(1000 + seed)
            opt = torch.optim.Adam(actor.parameters(), lr=cfg["learning_rate"])
            mask_bank = torch.tensor(cfg["fit_masks"])
            schedule = hashlib.sha256()
            start_grad_seen = False
            for step in range(cfg["updates"]):
                ix = torch.randint(n_fit, (cfg["batch"],), generator=rng)
                masks = mask_bank[torch.randint(len(mask_bank), (len(ix), 2), generator=rng)]
                noise = torch.rand(len(ix), 2, 2 + arm["width"], generator=rng)
                for value in (ix, masks, noise):
                    schedule.update(value.numpy().tobytes())
                lm_bits = None
                if matrix is not None:
                    # Preserve write7's training convention exactly. Its
                    # second prior-aware training query uses requested mask 0.
                    lm_bits = torch.stack([matrix[ix, masks[:, 0].new_full((len(ix),), 8), masks[:, 0]],
                                           matrix[ix, masks[:, 0], masks[:, 1]]], 1)
                reward = reward_function(ix.numpy(), archive["fit_keys"],
                                           archive["labels"], archive["fit_top"])
                loss, observed = train_batch(
                    actor, archive["members"][ix], reward, masks, noise,
                    arm["form"], channel, cfg["exploration_floor"], 1.,
                    lm_bits=lm_bits, rt_bits=rt, rt_feats=features,
                    cases=ix if features is not None or rt is not None else None)
                opt.zero_grad(); loss.backward()
                if actor.address.grad is not None:
                    start_grad_seen |= bool(actor.address.grad[0].abs().sum() > 0)
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                store.append("training.jsonl", dict(arm=name, seed=seed, step=step+1,
                    loss=float(loss.detach()), first_reward=observed[0], second_reward=observed[1]))
            old = old_fits[name, seed]
            finish = parameter_sha(actor)
            receipt = dict(arm=name, seed=seed, start=start, finish=finish,
                           schedule=schedule.hexdigest(), start_address_gradient_seen=start_grad_seen,
                           archived_finish=old["finish"], matches_archived_finish=finish == old["finish"], **arm)
            if start != old["start"] or receipt["schedule"] != old["schedule"]:
                raise ValueError("Initialization or training schedule changed")
            if features is None and finish != old["finish"]:
                raise ValueError("Unmodified control fit differs from write7")
            if features is not None and not start_grad_seen:
                raise ValueError("Text start address received no learning signal")
            fits.append(receipt)
            params.update({f"{name}_{seed}_{k}": v.detach().numpy()
                           for k, v in actor.state_dict().items()})
            print(json.dumps(dict(event="fit_closed", **receipt)), flush=True)
    with (store.path / "fits.npz").open("xb") as fh:
        np.savez_compressed(fh, **params)
    store.write("fit_closed.json", dict(fits=fits, weights=file_sha(store.path / "fits.npz")))
    return fits


def evaluate_job(job):
    mode, arm, seed = job
    cfg, archive, output = STATE
    name = arm_name(arm)
    path = output / mode / f"{name}_{seed}.npz"
    ix = torch.arange(len(archive["fit_keys"]), len(archive["members"]))
    channel, feats, rt, matrix = channel_args(arm, archive, ix)
    cls = ArchivedZeroStart if mode == "archived" else BoundaryMemory
    actor = cls(seed, arm["width"], feat_dim=32 if feats is not None else 0).requires_grad_(False)
    source = archive["archive"] if mode == "archived" else output
    prefix = f"{name}_{seed}_"
    with np.load(source / "fits.npz", allow_pickle=False) as saved:
        actor.load_state_dict({k: torch.from_numpy(saved[prefix+k]) for k in actor.state_dict()})
    masks = cfg["fit_masks"] + cfg["test_masks"]
    members = archive["members"][ix]
    if rt is not None or feats is not None:
        rolled = rollout_rt(actor, "hard", members, archive["keys"], masks, rt, rt_feats=feats)
    else:
        rolled = rollout_w(actor, "hard", channel, members, archive["keys"], masks, lm_matrix=matrix)
    rolled["twins"] = twins_w(actor, "hard", channel, members, masks,
                               lm_matrix=matrix, rt_bits=rt, rt_feats=feats)
    compact = compact_rollout(rolled)
    del rolled
    with path.open("xb") as fh:
        np.savez_compressed(fh, **compact)
    summary = reduce_compact(compact, archive["keys"], masks, archive["labels"], archive["top"],
                             donor_indices(archive["keys"]), cfg["test_masks"], cfg["fit_topics"])
    return dict(mode=mode, arm=name, seed=seed, summary=summary,
                file=str(path.relative_to(output)), sha256=file_sha(path))


def evaluate_all(cfg, archive, store, mode, workers):
    global STATE
    STATE = cfg, archive, store.path
    (store.path / mode).mkdir()
    jobs = [(mode, arm, seed) for seed in cfg["seeds"] for arm in cfg["arms"]]
    outcomes, files = {}, {}
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
        tasks = [pool.submit(evaluate_job, job) for job in jobs]
        for task in as_completed(tasks):
            result = task.result()
            outcomes.setdefault(result["arm"], {})[str(result["seed"])] = result["summary"]
            files[result["file"]] = result["sha256"]
            print(json.dumps(dict(event="evaluation_closed", mode=mode,
                                  arm=result["arm"], seed=result["seed"],
                                  all_four=result["summary"]["all_four_correct"])), flush=True)
    store.write(f"{mode}_summary.json", dict(schema="report_replay_summary_v1", arms=outcomes))
    store.write(f"{mode}_files.json", files)
    return outcomes


def compare_archived(outcomes, expected):
    for arm, seeds in outcomes.items():
        for seed, got in seeds.items():
            old = expected["arms"][arm][seed]
            for key in ("native_per_depth", "all_four_correct", "all_four_by_role_exchanges",
                        "twins_both_histories"):
                if got[key] != old[key]:
                    raise ValueError(f"Archived numerical replay differs: {arm}/{seed}/{key}")
            for depth, metrics in old["opposed"].items():
                if any(got["opposed"][depth][k] != v for k, v in metrics.items()):
                    raise ValueError(f"Archived donor replay differs: {arm}/{seed}/{depth}")


def run(out, workers):
    start = time.monotonic()
    store = RunStore(out)
    try:
        design = read(DESIGN)
        cfg = read(ROOT / design["base_config"])
        manifest = read(ROOT / design["inputs_manifest"])
        source = code_manifest(ROOT)
        if source["git_status"] or not source["git_commit"]:
            raise ValueError("A clean source seal is required")
        store.write("source.json", source)
        store.write("request.json", dict(design=design, design_sha256=file_sha(DESIGN),
                    inputs_manifest_sha256=file_sha(ROOT / design["inputs_manifest"]),
                    plan_sha256=file_sha(ROOT / design["plan"]), workers=workers,
                    generation_calls=0, new_native_calls=0, qualification_loaded=False))
        store.write("environment.json", dict(python=platform.python_version(),
                    torch=torch.__version__, numpy=np.__version__, device="cpu", threads=1))
        archive = load_archive(ROOT, cfg, manifest)
        store.write("restored_inputs.json", dict(report_counts=archive["counts"],
                    fit_sources=len(archive["fit_keys"]), test_sources=len(archive["keys"]),
                    row_recovery="exact_saved_call_sha256_of_float32_one_hot_bytes"))
        old = evaluate_all(cfg, archive, store, "archived", workers)
        compare_archived(old, read(archive["archive"].parent / "summary.json"))
        store.write("archive_replay.json", dict(exact=True, controllers=24,
                    comparisons=["native_per_depth", "all_four_correct", "all_four_by_role_exchanges",
                                 "twins_both_histories", "opposed"]))
        fit_all(cfg, archive, store)
        evaluate_all(cfg, archive, store, "refit", workers)
        verify_files(ROOT, manifest["files"])
        terminal = dict(status="SAVED_REPORT_REFIT_CLOSED", archived_controllers_replayed=24,
                        controllers_refitted=24, generation_calls=0, new_native_calls=0,
                        qualification_loaded=False, old_inputs_unchanged=True)
    except Exception as exc:
        store.write("terminal.json", dict(status="TECHNICAL_STOP", error=str(exc),
                                         elapsed_seconds=time.monotonic()-start))
        raise
    store.write("terminal.json", terminal | dict(elapsed_seconds=time.monotonic()-start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    run(args.out, args.workers)
