"""Bounded same-budget fit of text readers with a carried source reference."""
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
from selffold.records import RunStore, code_manifest, file_sha
from selffold.report_replay import compact_rollout, load_archive, read, reward_function, verify_files
from selffold.role_reference import (ReferenceReader, SlotReader, reference_twins,
    rollout_reference, train_reference_batch)
from selffold.reference_evaluation import (gate_results, paired_reference, report_audit,
    restore_slots, save_compact, summarize_panels)

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "data/role_reference_20261110.json"
STATE = None


def fit_all(design, cfg, archive, reference, store):
    previous = read(ROOT / design["previous_run"] / "fit_closed.json")["fits"]
    old = {(f["arm"], f["seed"]): f for f in previous}
    nfit = len(archive["fit_keys"])
    fits, parameters = [], {}
    for seed in cfg["seeds"]:
        for style in design["styles"]:
            for condition in design["text_fits"]:
                name = f"{condition}_{style}_{seed}"
                actor = ReferenceReader(seed, 1, 32)
                start = parameter_sha(actor)
                rng = torch.Generator().manual_seed(1000 + seed)
                optimizer = torch.optim.Adam(actor.parameters(), lr=cfg["learning_rate"])
                masks_available = torch.tensor(cfg["fit_masks"])
                schedule = hashlib.sha256()
                for update in range(cfg["updates"]):
                    cases = torch.randint(nfit, (cfg["batch"],), generator=rng)
                    masks = masks_available[torch.randint(len(masks_available), (len(cases), 2), generator=rng)]
                    noise = torch.rand(len(cases), 2, 3, generator=rng)
                    for value in (cases, masks, noise):
                        schedule.update(value.numpy().tobytes())
                    q = reference[cases] if condition == "referenced" else torch.zeros_like(reference[cases])
                    reward = reward_function(cases.numpy(), archive["fit_keys"], archive["labels"], archive["fit_top"])
                    loss, observed = train_reference_batch(actor, archive["members"][cases], reward,
                        masks, noise, archive["feats"][style], cases, q, cfg["exploration_floor"])
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    store.append("training.jsonl", dict(fit=name, update=update+1,
                        loss=float(loss.detach()), first_reward=observed[0], second_reward=observed[1]))
                expected = old[f"hard_lmtext_{style}_w1", seed]
                finish = parameter_sha(actor)
                if start != expected["start"] or schedule.hexdigest() != expected["schedule"]:
                    raise ValueError("Initialization or fitting schedule differs from write8")
                if condition == "unreferenced" and finish != expected["finish"]:
                    raise ValueError("Unreferenced fit does not reproduce write8 weights")
                fits.append(dict(name=name, seed=seed, condition=condition, style=style,
                    parameters=sum(p.numel() for p in actor.parameters()), start=start, finish=finish,
                    schedule=schedule.hexdigest(), previous_finish=expected["finish"],
                    equals_previous=finish == expected["finish"]))
                parameters.update({f"{name}_{k}": v.detach().numpy() for k, v in actor.state_dict().items()})
                print(json.dumps(dict(event="fit_closed", **fits[-1])), flush=True)
    with (store.path / "fits.npz").open("xb") as stream:
        np.savez_compressed(stream, **parameters)
    store.write("fit_closed.json", dict(fits=fits, weights=file_sha(store.path / "fits.npz")))


def evaluate_job(job):
    condition, style, seed, mode = job
    design, cfg, archive, slots, reference, out = STATE
    name = f"{condition}_{style}_{seed}_{mode}"
    ix = torch.arange(len(archive["fit_keys"]), len(archive["members"]))
    members, q = archive["members"][ix], reference[ix]
    if condition == "unreferenced":
        q = torch.zeros_like(q)
    kind = "slot" if condition == "decoded" else "text"
    actor = (SlotReader(seed, 1) if kind == "slot" else ReferenceReader(seed, 1, 32)).requires_grad_(False)
    weight_file = ROOT / design["previous_run"] / "fits.npz" if kind == "slot" else out / "fits.npz"
    prefix = f"hard_lm_{style}_w1_{seed}_" if kind == "slot" else f"{condition}_{style}_{seed}_"
    with np.load(weight_file, allow_pickle=False) as saved:
        actor.load_state_dict({k: torch.from_numpy(saved[prefix+k]) for k in actor.state_dict()})
    bank = slots[style][ix] if kind == "slot" else archive["feats"][style][ix]
    masks = cfg["fit_masks"] + cfg["test_masks"]
    rolled = rollout_reference(actor, members, archive["keys"], masks, bank, q, kind, mode)
    rolled["twins"] = reference_twins(actor, members, masks, bank, q, kind, mode)
    data = compact_rollout(rolled)
    for key in ("initial_reference", "boundary_reference", "reference_carried_unchanged"):
        data[key] = rolled[key]
    del rolled
    path = out / "trajectories" / f"{name}.npz"
    arrays = save_compact(path, data)
    summary = summarize_panels(data, archive["keys"], cfg, archive["top"], archive["labels"])
    return dict(name=name, condition=condition, style=style, seed=seed, mode=mode,
        path=str(path.relative_to(out)), sha256=file_sha(path), arrays=arrays, panels=summary)


def compare_previous(item, previous):
    if item["mode"] != "intact" or item["condition"] == "referenced":
        return
    arm = ("hard_lmtext_" if item["condition"] == "unreferenced" else "hard_lm_") + item["style"] + "_w1"
    expected = previous["arms"][arm][str(item["seed"])]
    actual = item["panels"]["all_menus"]
    if item["condition"] == "unreferenced":
        if actual != expected:
            raise ValueError(f"Baseline trajectory differs: {item['name']}")
    else:
        for key in expected:
            if key != "opposed" and expected[key] != actual[key]:
                raise ValueError(f"Decoded intact trajectory differs: {item['name']}/{key}")
        for depth, metrics in expected["opposed"].items():
            for key in ("matched", "opposed_correct", "equivalent_correct"):
                if metrics[key] != actual["opposed"][depth][key]:
                    raise ValueError("Decoded intact donor behavior changed")


def reduce_pairs(out, summaries, cfg, archive):
    pairs = {}
    for name in summaries:
        if not name.endswith("_intact") or name.startswith("unreferenced_"):
            continue
        prefix = name.removesuffix("_intact")
        with np.load(out / "trajectories" / f"{name}.npz", allow_pickle=False) as original, \
             np.load(out / "trajectories" / f"{prefix}_opposed.npz", allow_pickle=False) as flipped:
            pairs[prefix] = paired_reference(original, flipped, archive["keys"], cfg,
                                               archive["top"], archive["labels"])
    return pairs


def run(out, workers):
    global STATE
    started = time.monotonic()
    store = RunStore(out)
    try:
        design = read(DESIGN)
        cfg = read(ROOT / design["base_config"])
        manifest = read(ROOT / design["inputs_manifest"])
        source = code_manifest(ROOT)
        if not source["git_commit"] or source["git_status"]:
            raise ValueError("A clean source seal is required")
        store.write("source.json", source)
        store.write("request.json", dict(design=design, design_sha256=file_sha(DESIGN),
            plan_sha256=file_sha(ROOT / design["plan"]), inputs_sha256=file_sha(ROOT / design["inputs_manifest"]),
            workers=workers, generation_calls=0, new_native_calls=0, qualification_loaded=False))
        store.write("environment.json", dict(python=platform.python_version(), torch=torch.__version__,
            numpy=np.__version__, device="cpu", torch_threads=1))
        archive = load_archive(ROOT, cfg, manifest)
        slots, records, reference = restore_slots(ROOT, archive, design["styles"])
        store.write("restored_inputs.json", dict(report_counts=archive["counts"],
            initial_reference=reference.tolist(), derived_from="compiled_member0_option_slot",
            fit_sources=len(archive["fit_keys"]), test_sources=len(archive["keys"]),
            decoder_equality_verified=True))
        store.write("report_audit.json", report_audit(archive, records, reference, design["primary_topics"]))
        fit_all(design, cfg, archive, reference, store)
        (store.path / "trajectories").mkdir()
        STATE = design, cfg, archive, slots, reference, store.path
        jobs = [(condition, style, seed, mode)
                for seed in cfg["seeds"] for style in design["styles"]
                for condition in ("unreferenced", "referenced", "decoded")
                for mode in (["intact"] if condition == "unreferenced" else design["reference_modes"])]
        summaries, files = {}, {}
        previous = read(ROOT / design["previous_run"] / "refit_summary.json")
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
            futures = [pool.submit(evaluate_job, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                compare_previous(result, previous)
                summaries[result["name"]] = result["panels"]
                files[result["path"]] = dict(sha256=result["sha256"], arrays=result["arrays"])
                print(json.dumps(dict(event="evaluation_closed", name=result["name"],
                    primary=result["panels"]["primary_fit_menus"]["all_four_correct"])), flush=True)
        store.write("summary.json", summaries)
        store.write("trajectory_files.json", files)
        paired = reduce_pairs(store.path, summaries, cfg, archive)
        store.write("paired_reference.json", paired)
        store.write("gate.json", gate_results(design, cfg, summaries, paired))
        verify_files(ROOT, manifest["files"])
        terminal = dict(status="CARRIED_REFERENCE_EXPERIMENT_CLOSED", fits=12, evaluations=42,
            baseline_weight_replays=6, baseline_trajectory_replays=6, frozen_decoded_replays=6,
            generation_calls=0, new_native_calls=0, old_inputs_unchanged=True, qualification_loaded=False)
    except Exception as error:
        store.write("terminal.json", dict(status="TECHNICAL_STOP", error=str(error),
                                          elapsed_seconds=time.monotonic()-started))
        raise
    store.write("terminal.json", terminal | dict(elapsed_seconds=time.monotonic()-started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    torch.set_default_device("cpu")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    run(args.out, args.workers)
