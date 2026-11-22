"""Run or replay the question-blind Gemma writer experiment at a structured midpoint."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
import time
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from role_memory_native import score_rows
from selffold.cli import ROOT, make_host
from selffold.records import RunStore, code_manifest, digest, file_sha
from selffold.report_replay import verify_files
from selffold.structured_report import Midpoint, controller, evaluate, panel, summarize
from selffold.structured_report_writer import (attach_writer, choice_reports, detach_writer,
                                              fit_writer, free_reports, training_record)

STUDY = ROOT / "provenance/structured_report_20261122"
DESIGN = ROOT / "data/structured_report_20261122.json"
GROUPS = ROOT / "data/within_topic_wording_20261103.json"


def read(path):
    return json.loads(path.read_text())


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def prepare():
    cfg = read(DESIGN)
    verify_files(ROOT, read(STUDY / "INPUTS.json")["files"])
    menus = [g["options"] for g in read(GROUPS)["groups"]]
    names = [name for pair in menus for name in pair]
    if len(set(names)) != 16:
        raise ValueError("Ambiguous action vocabulary")
    actor = controller(ROOT, cfg["actor_seed"])
    train, test = (panel(actor, cfg, names, split) for split in ("train", "test"))
    if (len(train), len(test)) != (256, 192):
        raise ValueError("Wrong split sizes")
    if {r["z_sha256"] for r in train} & {r["z_sha256"] for r in test}:
        raise ValueError("Physical midpoint overlap")
    if len({r["prompt"] for r in train + test}) != len(train + test):
        raise ValueError("Duplicate writer prompt")
    return cfg, actor, menus, names, train, test


def binding(train, test):
    return dict(design=file_sha(DESIGN), plan=file_sha(STUDY/"PLAN.md"),
                inputs=file_sha(STUDY/"INPUTS.json"), train=digest(train), test=digest(test),
                train_prompts=len(train), test_prompts=len(test),
                train_midpoints=len({q["z_sha256"] for q in train}),
                test_midpoints=len({q["z_sha256"] for q in test}))


def options(q, names):
    # The choice diagnostic's vocabulary is restricted to names visible at this midpoint.
    return [names[i] for i in sorted(set(q["rows"]))]


def native_receipt(folder, menus):
    scores = np.load(folder / "scores.npz", allow_pickle=False)
    native = read(folder / "native.json")
    calls = read(folder / "calls.json")
    if native["unique"] != 128 or len(calls) != 128 or file_sha(folder/"scores.npz") != native["scores_sha256"]:
        raise ValueError("Wrong native recipient score bank")
    idx = scores["dispatch"].reshape(len(menus), 16, 2)
    if sorted(idx.flatten()) != list(range(128)):
        raise ValueError("Incomplete native dispatch")
    bank = dict(top=np.full((8, 16, 2), -1, np.int32), ab=np.zeros((8, 16, 2, 2), np.float32),
                labels=np.asarray(native["labels"]))
    for i, menu in enumerate(menus):
        bank["top"][menu] = scores["top"][idx[i]]
        bank["ab"][menu] = scores["ab"][idx[i]]
    scores.close()
    return bank


def native_host(cfg, checkpoint):
    host, identity = make_host(SimpleNamespace(backend="hf", profile=str(ROOT/"configs/models/gemma2b.yaml"),
        checkpoint=str(checkpoint), device="cuda", dtype="bfloat16", cpu_threads=1, slots=8,
        cut=None, writer=None, transport=None, reviewed_local_code=None, max_context=1024,
        max_new_tokens=None, wall_seconds=cfg["wall_seconds"]))
    if identity["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Wrong Gemma snapshot")
    parity = host.native_parity()
    if not parity["exact"]:
        raise ValueError("Native parity failed")
    return host, parity


def checked_source(seal):
    source = code_manifest(ROOT)
    if len(seal) != 40 or source["git_commit"] != seal or source["git_status"] != "":
        raise ValueError("A clean, sealed scientific checkout is required")
    return source


def run(store, checkpoint, seal):
    cfg, actor, menus, names, train, test = prepare()
    source = checked_source(seal)
    store.write("source.json", source)
    store.write("binding.json", binding(train, test))
    store.write("train_states.json", train)
    store.write("test_states.json", test)
    host, parity = native_host(cfg, checkpoint)
    before = parameter_sha(host.model)
    store.write("host.json", dict(identity=host.identity, parity=parity, gpu=torch.cuda.get_device_name(),
                                  torch=torch.__version__, checkpoint=CHECKPOINT_SHA, base_sha256=before))
    training = []
    for q in train:
        z = Midpoint(tuple(q["rows"]), q["bit"])
        training.append(training_record(host, q["prompt"], [names[i] for i in z.fields()], options(q, names)))
    adapters = attach_writer(host, cfg)
    count = fit_writer(host, adapters, training, cfg, store)
    params = {f"{layer}_{projection}_{direction}": getattr(a, direction).weight.detach().cpu().numpy()
              for (layer, projection), a in adapters.items() for direction in ("down", "up")}
    np.savez_compressed(store.path/"writer.npz", **params)
    store.write("fit_closed.json", dict(steps=cfg["steps"], trainable=count, test_generations=0,
                                        writer_sha256=file_sha(store.path/"writer.npz"),
                                        train_sha256=file_sha(store.path/"training.jsonl")))
    test_order = sorted(test, key=lambda q: digest(["fixed-presentation-order", q["id"]]))
    records = []
    for arm in ("before", "after"):
        for adapter in adapters.values():
            adapter.enabled = arm == "after"
        for start in range(0, len(test_order), cfg["eval_batch"]):
            batch = test_order[start:start+cfg["eval_batch"]]
            prompts = [q["prompt"] for q in batch]
            for mode in ("free", "choice"):
                if mode == "free":
                    outputs = free_reports(host, prompts, cfg["generation_cap"])
                else:
                    outputs = choice_reports(host, prompts, [options(q, names) for q in batch])
                for q, output in zip(batch, outputs, strict=True):
                    row = dict(id=q["id"], arm=arm, mode=mode, prompt_sha256=digest(q["prompt"]), **output)
                    store.append("reports.jsonl", row)
                    records.append(row)
            if (start + len(batch)) % 64 == 0:
                print(f"{arm}: {start + len(batch)}/{len(test)} held-out presentations", flush=True)
    detach_writer(host, adapters)
    if before != parameter_sha(host.model):
        raise ValueError("Frozen Gemma changed during writer training")
    return score_saved(store, host, before, source, cfg, actor, menus, names, test, records)


def score_saved(store, host, before, source, cfg, actor, menus, names, test, records):
    store.write("scoring_source.json", source)
    expected_host = read(ROOT/"provenance/role_memory_20261104/run/produce/host.json")
    if host.identity != expected_host:
        raise ValueError("Native recipient host does not match the historical binding")
    store.write("recipient_host.json", dict(identity=host.identity, base_sha256=before))
    # Measure native choices after D has been taken off the model. Test/scoring code alone gets these.
    actions = torch.eye(16).reshape(8, 2, 16)
    native = RunStore(store.path/"native")
    score_rows(host, menus, actions.repeat(len(cfg["menus"]), 1, 1),
               [menu for menu in cfg["menus"] for _ in range(8)], native)
    if before != parameter_sha(host.model):
        raise ValueError("Frozen final recipient changed its base")
    scores = evaluate(actor, cfg, names, test, records, native_receipt(native.path, cfg["menus"]))
    summary = summarize(scores, cfg["menus"])
    store.write("outcomes.json", scores)
    store.write("summary.json", summary)
    files = {str(p.relative_to(store.path)): file_sha(p) for p in sorted(store.path.rglob("*")) if p.is_file()}
    store.write("closed.json", dict(files=files, reports=len(records), native_inputs=128,
              free_stops=dict(Counter(r["stop"] for r in records if r["mode"] == "free")),
              original_model_sha256=before, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
              replay_qualification_loaded=False))
    return summary["decisions"]


def finish_saved(store, checkpoint, seal, stopped):
    source = checked_source(seal)
    cfg, actor, menus, names, train, test = prepare()
    receipt = read(STUDY/"INITIAL.json")
    verify_files(stopped, receipt["files"])
    failure = read(stopped/"terminal.json")
    if failure != receipt["first_stop"] or failure["error"] != "Recipient artifact belongs to another host":
        raise ValueError("Unexpected first-stop receipt")
    writer_source = read(stopped/"source.json")
    if writer_source["git_commit"] != receipt["source"] or read(stopped/"binding.json") != binding(train, test):
        raise ValueError("Original writer or panel changed")
    reports = read_lines(stopped/"reports.jsonl")
    expected = {(q["id"], arm, mode) for q in test for arm in ("before","after") for mode in ("free","choice")}
    if {(r["id"], r["arm"], r["mode"]) for r in reports} != expected or len(reports) != len(expected):
        raise ValueError("The stopped run did not save all reports")
    for name in receipt["files"]:
        if name != "terminal.json":
            shutil.copyfile(stopped/name, store.path/name)
    store.write("recovery.json", dict(first_stop=failure, input_seal=file_sha(STUDY/"INITIAL.json"),
                                      refits=0, regenerated_reports=0))
    host, _ = native_host(cfg, checkpoint)
    before = parameter_sha(host.model)
    if before != read(stopped/"host.json")["base_sha256"]:
        raise ValueError("New recipient host weights differ from writer backbone")
    return score_saved(store, host, before, source, cfg, actor, menus, names, test, reports)


def replay(folder):
    cfg, actor, menus, names, train, test = prepare()
    files = read(folder/"closed.json")["files"]
    required = {"source.json","host.json","binding.json","train_states.json","test_states.json",
                "training.jsonl","writer.npz","fit_closed.json","reports.jsonl","native/scores.npz",
                "native/native.json","native/calls.json","outcomes.json","summary.json",
                "scoring_source.json","recipient_host.json"}
    if (folder/"recovery.json").exists():
        required.add("recovery.json")
    if set(files) != required:
        raise ValueError("Incomplete scientific archive")
    verify_files(folder, files)
    if read(folder/"binding.json") != binding(train, test) or train != read(folder/"train_states.json") or test != read(folder/"test_states.json"):
        raise ValueError("Experimental panel changed")
    source = read(folder/"source.json")
    scoring_source = read(folder/"scoring_source.json")
    runner_path = "scripts/fit_structured_report.py"
    if file_sha(ROOT/runner_path) != scoring_source["files"][runner_path]:
        raise ValueError("Scoring runner differs from the sealed version")
    if "recovery.json" in required:
        receipt = read(STUDY/"INITIAL.json")
        if read(folder/"recovery.json")["input_seal"] != file_sha(STUDY/"INITIAL.json") or source["git_commit"] != receipt["source"]:
            raise ValueError("Original source or partial run changed")
        verify_files(folder, {k: v for k, v in receipt["files"].items() if k != "terminal.json"})
        writer_program = subprocess.check_output(["git", "-C", str(ROOT), "show", source["git_commit"]+":"+runner_path])
        if hashlib.sha256(writer_program).hexdigest() != source["files"][runner_path]:
            raise ValueError("Original writer runner differs from the seal")
    elif source != scoring_source:
        raise ValueError("Unexpected scoring source for an uninterrupted run")
    for path in ("src/selffold/structured_report.py","src/selffold/structured_report_writer.py",
                 "src/selffold/role_memory.py"):
        if file_sha(ROOT/path) != source["files"][path] or file_sha(ROOT/path) != scoring_source["files"][path]:
            raise ValueError("Writer or scorer differs from the sealed run")
    if read(folder/"terminal.json")["status"] != "COMPLETE":
        raise ValueError("Incomplete run")
    records = read_lines(folder/"reports.jsonl")
    fit = read(folder/"fit_closed.json")
    steps = read_lines(folder/"training.jsonl")
    if fit["writer_sha256"] != file_sha(folder/"writer.npz") or fit["steps"] != len(steps) or len(steps) != cfg["steps"]:
        raise ValueError("Incomplete fit")
    for epoch in range(cfg["epochs"]):
        seen = [i for s in steps if s["epoch"] == epoch + 1 for i in s["examples"]]
        if sorted(seen) != list(range(len(train))):
            raise ValueError("Training coverage differs from its schedule")
    scores = evaluate(actor, cfg, names, test, records, native_receipt(folder/"native", cfg["menus"]))
    summary = summarize(scores, cfg["menus"])
    if summary != read(folder/"summary.json") or scores != read(folder/"outcomes.json"):
        raise ValueError("Reducer disagrees with downstream judgments")
    return dict(exact=True, source_commit=source["git_commit"], reports=len(records),
                evaluations=len(scores), refits=0, new_gemma_calls=0, decisions=summary["decisions"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--run", type=Path)
    modes.add_argument("--reduce", type=Path)
    parser.add_argument("--from-stopped", type=Path, help="Finish the preserved writer run without regenerating reports")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seal")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(binding(*prepare()[-2:]), sort_keys=True))
    elif args.reduce:
        print(json.dumps(replay(args.reduce), sort_keys=True))
    else:
        if not args.seal or not args.checkpoint:
            parser.error("--run needs --checkpoint and --seal")
        store = RunStore(args.run)
        start = time.monotonic()
        try:
            decisions = (finish_saved(store, args.checkpoint, args.seal, args.from_stopped) if args.from_stopped
                         else run(store, args.checkpoint, args.seal))
            terminal = dict(status="COMPLETE", decisions=decisions)
        except Exception as exc:
            store.write("terminal.json", dict(status="TECHNICAL_STOP", error=str(exc),
                                              elapsed_s=time.monotonic()-start))
            raise
        store.write("terminal.json", terminal | dict(elapsed_s=time.monotonic()-start))
