"""Generate the sealed single/full-record comparison or replay saved answers on CPU."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import time

import torch

from selffold.cli import make_host
from selffold.lm_writer_reports import fingerprint, generate_kv
from selffold.oracle_row_readout import analyze, requests
from selffold.records import RunStore, canonical, code_manifest, file_sha
from selffold.report_replay import read, verify_files


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "data/oracle_row_readout_20261122.json"
GROUPS = ROOT / "data/within_topic_wording_20261103.json"
STUDIES = ROOT / "provenance/message_boundary_20261106"
STUDY = STUDIES / "write12"
OLD_REQUESTS = (STUDIES / "write10/reports/requests.json", STUDIES / "write11/produce/requests.json")


def prepare():
    design = read(DESIGN)
    inputs = read(STUDY / "INPUTS.json")
    verify_files(ROOT, inputs["files"])
    reqs = requests(design, read(GROUPS)["groups"])
    prompts = {q["prompt"] for q in reqs}
    keys = {(q["topic"], q["prior"], q["instruction"], q["observation"])
            for q in reqs if q["context"] != "single"}
    for path in OLD_REQUESTS:
        prior = read(path)
        prior_keys = {(r["topic"], r["prior"], r["mask"], r["out"]) for r in prior}
        if keys & prior_keys or prompts & {r["prompt"] for r in prior}:
            raise ValueError(f"Panel overlaps {path.relative_to(ROOT)}")
    binding = dict(design_sha256=file_sha(DESIGN), plan_sha256=file_sha(STUDY / "PLAN.md"),
                   input_manifest_sha256=file_sha(STUDY / "INPUTS.json"), requests_sha256=fingerprint(reqs),
                   unique_transitions=len(keys), unique_prompts=len(prompts),
                   previous_prompt_overlap=0, previous_transition_overlap=0)
    return design, reqs, binding


def run(out, checkpoint, seal):
    store = RunStore(out)
    start = time.monotonic()
    rows = []
    try:
        source = code_manifest(ROOT)
        if len(seal) != 40 or source["git_commit"] != seal or source["git_status"] != "":
            raise ValueError("Scientific execution requires the clean specified source commit")
        store.write("source.json", source)
        design, reqs, binding = prepare()
        store.write("binding.json", binding)
        with gzip.GzipFile(filename=str(out / "requests.jsonl.gz"), mode="xb", mtime=0) as stream:
            for q in reqs:
                stream.write((canonical(q) + "\n").encode())
        args = SimpleNamespace(backend="hf", profile=str(ROOT / design["model_profile"]),
                               checkpoint=str(checkpoint), slots=8, cut=None, writer=None, transport=None,
                               encoder_bridge=None, reviewed_local_code=None, **{
                                   k: design[k] for k in ("device", "dtype", "cpu_threads", "max_context",
                                                         "max_new_tokens", "wall_seconds")})
        host, identity = make_host(args)
        if identity["checkpoint_sha256"] != design["checkpoint_sha256"]:
            raise ValueError("Checkpoint differs from the protocol")
        host.max_forwards = design["max_forwards"]
        host.max_processed_tokens = design["max_processed_tokens"]
        store.write("host.json", dict(host.identity, device=design["device"], dtype=design["dtype"],
                                      gpu=torch.cuda.get_device_name(), generation_policy="greedy_fresh_kv_per_prompt",
                                      max_forwards=host.max_forwards, max_processed_tokens=host.max_processed_tokens))
        parity = host.native_parity()
        store.write("native_parity.json", parity)
        if not parity["exact"]:
            raise ValueError("Native hook parity failed")
        for q in reqs:
            response = generate_kv(host, q["prompt"])
            generation = response.receipt()
            generation.pop("text")
            row = dict(id=q["id"], request_sha256=fingerprint(q), text=response.text, generation=generation)
            store.append("reports.jsonl", row)
            rows.append(row)
            if len(rows) % 128 == 0:
                print(canonical(dict(generated=len(rows), total=len(reqs))), flush=True)
        store.write("records_closed.json", dict(unique_generations=len(rows), compute=host.meter.snapshot(),
                    reports_sha256=file_sha(out / "reports.jsonl"), requests_file_sha256=file_sha(out / "requests.jsonl.gz"),
                    elapsed_seconds=time.monotonic()-start))
        del host
        torch.cuda.empty_cache()
        summary, scores = analyze(design, reqs, rows)
        store.write("summary.json", summary)
        store.write("scores.json", scores)
        verify_files(ROOT, read(STUDY / "INPUTS.json")["files"])
        store.write("terminal.json", dict(status="ORACLE_ROW_READOUT_CLOSED", generated=len(rows),
                    elapsed_seconds=time.monotonic()-start, decision=summary["decision"], fits=0,
                    native_choice_scoring_calls=0, qualification_loaded=False))
    except Exception as exc:
        store.write("terminal.json", dict(status="TECHNICAL_STOP", generated=len(rows),
                    elapsed_seconds=time.monotonic()-start, error_type=type(exc).__name__, error=str(exc)))
        raise


def reduce_run(folder):
    design, reqs, binding = prepare()
    terminal = read(folder / "terminal.json")
    closed = read(folder / "records_closed.json")
    source = read(folder / "source.json")
    # Verify the parts used to regenerate the instrument and its scores after publication.
    for rel in ("src/selffold/oracle_row_readout.py", "src/selffold/actual_report.py",
                "src/selffold/lm_writer.py", "src/selffold/lm_writer_reports.py", "src/selffold/roundtrip.py",
                "scripts/measure_oracle_row_readout.py"):
        if file_sha(ROOT / rel) != source["files"][rel]:
            raise ValueError(f"Replay source differs: {rel}")
    if (terminal["status"] != "ORACLE_ROW_READOUT_CLOSED" or terminal["generated"] != len(reqs)
            or closed["unique_generations"] != len(reqs) or read(folder / "binding.json") != binding
            or file_sha(folder / "reports.jsonl") != closed["reports_sha256"]
            or file_sha(folder / "requests.jsonl.gz") != closed["requests_file_sha256"]):
        raise ValueError("Incomplete or changed run receipts")
    with gzip.open(folder / "requests.jsonl.gz", "rt", encoding="utf8") as fh:
        if [json.loads(line) for line in fh] != reqs:
            raise ValueError("Saved prompts differ from the preregistered panel")
    with (folder / "reports.jsonl").open(encoding="utf8") as fh:
        answers = [json.loads(line) for line in fh]
    summary, scores = analyze(design, reqs, answers)
    if (summary != read(folder / "summary.json") or scores != read(folder / "scores.json")
            or terminal["decision"] != summary["decision"]):
        raise ValueError("Score, summary or decision differs on replay")
    return dict(exact=True, distinct_reports=len(answers), exact_scores=len(scores), profiles=len(summary["profiles"]),
                decision=summary["decision"], generation_calls=0, source_seal=source["git_commit"])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", type=Path)
    mode.add_argument("--reduce", type=Path)
    mode.add_argument("--check-design", action="store_true")
    cli.add_argument("--checkpoint", type=Path)
    cli.add_argument("--seal")
    args = cli.parse_args()
    torch.set_default_device("cpu")
    torch.set_num_threads(1)
    if args.check_design:
        print(canonical(prepare()[2]))
    elif args.reduce:
        print(canonical(reduce_run(args.reduce)))
    elif args.checkpoint and args.seal:
        run(args.run, args.checkpoint, args.seal)
    else:
        cli.error("--run requires --checkpoint and --seal")
