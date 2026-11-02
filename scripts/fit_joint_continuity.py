"""Fit one question-independent source encoder with its scoped native recipient."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_continuity_recipient import DATA, STUDY, STAGES, batch_inputs, evaluate, tasks_for_group, training_loss
from selffold.cli import ROOT, make_host
from selffold.continuity_data import examples, load_groups
from selffold.recipient_learning import (
    RecipientAdapter, batched_recipient_logits, differentiable_source_slots, load_recipient, save_recipient,
)
from selffold.records import RunStore, State, code_manifest, digest, file_sha
from selffold.resources import inventory

CONFIG = {
    "candidate_order": ["joint_source"], "selection": "fixed_endpoint_only", "updates": 256,
    "effective_batch": 32, "microbatch": 8, "learning_rate": .0003, "weight_decay": 0.,
    "rank": 16, "source_layers_before_cut": 4, "recipient_last_layers": 4,
    "source_seed": 947, "recipient_seed": 47, "scale": 1., "gradient_clip": 1.,
    "objective": "native_binary", "loss": "conditional_A_or_B_cross_entropy_on_native_output_logits",
    "train_stages": list(STAGES), "source_forward_length": 256,
    "source_questions_in_capture": False, "source_captures_per_update": 4,
    "native_forwards_cap": 30000, "processed_tokens_cap": 30000000,
    "source_capture_stop": "before_pre_cut_layer_with_gradient_tensor",
}


def load_reference(folder):
    request = json.loads((folder / "request.json").read_text())
    terminal = json.loads((folder / "terminal.json").read_text())
    if (terminal["status"] != "COMPLETED_RECIPIENT_DEVELOPMENT" or terminal["G_trained"]
            or terminal["qualification_content_loaded"] or request["checkpoint_sha256"] != CHECKPOINT_SHA
            or request["data_sha256"] != file_sha(DATA / "manifest.json")):
        raise ValueError("Need the completed original recipient reference and identical data")
    for key in ("updates", "effective_batch", "microbatch", "rank", "train_stages"):
        if request[key] != CONFIG[key]:
            raise ValueError("Joint fit differs from the fixed reference schedule envelope")
    schedule = json.loads((folder / "schedule.json").read_text())
    if len(schedule) != CONFIG["updates"]:
        raise ValueError("Incomplete reference schedule")
    donors = {row["key"]: row for row in read_rows(folder / "donors.jsonl")}
    fits = [row for row in read_rows(folder / "fits.jsonl") if row["candidate"] == "raw"]
    if len(donors) != 4320 or len(fits) != 1:
        raise ValueError("Incomplete original donor or raw-fit reference")
    reference = {"request_sha256": file_sha(folder / "request.json"),
                 "terminal_sha256": file_sha(folder / "terminal.json"),
                 "schedule_sha256": digest(schedule), "donors_sha256": file_sha(folder / "donors.jsonl"),
                 "literal_base": {"functional_file_sha256": file_sha(folder / "functional.jsonl"),
                                  "candidate": "base", "route": "literal_base"}}
    return schedule, donors, fits[0], reference


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def gradient_norm(module):
    norms = [p.grad.float().norm() for p in module.parameters() if p.grad is not None]
    if not norms:
        raise ValueError("A jointly trained module received no gradient")
    value = torch.linalg.vector_norm(torch.stack(norms)).item()
    if not torch.isfinite(torch.tensor(value)) or value <= 0:
        raise ValueError("A jointly trained module received a nonpositive or nonfinite gradient")
    return value


def source_parity(host, encoder, cases, donors, store):
    chosen = []
    for split in ("train", "dev"):
        for stage in STAGES:
            key = next(k for k, row in cases.items() if row["split"] == split and row["input_stage"] == stage)
            chosen.append(key)
    with torch.no_grad():
        for key in chosen:
            slots, layout = differentiable_source_slots(host, encoder, cases[key]["source"], forward_length=256)
            actual = State(slots=slots).sha()
            store.append("initial_source_parity.jsonl", {"key": key, "state_sha256": actual,
                "baseline_state_sha256": donors[key]["state_sha256"], "positions": layout["capture_positions"]})
            if actual != donors[key]["state_sha256"] or layout["capture_positions"] != donors[key]["positions"]:
                raise ValueError("Zero-initialized source encoder differs from the original capture")


def main_run(args, store):
    source = code_manifest(ROOT)
    store.write("source.json", source)
    if source["git_status"] != "" or not source["git_commit"]:
        raise ValueError("Clean source required")
    schedule, old_donors, raw_fit, reference = load_reference(args.baseline_fit)
    plan_sha = file_sha(STUDY / "JOINT_PLAN.md")
    store.write("request.json", {**CONFIG, "data_sha256": file_sha(DATA / "manifest.json"),
        "plan_sha256": plan_sha, "protocol_sha256": file_sha(ROOT / "docs/FUNCTIONAL_CONTINUITY.md"),
        "scorer_sha256": file_sha(STUDY / "summarize.py"), "checkpoint_sha256": CHECKPOINT_SHA,
        "baseline_reference": reference})
    store.write("schedule.json", schedule)
    store.write("environment.before.json", inventory(ROOT))
    groups = {split: {g["id"]: g for g in load_groups(DATA, split)} for split in ("train", "dev")}
    cases = {stage + "/" + row["id"]: {**row, "split": split} for split in groups for stage in STAGES
             for row in examples(DATA, split, stage)}
    if len(cases) != 4320:
        raise ValueError("Unexpected source denominator")
    host, checkpoint = make_host(args)
    host.max_forwards = CONFIG["native_forwards_cap"]
    host.max_processed_tokens = CONFIG["processed_tokens_cap"]
    store.write("checkpoint.json", checkpoint)
    store.write("host.json", host.identity)
    if checkpoint["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Unexpected checkpoint")
    label_ids = [host.tokenizer.encode(label, add_special_tokens=False) for label in ("A", "B")]
    if any(len(ids) != 1 for ids in label_ids) or label_ids[0] == label_ids[1]:
        raise ValueError("Need two distinct native one-token labels")
    native_ids = [ids[0] for ids in label_ids]
    label_map = dict(zip(("A", "B"), native_ids))
    backbone = parameter_sha(host.model)
    encoder = RecipientAdapter(host, rank=16, last_layers=4, seed=947, scale=1.,
        layer_indices=list(range(host.cut-4, host.cut)), scope="source_encoding")
    recipient = RecipientAdapter(host, rank=16, last_layers=4, seed=47, scale=1.)
    parameters = [*encoder.parameters(), *recipient.parameters()]
    if sum(p.numel() for p in parameters) > 2_000_000:
        raise ValueError("Joint trainable parameter cap exceeded")
    initial_encoder = parameter_sha(encoder)
    initial_recipient = parameter_sha(recipient)
    if initial_recipient != raw_fit["initial_parameters_sha256"]:
        raise ValueError("Recipient initialization differs from the original raw candidate")
    store.write("native_parity.before.json", parity := host.native_parity())
    if not parity["exact"]:
        raise ValueError("Native parity failed before joint fitting")
    source_parity(host, encoder, cases, old_donors, store)
    optimizer = torch.optim.AdamW(parameters, lr=.0003, weight_decay=0.)
    before = host.meter.snapshot()
    source_forwards = source_tokens = task_forwards = task_tokens = 0
    for step, entry in enumerate(schedule, 1):
        optimizer.zero_grad(set_to_none=True)
        tasks = tasks_for_group(groups["train"][entry["group"]], cases, entry["stage"], entry["frame"], entry["wording"])
        if len(tasks) != 32:
            raise ValueError("Unbalanced effective batch")
        losses, source_keys = [], []
        for start in range(0, 32, 8):
            batch = tasks[start:start+8]
            keys = {task["key"] for task in batch}
            if len(keys) != 1:
                raise ValueError("A source microbatch must share exactly one description")
            key = keys.pop()
            meter = host.meter.snapshot()
            slots, _ = differentiable_source_slots(host, encoder, cases[key]["source"], forward_length=256)
            if not slots.requires_grad:
                raise ValueError("Source capture detached the encoder learning path")
            after_source = host.meter.snapshot()
            source_forwards += after_source["forwards"] - meter["forwards"]
            source_tokens += after_source["processed_tokens"] - meter["processed_tokens"]
            inputs = batch_inputs(host, batch, {key: slots}, "matched")
            logits = batched_recipient_logits(host, recipient, *inputs)
            targets = torch.tensor([label_map[task["gold"]] for task in batch], device=host.device)
            loss = training_loss(logits, targets, native_ids, "native_binary")
            (loss / 4).backward()
            after_task = host.meter.snapshot()
            task_forwards += after_task["forwards"] - after_source["forwards"]
            task_tokens += after_task["processed_tokens"] - after_source["processed_tokens"]
            losses.append(loss.item())
            source_keys.append(key)
        if any(p.grad is not None for p in host.model.parameters()):
            raise ValueError("Frozen backbone received a gradient")
        encoder_norm, recipient_norm = gradient_norm(encoder), gradient_norm(recipient)
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True).item()
        optimizer.step()
        store.append("training.jsonl", {"candidate": "joint_source", "step": step, **entry, "losses": losses,
            "source_keys": source_keys, "encoder_gradient_norm": encoder_norm,
            "recipient_gradient_norm": recipient_norm, "gradient_norm": total_norm,
            "tasks_sha256": digest([{k: v for k, v in task.items() if k != "source"} for task in tasks])})
    after = host.meter.snapshot()
    if source_forwards != 1024 or task_forwards != raw_fit["training_forwards"] or task_tokens != raw_fit["training_processed_tokens"]:
        raise ValueError("Joint source or recipient training compute differed from the fixed schedule")
    final_encoder, final_recipient = parameter_sha(encoder), parameter_sha(recipient)
    pair_binding = digest({"encoder_parameters_sha256": final_encoder, "recipient_parameters_sha256": final_recipient,
                           "data_sha256": file_sha(DATA / "manifest.json"), "schedule_sha256": digest(schedule)})
    training = {**CONFIG, "candidate": "joint_source", "schedule_sha256": digest(schedule),
                "plan_sha256": plan_sha, "joint_parameter_binding_sha256": pair_binding}
    encoder_path, recipient_path = store.path / "encoder_source.npz", store.path / "recipient_joint_source.npz"
    encoder_sha = save_recipient(encoder_path, encoder, host.identity, dataset_sha256=file_sha(DATA / "manifest.json"),
        source_sha256=source["source_sha256"], training={**training, "role": "source_encoding"})
    recipient_sha = save_recipient(recipient_path, recipient, host.identity, dataset_sha256=file_sha(DATA / "manifest.json"),
        source_sha256=source["source_sha256"], training={**training, "role": "task_readout"})
    encoder.close()
    recipient.close()
    encoder, _, encoder_restored = load_recipient(encoder_path, host)
    recipient, _, recipient_restored = load_recipient(recipient_path, host)
    if (encoder_restored != encoder_sha or recipient_restored != recipient_sha
            or parameter_sha(encoder) != final_encoder or parameter_sha(recipient) != final_recipient):
        raise ValueError("Joint artifacts failed serialization roundtrip")
    store.write("fits.json", {"candidate": "joint_source", "initial_encoder_sha256": initial_encoder,
        "initial_recipient_sha256": initial_recipient, "final_encoder_sha256": final_encoder,
        "final_recipient_sha256": final_recipient, "encoder_artifact_sha256": encoder_sha,
        "recipient_artifact_sha256": recipient_sha, "joint_parameter_binding_sha256": pair_binding,
        "source_trainable_parameters": sum(p.numel() for p in encoder.parameters()),
        "recipient_trainable_parameters": sum(p.numel() for p in recipient.parameters()),
        "training_forwards": after["forwards"]-before["forwards"],
        "training_processed_tokens": after["processed_tokens"]-before["processed_tokens"],
        "source_forwards": source_forwards, "source_processed_tokens": source_tokens,
        "source_prefix_layers": host.cut, "task_forwards": task_forwards,
        "task_processed_tokens": task_tokens, "task_decoder_layers": host.depth})
    # Complete all frozen source encodings before constructing development questions.
    raw, cache = {}, {}
    with torch.no_grad():
        for key, row in cases.items():
            text = row["source"]
            if text not in cache:
                slots, layout = differentiable_source_slots(host, encoder, text, forward_length=256)
                cache[text] = (slots.detach().clone(), State(slots=slots).sha(), layout["capture_positions"])
            slots, sha, positions = cache[text]
            raw[key] = slots
            store.append("donors.jsonl", {"key": key, "split": row["split"], "stage": row["input_stage"],
                "group": row["group"], "source_sha256": digest(text), "state_sha256": sha,
                "positions": positions, "encoder_artifact_sha256": encoder_sha})
        records = evaluate(host, recipient, "joint_source", cases, groups, raw, lambda value: value, store)
    spec = importlib.util.spec_from_file_location("continuity_summary", STUDY / "summarize.py")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    summary = scorer.score_candidate(records, "joint_source")
    store.write("summary_joint_source.json", summary)
    if parameter_sha(encoder) != final_encoder or parameter_sha(recipient) != final_recipient:
        raise ValueError("Frozen evaluation changed a joint adapter")
    store.write("native_parity.after.json", parity := host.native_parity())
    if not parity["exact"] or parameter_sha(host.model) != backbone:
        raise ValueError("Frozen backbone parity or parameter binding changed")
    encoder.close()
    recipient.close()
    store.write("environment.after.json", inventory(ROOT))
    passed = summary["recipient_screen"]["pass"]
    return {"status": "COMPLETED_JOINT_RECIPIENT_DEVELOPMENT", "screen_pass": passed,
        "selected": "joint_source" if passed else None, "optimizer_updates": 256, "G_trained": False,
        "backbone_sha256_before": backbone, "backbone_sha256_after": parameter_sha(host.model),
        "encoder_artifact_sha256": encoder_sha, "recipient_artifact_sha256": recipient_sha,
        "joint_parameter_binding_sha256": pair_binding, "source_encoder_initial_sha256": initial_encoder,
        "source_encoder_final_sha256": final_encoder, "recipient_initial_sha256": initial_recipient,
        "recipient_final_sha256": final_recipient, "logical_donor_rows": len(cases),
        "unique_donor_forwards": len(cache), "qualification_content_loaded": False,
        "compute": host.meter.snapshot()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-fit", type=Path, required=True)
    parser.set_defaults(backend="hf", cpu_threads=1, device="cuda:0", dtype="bfloat16", slots=8, cut=None,
        writer=None, max_context=1024, max_new_tokens=None, until_eos=True, transport=None, encoder_bridge=None,
        reviewed_local_code=None, wall_seconds=3600, profile=ROOT / "configs/models/gemma2b.yaml")
    args = parser.parse_args()
    store, started = RunStore(args.out), time.monotonic()
    try:
        terminal = main_run(args, store)
    except Exception as exc:
        store.write("terminal.json", {"status": "INVALID_STOPPED", "error": str(exc), "type": type(exc).__name__,
                    "elapsed_seconds": time.monotonic()-started})
        raise
    store.write("terminal.json", {**terminal, "elapsed_seconds": time.monotonic()-started})


if __name__ == "__main__":
    main()
