"""Matched policy renderings with the two previously fitted native recipients."""
import argparse
import json
import re
import time
from pathlib import Path

import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_continuity_recipient import DATA, batch_inputs, model_scores, tasks_for_group, training_loss
from probe_recipient_trainability import CONFIG, TRAIN_GROUPS, rows
from selffold.carrier_probe import capture_description
from selffold.cli import ROOT, make_host
from selffold.continuity_data import load_groups
from selffold.recipient_learning import RecipientAdapter, batched_recipient_logits, load_recipient, save_recipient
from selffold.records import RunStore, code_manifest, digest, file_sha

STUDY = ROOT / "provenance/policy_form_20261103"
PRIOR = ROOT / "provenance/continuity_20261103"
CONTROL = PRIOR / "trainability"
FORMS = ("rule", "forward", "reverse")
CASE_CONFIG = {**CONFIG, "stage": "stage1", "fit_form": "canonical_forward_cases"}


def verify_form(group, pole, variants):
    """Interpret the literal clauses independently of role labels or model scores."""
    ledger = group["poles"][pole]["truth_ledger"]
    expected = {" and ".join(f"{flag} is {'on' if r['world'][flag] else 'off'}"
                            for flag in group["flags"]): r["choice"] for r in ledger}
    clauses = [re.findall(r"When ([^,]+), I choose ([^.]+)\.", v["stage1"]) for v in variants]
    if len(clauses[0]) != 4 or clauses[0] != clauses[1][::-1] or dict(clauses[0]) != expected:
        raise ValueError("Case lists disagree with the world ledger")
    rule = variants[0]["stage2"]
    match = re.search(r"I choose ([^.]+?) exactly when (.+)\. In every other case, I choose ([^.]+)\.$", rule)
    if rule != variants[1]["stage2"] or match is None:
        raise ValueError("Unexpected canonical rule")
    yes, condition, no = match.groups()
    terms = condition.split(" or ")
    listed = {term[1:-1] for term in terms if term.startswith("(") and term.endswith(")")}
    if len(listed) != len(terms) or not listed <= expected.keys() or any(
            (yes if world in listed else no) != gold for world, gold in expected.items()):
        raise ValueError("Rule disagrees with the world ledger")


def material(groups):
    forms = {form: {} for form in FORMS}
    for split in groups.values():
        for group in split:
            for pole, detail in group["poles"].items():
                variants = sorted((v for v in detail["variants"] if v["frame"] == "self"), key=lambda v: v["order"])
                if [v["order"] for v in variants] != [0, 1]:
                    raise ValueError("Incomplete paired sources")
                verify_form(group, pole, variants)
                for v in variants:
                    key = "stage2/" + v["id"]
                    for form, stage, source in (("rule", "stage2", v), ("forward", "stage1", variants[0]),
                                                ("reverse", "stage1", variants[1])):
                        forms[form][key] = {"source": source[stage], "donor_key": stage + "/" + source["id"]}
    return forms


def evaluate(host, adapter, condition, endpoint, tasks, states, previous, store):
    route = "literal_base" if condition == "literal" else "matched"
    for form, splits in tasks.items():
        for split, panel in splits.items():
            for start in range(0, len(panel), 8):
                batch = panel[start:start + 8]
                with torch.no_grad():
                    scored = model_scores(host, adapter, batch, states[form], route)
                for task, scores in zip(batch, scored, strict=True):
                    record = {k: task[k] for k in ("case","group","pole","source_order","world","label_order","gold")}
                    record.update(condition=condition, endpoint=endpoint, form=form, split=split,
                                  question_sha256=digest(task["question"]), source_sha256=digest(task["source"]), **scores)
                    if form == "rule" and endpoint in ("initial", "rule_fit"):
                        phase = "initial" if endpoint == "initial" else "final"
                        ref = previous[condition, phase, task["case"], task["world"], task["label_order"]]
                        if any(record[k] != ref[k] for k in ("gold","source_sha256","question_sha256","A","B","greedy_token")):
                            raise ValueError("Original decision replay mismatch")
                    store.append("decisions.jsonl", record)


def run(args, store):
    source = code_manifest(ROOT)
    store.write("source.json", source)
    if source["git_status"] != "" or not source["git_commit"]:
        raise ValueError("Clean source required")
    train = {g["id"]: g for g in load_groups(DATA, "train")}
    groups = {"train": [train[k] for k in TRAIN_GROUPS], "dev": load_groups(DATA,"dev")[:2]}
    forms = material(groups)
    schedule = json.loads((CONTROL/"schedule.json").read_text())
    if len(schedule) != 64 or any(schedule.count(k) != 16 for k in TRAIN_GROUPS):
        raise ValueError("Original schedule differs")
    store.write("request.json", {"plan_sha256": file_sha(STUDY/"PLAN.md"), "material_sha256": digest(forms),
                "data_sha256": file_sha(DATA/"manifest.json"), "prior_decisions_sha256": file_sha(CONTROL/"decisions.jsonl"),
                "schedule_sha256": digest(schedule), "training": CASE_CONFIG, "forms": list(FORMS)})
    host, checkpoint = make_host(args)
    store.write("host.json", host.identity)
    store.write("native_parity.json", parity := host.native_parity())
    if (not parity["exact"] or checkpoint["checkpoint_sha256"] != CHECKPOINT_SHA
            or host.identity != json.loads((CONTROL/"host.json").read_text())):
        raise ValueError("Native host/geometry changed")
    before = parameter_sha(host.model)
    old = {r["key"]:r for r in rows(PRIOR/"recipient/donors.jsonl")}
    states, cache = {}, {}
    for form, panel in forms.items():
        states[form] = {}
        for key, row in panel.items():
            text = row["source"]
            if text not in cache:
                cache[text] = capture_description(host, text, "workspace")[0]
            state = cache[text]
            donor = old[row["donor_key"]]
            if state.sha() != donor["state_sha256"] or digest(text) != donor["source_sha256"]:
                raise ValueError("Archived source capture mismatch")
            states[form][key] = state.slots
            store.append("donors.jsonl", {"form": form, "key": key, "reference": row["donor_key"],
                         "source_sha256": digest(text), "state_sha256": state.sha()})
    tasks = {f:{s:[t for g in gl for t in tasks_for_group(g, panel, "stage2", "self", 0)]
                  for s, gl in groups.items()} for f, panel in forms.items()}
    previous = {(r["condition"],r["phase"],r["case"],r["world"],r["label_order"]): r for r in rows(CONTROL/"decisions.jsonl")}
    prior_fits = {r["condition"]:r for r in rows(CONTROL/"fits.jsonl")}
    ids = [host.tokenizer.encode(x, add_special_tokens=False) for x in ("A","B")]
    if any(len(x) != 1 for x in ids):
        raise ValueError("Expected one-token labels")
    label_ids = [x[0] for x in ids]
    for condition in ("literal","latent"):
        state_maps = states if condition == "latent" else {f:dict.fromkeys(panel) for f,panel in forms.items()}
        base = prior_fits[condition]
        adapter, _, actual = load_recipient(CONTROL/base["artifact"], host)
        try:
            if actual != base["sha256"]:
                raise ValueError("Archived adapter changed")
            evaluate(host, adapter, condition, "rule_fit", tasks, state_maps, previous, store)
        finally:
            adapter.close()
        adapter = RecipientAdapter(host, rank=16, last_layers=4, seed=47, scale=1.)
        try:
            initial = parameter_sha(adapter)
            if initial != base["initial_parameter_sha256"]:
                raise ValueError("Initial parameters differ")
            evaluate(host, adapter, condition, "initial", tasks, state_maps, previous, store)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=.0003, weight_decay=0.)
            route = "literal_base" if condition == "literal" else "matched"
            start = host.meter.snapshot()
            for step, group_id in enumerate(schedule, 1):
                panel = [t for t in tasks["forward"]["train"] if t["group"] == group_id]
                if len(panel) != 32:
                    raise ValueError("Incomplete training batch")
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for offset in range(0, 32, 8):
                    batch = panel[offset:offset+8]
                    logits = batched_recipient_logits(host, adapter, *batch_inputs(host,batch,state_maps["forward"],route))
                    targets = torch.tensor([label_ids[0 if t["gold"] == "A" else 1] for t in batch], device=host.device)
                    loss = training_loss(logits, targets, label_ids, "native_binary")
                    (loss / 4).backward()
                    losses.append(loss.item())
                if any(p.grad is not None for p in host.model.parameters()) or not any(p.grad is not None for p in adapter.parameters()):
                    raise ValueError("Gradient isolation failed")
                norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True).item()
                optimizer.step()
                records = [{k:v for k,v in t.items() if k != "source"} | {"split":"train"} for t in panel]
                store.append("training.jsonl", {"condition":condition,"step":step,"group":group_id,"losses":losses,
                                               "gradient_norm":norm,"tasks_sha256":digest(records)})
            end = host.meter.snapshot()
            history = [r["tasks_sha256"] for r in rows(store.path/"training.jsonl") if r["condition"] == condition]
            if digest(history) != base["task_order_sha256"]:
                raise ValueError("Questions, gold or training order differ from the original schedule")
            artifact = store.path / ("case_"+condition+".npz")
            first = [t for t in tasks["forward"]["train"] if t["group"] == schedule[0]][:8]
            inputs = batch_inputs(host, first, state_maps["forward"], route)
            with torch.no_grad():
                expected_logits = batched_recipient_logits(host, adapter, *inputs).clone()
            param_sha = parameter_sha(adapter)
            sha = save_recipient(artifact,adapter,host.identity,dataset_sha256=file_sha(DATA/"manifest.json"),
                                 source_sha256=source["source_sha256"], training={**CASE_CONFIG,"input_condition":condition,
                                 "form":"forward","plan_sha256":file_sha(STUDY/"PLAN.md"),"schedule_sha256":digest(schedule)})
            adapter.close()
            adapter, _, reloaded = load_recipient(artifact, host)
            with torch.no_grad():
                restored_logits = batched_recipient_logits(host, adapter, *inputs)
            if sha != reloaded or parameter_sha(adapter) != param_sha or not torch.equal(expected_logits, restored_logits):
                raise ValueError("Adapter serialization mismatch")
            store.append("fits.jsonl", {"condition":condition,"initial_parameter_sha256":initial,"final_parameter_sha256":param_sha,
                                        "artifact":artifact.name,"sha256":sha,"task_order_sha256":digest(history),
                                        "training_forwards":end["forwards"]-start["forwards"],
                                        "training_processed_tokens":end["processed_tokens"]-start["processed_tokens"]})
            evaluate(host, adapter, condition, "case_fit", tasks, state_maps, previous, store)
        finally:
            adapter.close()
    after = parameter_sha(host.model)
    if before != after:
        raise ValueError("Base weights changed")
    return {"status":"COMPLETED_POLICY_FORM_DEVELOPMENT","updates":128,"replayed_original_decisions":len(previous),
            "unique_donors":len(cache),"backbone_sha256_before":before,"backbone_sha256_after":after,
            "compute":host.meter.snapshot(),"G_trained":False,"qualification_content_loaded":False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.set_defaults(backend="hf", cpu_threads=1, device="cuda:0", dtype="bfloat16", slots=8, cut=None, writer=None,
                   max_context=1024, max_new_tokens=None, until_eos=True, transport=None, encoder_bridge=None,
                   reviewed_local_code=None, wall_seconds=900, profile=ROOT/"configs/models/gemma2b.yaml")
    args = p.parse_args()
    store = RunStore(args.out)
    start = time.monotonic()
    try:
        result = run(args, store)
    except Exception as exc:
        store.write("terminal.json", {"status":"INVALID_STOPPED","error":str(exc),"type":type(exc).__name__,
                                     "elapsed_seconds":time.monotonic()-start})
        raise
    store.write("terminal.json", result | {"elapsed_seconds":time.monotonic()-start})


if __name__ == "__main__":
    main()
