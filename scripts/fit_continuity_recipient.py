"""Learn the bounded native recipient for the two-stage continuity development bank."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from selffold import prompts
from selffold.carrier_probe import capture_description
from selffold.cli import ROOT, make_host
from selffold.continuity_data import examples, load_groups, question_cases, FRAMES
from selffold.recipient_learning import RecipientAdapter, batched_recipient_logits, save_recipient, load_recipient
from selffold.records import RunStore, State, code_manifest, digest, file_sha
from selffold.resources import inventory

DATA = ROOT / "data/continuity_20261103"
STUDY = ROOT / "provenance/continuity_20261103"
STAGES = ("source", "stage1", "stage2")
ROUTES = ("matched", "opposed", "empty", "zero", "random", "unrelated")
CONFIG = {"rank": 16, "last_layers": 4, "scale": 1., "seed": 47, "updates": 256,
          "effective_batch": 32, "microbatch": 8, "learning_rate": .001, "weight_decay": 0.,
          "gradient_clip": 1., "candidate_order": ["raw", "standardized"],
          "second_candidate": "only_if_raw_misses_fixed_development_screen",
          "selection": "first_fixed_endpoint_meeting_screen_else_none",
          "loss": "full_vocabulary_gold_native_A_or_B_cross_entropy",
          "train_stages": list(STAGES), "source_questions_in_capture": False,
          "standardization": "mu + row_mean_std * (z-mu)/max(std,0.01*row_mean_std)",
          "native_forwards_cap": 30000, "processed_tokens_cap": 30000000}


def run_config(objective):
    config = deepcopy(CONFIG)
    config["objective"] = objective
    if objective == "native_binary":
        config.update(candidate_order=["raw_binary"], second_candidate="none",
                      loss="conditional_A_or_B_cross_entropy_on_native_output_logits",
                      selection="fixed_raw_binary_endpoint_meeting_existing_screen_else_none",
                      preprocessing="identity_raw")
        config.pop("standardization")
    elif objective != "native_full":
        raise ValueError("Unknown recipient training objective")
    return config


def training_loss(logits, targets, label_token_ids, objective):
    """Train on returned native logits, retaining full-vocabulary evaluation."""
    if objective == "native_full":
        return torch.nn.functional.cross_entropy(logits, targets)
    if objective != "native_binary":
        raise ValueError("Unknown recipient training objective")
    if len(label_token_ids) != 2 or label_token_ids[0] == label_token_ids[1]:
        raise ValueError("Binary recipient training requires distinct A/B token IDs")
    indices = torch.tensor(label_token_ids, device=logits.device, dtype=torch.long)
    matches = targets[:, None] == indices[None, :]
    if not torch.all(matches.sum(-1) == 1):
        raise ValueError("Binary recipient target must be a native A or B token")
    classes = matches.long().argmax(-1)
    return torch.nn.functional.cross_entropy(logits.index_select(-1, indices), classes)


def load_baseline(folder, config):
    """Bind the closed full-vocabulary fit before any fresh candidate is trained."""
    if folder is None:
        raise ValueError("Binary training requires --baseline-fit")
    folder = Path(folder)
    request = json.loads((folder / "request.json").read_text())
    terminal = json.loads((folder / "terminal.json").read_text())
    if (terminal["status"] != "COMPLETED_RECIPIENT_DEVELOPMENT" or terminal["screen_pass"]
            or terminal["selected"] is not None or terminal["G_trained"]
            or terminal["qualification_content_loaded"]):
        raise ValueError("Need the completed, closed recipient baseline")
    if (request["loss"] != CONFIG["loss"] or request.get("objective", "native_full") != "native_full"
            or request["checkpoint_sha256"] != CHECKPOINT_SHA
            or request["data_sha256"] != file_sha(DATA / "manifest.json")
            or request["scorer_sha256"] != file_sha(STUDY / "summarize.py")):
        raise ValueError("Baseline loss, checkpoint, data or scorer binding differs")
    matched_keys = ("rank", "last_layers", "scale", "seed", "updates", "effective_batch", "microbatch",
                    "learning_rate", "weight_decay", "gradient_clip", "train_stages", "source_questions_in_capture")
    if any(request[key] != config[key] for key in matched_keys):
        raise ValueError("Fresh binary fit differs from the baseline training envelope")
    fits = [json.loads(line) for line in (folder / "fits.jsonl").read_text().splitlines()]
    raw = [fit for fit in fits if fit["candidate"] == "raw"]
    if len(raw) != 1:
        raise ValueError("Need exactly one completed raw baseline")
    raw = raw[0]
    if Path(raw["artifact"]).name != raw["artifact"] or file_sha(folder / raw["artifact"]) != raw["sha256"]:
        raise ValueError("Baseline raw artifact binding differs")
    schedule = json.loads((folder / "schedule.json").read_text())
    if len(schedule) != config["updates"]:
        raise ValueError("Baseline schedule length differs")
    donors = {}
    for line in (folder / "donors.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["key"] in donors:
            raise ValueError("Duplicate baseline source capture")
        donors[row["key"]] = row
    if len(donors) != 4320:
        raise ValueError("Baseline source capture denominator differs")
    reference = {
        "terminal_sha256": file_sha(folder / "terminal.json"),
        "request_sha256": file_sha(folder / "request.json"),
        "fits_sha256": file_sha(folder / "fits.jsonl"),
        "donors_sha256": file_sha(folder / "donors.jsonl"),
        "schedule_sha256": digest(schedule),
        "raw_artifact_sha256": raw["sha256"],
        "literal_base": {"functional_file_sha256": file_sha(folder / "functional.jsonl"),
                         "candidate": "base", "route": "literal_base", "reuse": "immutable_prior_run_reference"},
    }
    return {"raw_fit": raw, "schedule": schedule, "donors": donors,
            "host": json.loads((folder / "host.json").read_text()), "reference": reference}


def load_numerical_audit(folder, baseline):
    if folder is None:
        raise ValueError("Binary training requires --audit")
    folder = Path(folder)
    terminal = json.loads((folder / "terminal.json").read_text())
    request = json.loads((folder / "request.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if (terminal["status"] != "COMPLETED_READ_ONLY_RECIPIENT_NUMERICAL_AUDIT"
            or terminal["optimizer_updates"] != 0 or terminal["qualification_content_loaded"]
            or terminal["checkpoint_sha256"] != CHECKPOINT_SHA):
        raise ValueError("Need a completed read-only numerical audit")
    if (request["fit_terminal_sha256"] != baseline["reference"]["terminal_sha256"]
            or request["data_sha256"] != file_sha(DATA / "manifest.json")):
        raise ValueError("Numerical audit does not bind this baseline and dataset")
    ties = summary["raw"]["native_ties_without_raw_ties"]
    count = summary["raw"]["scores"]["all"]["native_capped"]["n"]
    if count <= 0 or not 0 <= ties <= count or ties / count < .10:
        raise ValueError("Numerical audit did not meet the binary-phase admission condition")
    return {"terminal_sha256": file_sha(folder / "terminal.json"),
            "summary_sha256": file_sha(folder / "summary.json"),
            "request_sha256": file_sha(folder / "request.json"),
            "raw_native_ties_without_raw_ties": ties, "raw_scores": count,
            "admission_minimum_fraction": .10}


def batch_inputs(host, tasks, state_map, route):
    contexts, positions, score_at = [], [], []
    values = []
    for task in tasks:
        supplied = state_map[task["key"]]
        text = task["source"] if route == "literal_base" else None
        ids, pos = host.render(prompts.task_prompt(task["question"], text=text))
        score_at.append(len(ids)-1)
        contexts.append(ids + [host.tokenizer.encode("A", add_special_tokens=False)[0]])
        if supplied is not None:
            positions.append(pos[prompts.INPUT_MARKER])
            values.append(supplied)
    length = max(map(len, contexts))
    padding = host.tokenizer.pad_token_id
    ids = [row + [padding]*(length-len(row)) for row in contexts]
    mask = [[1]*len(row) + [0]*(length-len(row)) for row in contexts]
    states = torch.stack(values) if values else None
    return ids, mask, states, (positions if values else None), score_at


def tasks_for_group(group, cases, stage, frame=None, wording=None):
    output = []
    for pole, detail in group["poles"].items():
        for variant in detail["variants"]:
            if frame is not None and variant["frame"] != frame:
                continue
            key = stage + "/" + variant["id"]
            for q in question_cases(group, pole):
                if wording is not None and q["wording"] != wording:
                    continue
                output.append({"key": key, "case": variant["id"], "stage": stage, "group": group["id"],
                    "pole": pole, "frame": variant["frame"], "source_order": variant["order"],
                    "world": q["world_index"], "wording": q["wording"], "label_order": q["order"],
                    "question": q["question"], "gold": q["answer"], "source": cases[key]["source"]})
    return output


def model_scores(host, adapter, tasks, states, route):
    args = batch_inputs(host, tasks, states, route)
    logits = batched_recipient_logits(host, adapter, *args)
    label_ids = [host.tokenizer.encode(x, add_special_tokens=False)[0] for x in ("A", "B")]
    lp = logits.log_softmax(-1)
    best = logits.argmax(-1).tolist()
    return [{"A": lp[i, label_ids[0]].item(), "B": lp[i, label_ids[1]].item(),
             "greedy_token": token, "greedy_label": ("A" if token == label_ids[0] else "B" if token == label_ids[1] else None)}
            for i, token in enumerate(best)]


def evaluate(host, adapter, candidate, cases, groups, raw, preprocess, store, *, literal=False):
    records = []
    normal = {key: preprocess(value) for key, value in raw.items()}
    first_train = next(g for g in groups["train"].values())
    unrelated = {stage: normal[stage + "/" + first_train["poles"]["plus"]["variants"][0]["id"]] for stage in STAGES}
    train_values = torch.stack([v.float() for k, v in raw.items() if cases[k]["split"] == "train"])
    mean_norm = train_values.norm(dim=-1).mean().item()
    direction = torch.nn.functional.normalize(torch.randn((host.slots, host.width),
                generator=torch.Generator().manual_seed(9292)), dim=-1)
    random = preprocess((direction * mean_norm).to(host.device, host.dtype))
    zero = preprocess(torch.zeros((host.slots, host.width), device=host.device, dtype=host.dtype))
    for stage in STAGES:
        tasks = [t for g in groups["dev"].values() for t in tasks_for_group(g, cases, stage)]
        routes = ("literal_base",) if literal else ROUTES
        for route in routes:
            state_map = {}
            for task in tasks:
                key = task["key"]
                if route == "matched":
                    state = normal[key]
                elif route == "opposed":
                    other = key.replace("-"+task["pole"]+"-", "-"+("minus" if task["pole"] == "plus" else "plus")+"-")
                    state = normal[other]
                elif route in ("empty", "literal_base"):
                    state = None
                elif route == "zero":
                    state = zero
                elif route == "random":
                    state = random
                else:
                    state = unrelated[stage]
                state_map[key] = state
            fingerprints = {key: ("literal:"+digest(cases[key]["source"]) if literal else
                            State(slots=value).sha() if value is not None else "empty")
                            for key, value in state_map.items()}
            # Repeated controls preserve their logical denominators while sharing exact computed scores.
            unique, pending = {}, []
            for task in tasks:
                cache_key = (fingerprints[task["key"]], task["question"])
                if cache_key not in unique:
                    unique[cache_key] = None
                    pending.append(task)
            with torch.no_grad():
                for start in range(0, len(pending), 8):
                    batch = pending[start:start+8]
                    scores = model_scores(host, adapter, batch, state_map, route)
                    for t, result in zip(batch, scores, strict=True):
                        unique[fingerprints[t["key"]], t["question"]] = result
            for task in tasks:
                record = {k:v for k,v in task.items() if k not in {"question", "source", "key"}}
                record.update(candidate=candidate, route=route, state_sha256=fingerprints[task["key"]],
                              question_sha256=digest(task["question"]), **unique[fingerprints[task["key"]],task["question"]])
                store.append("functional.jsonl", record)
                records.append(record)
    return records


def main_run(args, store):
    source = code_manifest(ROOT)
    store.write("source.json", source)
    if source["git_status"] != "" or not source["git_commit"]:
        raise ValueError("Need a clean source seal")
    config = run_config(args.objective)
    baseline = load_baseline(args.baseline_fit, config) if args.objective == "native_binary" else None
    audit = load_numerical_audit(args.audit, baseline) if baseline is not None else None
    request = {**config, "data_sha256": file_sha(DATA/"manifest.json"),
        "protocol_sha256": file_sha(ROOT/"docs/FUNCTIONAL_CONTINUITY.md"),
        "scorer_sha256": file_sha(STUDY/"summarize.py"), "checkpoint_sha256": CHECKPOINT_SHA}
    if baseline is not None:
        request["baseline_reference"] = baseline["reference"]
        request["numerical_audit_reference"] = audit
        request["binary_plan_sha256"] = file_sha(STUDY / "BINARY_PLAN.md")
    store.write("request.json", request)
    store.write("environment.before.json", inventory(ROOT))
    groups = {s: {g["id"]:g for g in load_groups(DATA,s)} for s in ("train", "dev")}
    cases = {}
    for split in groups:
        for stage in STAGES:
            for row in examples(DATA, split, stage):
                cases[stage+"/"+row["id"]] = {**row, "split": split}
    if len(cases) != 4320:
        raise ValueError("Unexpected source denominators")
    host, checkpoint = make_host(args)
    host.max_forwards, host.max_processed_tokens = 30000, 30000000
    store.write("checkpoint.json", checkpoint)
    store.write("host.json", host.identity)
    store.write("native_parity.json", parity := host.native_parity())
    label_ids = {x:host.tokenizer.encode(x, add_special_tokens=False) for x in ("A","B")}
    if checkpoint["checkpoint_sha256"] != CHECKPOINT_SHA or not parity["exact"] or any(len(ids)!=1 for ids in label_ids.values()):
        raise ValueError("Checkpoint, parity or one-token label mismatch")
    if baseline is not None and host.identity != baseline["host"]:
        raise ValueError("Fresh binary host differs from the baseline host")
    backbone = parameter_sha(host.model)
    # All source captures finish before constructing any task question.
    text_cache, raw = {}, {}
    for key, row in cases.items():
        text = row["source"]
        if text not in text_cache:
            state, layout = capture_description(host, text, "workspace")
            text_cache[text] = (state.slots, state.sha(), layout["capture_positions"])
        value, sha, positions = text_cache[text]
        raw[key] = value
        if baseline is not None:
            previous = baseline["donors"][key]
            if (previous["state_sha256"] != sha or previous["source_sha256"] != digest(text)
                    or previous["positions"] != positions or previous["stage"] != row["input_stage"]):
                raise ValueError("Fresh binary source capture differs from the baseline")
        store.append("donors.jsonl", {"key":key,"split":row["split"],"source_sha256":digest(text),
            "state_sha256":sha,"positions":positions,"stage":row["input_stage"],"group":row["group"]})
    training = torch.stack([raw[k].float() for k in raw if cases[k]["split"] == "train"])
    mean, std = training.mean(0), training.std(0, unbiased=False)
    row_std = std.mean(-1, keepdim=True)
    if not torch.all(row_std > 0):
        raise ValueError("No training variation")
    with (store.path/"preprocessing.npz").open("xb") as f:
        np.savez_compressed(f, mean=mean.cpu().numpy(), std=std.cpu().numpy(), row_std=row_std.cpu().numpy())
    del training
    def standardized(value):
        return (mean + row_std * (value.float()-mean)/std.clamp_min(row_std*.01)).to(host.dtype)
    transforms = {"raw":lambda x:x, "standardized":standardized}
    if args.objective == "native_binary":
        transforms = {"raw_binary":lambda x:x}
    generator = torch.Generator().manual_seed(47)
    ids = list(groups["train"])
    order = []
    while len(order) < 256:
        order.extend(torch.randperm(len(ids), generator=generator).tolist())
    schedule = []
    for index in order[:256]:
        schedule.append({"group":ids[index], "frame":FRAMES[int(torch.randint(3,(),generator=generator))],
            "stage":STAGES[int(torch.randint(3,(),generator=generator))],
            "wording":int(torch.randint(2,(),generator=generator))})
    if baseline is not None and schedule != baseline["schedule"]:
        raise ValueError("Fresh binary training schedule differs from the baseline")
    store.write("schedule.json", schedule)
    spec=importlib.util.spec_from_file_location("continuity_summary",STUDY/"summarize.py")
    scorer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    if baseline is None:
        adapter = RecipientAdapter(host, rank=16, last_layers=4, seed=47, scale=1.)
        with torch.no_grad():
            evaluate(host,adapter,"base",cases,groups,raw,transforms["raw"],store,literal=True)
        adapter.close()
    summaries, selected = {}, None
    for candidate, transform in transforms.items():
        adapter=RecipientAdapter(host,rank=16,last_layers=4,seed=47,scale=1.)
        initial=parameter_sha(adapter)
        if baseline is not None and initial != baseline["raw_fit"]["initial_parameters_sha256"]:
            adapter.close()
            raise ValueError("Fresh binary recipient initialization differs from the baseline")
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=.001,weight_decay=0.)
        normalized={k:transform(v) for k,v in raw.items()}
        before=host.meter.snapshot()
        for step, entry in enumerate(schedule,1):
            optimizer.zero_grad(set_to_none=True)
            tasks=tasks_for_group(groups["train"][entry["group"]],cases,entry["stage"],entry["frame"],entry["wording"])
            if len(tasks)!=32:
                raise ValueError("Unbalanced effective batch")
            losses=[]
            for start in range(0,32,8):
                batch=tasks[start:start+8]
                inputs=batch_inputs(host,batch,normalized,"matched")
                logits=batched_recipient_logits(host,adapter,*inputs)
                target=torch.tensor([label_ids[t["gold"]][0] for t in batch],device=host.device)
                loss=training_loss(logits,target,[label_ids[label][0] for label in ("A", "B")],args.objective)
                (loss/4).backward()
                losses.append(loss.item())
            if any(p.grad is not None for p in host.model.parameters()):
                raise ValueError("Backbone received a gradient")
            norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True).item()
            if norm<=0:
                raise ValueError("Recipient received no learning signal")
            optimizer.step()
            store.append("training.jsonl",{"candidate":candidate,"step":step,"losses":losses,
                "gradient_norm":norm,**entry,"tasks_sha256":digest([{k:v for k,v in t.items() if k!="source"} for t in tasks])})
        after=host.meter.snapshot()
        training_forwards = after["forwards"] - before["forwards"]
        training_tokens = after["processed_tokens"] - before["processed_tokens"]
        if baseline is not None:
            previous = baseline["raw_fit"]
            if (training_forwards != previous["training_forwards"]
                    or training_tokens != previous["training_processed_tokens"]):
                adapter.close()
                raise ValueError("Fresh binary training forward/token counts differ from the baseline")
        artifact=store.path/("recipient_"+candidate+".npz")
        saved=save_recipient(artifact,adapter,host.identity,dataset_sha256=file_sha(DATA/"manifest.json"),
            source_sha256=source["source_sha256"],training={**config,"candidate":candidate,
            "schedule_sha256":digest(schedule),"preprocessing_sha256":file_sha(store.path/"preprocessing.npz")})
        adapter.close()
        adapter,metadata,sha=load_recipient(artifact,host)
        if sha!=saved: raise ValueError("Recipient artifact failed serialization")
        store.append("fits.jsonl",{"candidate":candidate,"objective":args.objective,"initial_parameters_sha256":initial,"artifact":artifact.name,
            "sha256":sha,"trainable_parameters":sum(p.numel() for p in adapter.parameters()),
            "training_forwards":training_forwards,"training_processed_tokens":training_tokens,
            "schedule_sha256":digest(schedule)})
        with torch.no_grad():
            records=evaluate(host,adapter,candidate,cases,groups,raw,transform,store)
        summaries[candidate]=scorer.score_candidate(records,candidate)
        store.write("summary_"+candidate+".json",summaries[candidate])
        adapter.close()
        if summaries[candidate]["recipient_screen"]["pass"]:
            selected=candidate
            break
    if parameter_sha(host.model)!=backbone: raise ValueError("Backbone changed")
    store.write("environment.after.json",inventory(ROOT))
    return {"status":"COMPLETED_RECIPIENT_DEVELOPMENT","selected":selected,"screen_pass":selected is not None,
        "backbone_sha256_before":backbone,"backbone_sha256_after":parameter_sha(host.model),
        "optimizer_updates":256*len(summaries),"candidates":list(summaries),"unique_donor_forwards":len(text_cache),
        "logical_donor_rows":len(cases),"qualification_content_loaded":False,"G_trained":False,
        "objective":args.objective,"baseline_donor_hashes_replayed":len(cases) if baseline is not None else None,
        "compute":host.meter.snapshot()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--objective", choices=("native_full", "native_binary"), default="native_full")
    p.add_argument("--baseline-fit", type=Path)
    p.add_argument("--audit", type=Path)
    p.set_defaults(backend="hf",cpu_threads=1,device="cuda:0",dtype="bfloat16",slots=8,cut=None,writer=None,
        max_context=1024,max_new_tokens=None,until_eos=True,transport=None,encoder_bridge=None,
        reviewed_local_code=None,wall_seconds=3600,profile=ROOT/"configs/models/gemma2b.yaml")
    args=p.parse_args()
    if args.objective == "native_binary" and args.baseline_fit is None:
        p.error("--objective native_binary requires --baseline-fit")
    if args.objective == "native_binary" and args.audit is None:
        p.error("--objective native_binary requires --audit")
    if args.objective == "native_full" and args.baseline_fit is not None:
        p.error("--baseline-fit is used only by native_binary")
    if args.objective == "native_full" and args.audit is not None:
        p.error("--audit is used only by native_binary")
    store,start=RunStore(args.out),time.monotonic()
    try: result=main_run(args,store)
    except Exception as exc:
        store.write("terminal.json",{"status":"INVALID_STOPPED","error":str(exc),"type":type(exc).__name__,
            "elapsed_seconds":time.monotonic()-start})
        raise
    store.write("terminal.json",{**result,"elapsed_seconds":time.monotonic()-start})


if __name__=="__main__":
    main()
