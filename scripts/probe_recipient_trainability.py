"""A fixed paired literal-versus-slot small-set trainability control."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import torch
from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_continuity_recipient import DATA, STUDY, batch_inputs, tasks_for_group, model_scores, training_loss
from selffold.carrier_probe import capture_description
from selffold.cli import ROOT, make_host
from selffold.continuity_data import examples, load_groups
from selffold.recipient_learning import RecipientAdapter, batched_recipient_logits, save_recipient, load_recipient
from selffold.records import RunStore, State, code_manifest, digest, file_sha
from selffold.resources import inventory

TRAIN_GROUPS = ("both_if-01", "either_if-01", "unless-01", "exclusive-01")
CONFIG = {"rank":16, "last_layers":4, "scale":1., "seed":47, "updates":64,
          "effective_batch":32, "microbatch":8, "learning_rate":3e-4,
          "weight_decay":0., "gradient_clip":1., "objective":"native_binary",
          "conditions":["literal","latent"], "stage":"stage2", "frame":"self", "wording":0}


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def metrics(records):
    groups, compound = defaultdict(list), defaultdict(list)
    forced_correct = ties = coverage = 0
    for x in records:
        native = x["greedy_label"] == x["gold"]
        groups[x["group"]].append(native)
        compound[x["group"],x["world"],x["label_order"]].append(native)
        forced = "A" if x["A"] > x["B"] else "B" if x["B"] > x["A"] else None
        forced_correct += forced == x["gold"]
        ties += forced is None
        coverage += x["greedy_label"] in ("A","B")
    if not groups or any(len(x) != 32 for x in groups.values()) or any(len(x) != 4 for x in compound.values()):
        raise ValueError("Incomplete small-set policy/order panel")
    native = sum(sum(x)/len(x) for x in groups.values())/len(groups)
    joint = sum(all(x) for x in compound.values())/len(compound)
    return {"n":len(records), "native_correct":sum(sum(x) for x in groups.values()),
        "native_accuracy":native, "forced_correct":forced_correct, "forced_ties":ties,
        "native_label_coverage":coverage, "compound_n":len(compound),
        "native_compound_correct":sum(all(x) for x in compound.values()),
        "native_compound_accuracy":joint}


def evaluate(host, adapter, condition, phase, tasks, state_map, store):
    output = []
    route = "literal_base" if condition == "literal" else "matched"
    with torch.no_grad():
        for start in range(0,len(tasks),8):
            batch = tasks[start:start+8]
            scores = model_scores(host,adapter,batch,state_map,route)
            for task,score in zip(batch,scores,strict=True):
                record = {k:v for k,v in task.items() if k not in {"source","question","key"}}
                state = state_map[task["key"]]
                record.update(condition=condition,phase=phase,question_sha256=digest(task["question"]),
                    source_sha256=digest(task["source"]),
                    state_sha256=State(slots=state).sha() if state is not None else "literal:"+digest(task["source"]),**score)
                store.append("decisions.jsonl",record)
                output.append(record)
    return output


def run(args, store):
    seal = code_manifest(ROOT)
    store.write("source.json",seal)
    if seal["git_status"] != "" or not seal["git_commit"]:
        raise ValueError("Clean trainability source required")
    baseline = args.baseline_fit
    prior_terminal = json.loads((baseline/"terminal.json").read_text())
    prior_request = json.loads((baseline/"request.json").read_text())
    if prior_terminal["status"] != "COMPLETED_RECIPIENT_DEVELOPMENT" or prior_request["data_sha256"] != file_sha(DATA/"manifest.json"):
        raise ValueError("Unexpected baseline fit or data")
    all_train = {g["id"]:g for g in load_groups(DATA,"train")}
    groups = {"train":{key:all_train[key] for key in TRAIN_GROUPS},
              "dev":{g["id"]:g for g in load_groups(DATA,"dev")[:2]}}
    store.write("request.json",{**CONFIG,"train_groups":list(groups["train"]),"dev_groups":list(groups["dev"]),
        "plan_sha256":file_sha(STUDY/"TRAINABILITY_PLAN.md"),"data_sha256":file_sha(DATA/"manifest.json"),
        "baseline_source_commit":json.loads((baseline/"source.json").read_text())["git_commit"],
        "baseline_source_sha256":file_sha(baseline/"source.json"),"baseline_terminal_sha256":file_sha(baseline/"terminal.json"),
        "baseline_donors_sha256":file_sha(baseline/"donors.jsonl"),"baseline_host_sha256":file_sha(baseline/"host.json")})
    store.write("environment.before.json",inventory(ROOT))
    cases = {}
    for split in groups:
        for row in examples(DATA,split,"stage2"):
            if row["group"] in groups[split] and "-self-" in row["id"]:
                cases["stage2/"+row["id"]] = {**row,"split":split}
    if len(cases) != 24:
        raise ValueError("Unexpected trainability sources")
    host,checkpoint = make_host(args)
    if checkpoint["checkpoint_sha256"] != CHECKPOINT_SHA or host.identity != json.loads((baseline/"host.json").read_text()):
        raise ValueError("Trainability base host differs")
    store.write("host.json",host.identity)
    store.write("native_parity.json",parity := host.native_parity())
    labels = [host.tokenizer.encode(x,add_special_tokens=False) for x in ("A","B")]
    if not parity["exact"] or any(len(x) != 1 for x in labels):
        raise ValueError("Native parity or label tokenization failed")
    label_ids = [x[0] for x in labels]
    backbone = parameter_sha(host.model)
    previous = {x["key"]:x for x in rows(baseline/"donors.jsonl")}
    raw,cached = {},{}
    for key,row in cases.items():
        if row["source"] not in cached:
            state,_ = capture_description(host,row["source"],"workspace")
            cached[row["source"]] = state
        state = cached[row["source"]]
        if previous[key]["state_sha256"] != state.sha() or previous[key]["source_sha256"] != digest(row["source"]):
            raise ValueError("Trainability donor differs from original fit")
        raw[key] = state.slots
        store.append("donors.jsonl",{"key":key,"split":row["split"],"state_sha256":state.sha(),"source_sha256":digest(row["source"])})
    tasks = {split:[t for group in groups[split].values() for t in tasks_for_group(group,cases,"stage2",frame="self",wording=0)] for split in groups}
    for split, panel in tasks.items():
        for task in panel:
            task["split"] = split
    if (len(tasks["train"]),len(tasks["dev"])) != (128,64):
        raise ValueError("Unexpected trainability tasks")
    by_group = {key:[t for t in tasks["train"] if t["group"] == key] for key in TRAIN_GROUPS}
    generator = torch.Generator().manual_seed(47)
    indices = []
    while len(indices) < 64:
        indices.extend(torch.randperm(4,generator=generator).tolist())
    schedule = [TRAIN_GROUPS[i] for i in indices[:64]]
    store.write("schedule.json",schedule)
    summaries,initial_shas,task_orders = {},[],[]
    all_records = []
    for condition in CONFIG["conditions"]:
        adapter = RecipientAdapter(host,rank=16,last_layers=4,seed=47,scale=1.)
        try:
            initial_shas.append(parameter_sha(adapter))
            state_map = raw if condition == "latent" else {key:None for key in raw}
            initial = {split:evaluate(host,adapter,condition,"initial",value,state_map,store) for split,value in tasks.items()}
            route = "matched" if condition == "latent" else "literal_base"
            optimizer = torch.optim.AdamW(adapter.parameters(),lr=3e-4,weight_decay=0.)
            before = host.meter.snapshot()
            order = []
            for step,group in enumerate(schedule,1):
                batch_tasks = by_group[group]
                order.append(digest([{k:v for k,v in t.items() if k != "source"} for t in batch_tasks]))
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for start in range(0,32,8):
                    batch = batch_tasks[start:start+8]
                    logits = batched_recipient_logits(host,adapter,*batch_inputs(host,batch,state_map,route))
                    target = torch.tensor([label_ids[0 if t["gold"] == "A" else 1] for t in batch],device=host.device)
                    loss = training_loss(logits,target,label_ids,"native_binary")
                    (loss/4).backward()
                    losses.append(loss.item())
                if any(p.grad is not None for p in host.model.parameters()) or not any(p.grad is not None for p in adapter.parameters()):
                    raise ValueError("Trainability gradient isolation failed")
                norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True).item()
                optimizer.step()
                store.append("training.jsonl",{"condition":condition,"step":step,"group":group,
                    "losses":losses,"gradient_norm":norm,"tasks_sha256":order[-1]})
            after = host.meter.snapshot()
            task_orders.append(digest(order))
            first = by_group[schedule[0]][:8]
            with torch.no_grad():
                expected = batched_recipient_logits(host,adapter,*batch_inputs(host,first,state_map,route)).clone()
            artifact = store.path/("recipient_"+condition+".npz")
            sha = save_recipient(artifact,adapter,host.identity,dataset_sha256=file_sha(DATA/"manifest.json"),
                source_sha256=seal["source_sha256"],training={**CONFIG,"input_condition":condition,"schedule_sha256":digest(schedule)})
            adapter_sha = parameter_sha(adapter)
            adapter.close()
            adapter,_,loaded_sha = load_recipient(artifact,host)
            with torch.no_grad():
                actual = batched_recipient_logits(host,adapter,*batch_inputs(host,first,state_map,route))
            if sha != loaded_sha or adapter_sha != parameter_sha(adapter) or not torch.equal(expected,actual):
                raise ValueError("Trainability serialization parity failed")
            final = {split:evaluate(host,adapter,condition,"final",value,state_map,store) for split,value in tasks.items()}
            all_records += [x for panel in (initial,final) for value in panel.values() for x in value]
            summaries[condition] = {phase:{split:metrics(value) for split,value in panel.items()} for phase,panel in (("initial",initial),("final",final))}
            train_metrics = summaries[condition]["final"]["train"]
            summaries[condition]["engineering_trainability_positive"] = train_metrics["native_accuracy"] >= .95 and train_metrics["native_compound_accuracy"] >= .90
            store.append("fits.jsonl",{"condition":condition,"sha256":sha,"artifact":artifact.name,
                "initial_parameter_sha256":initial_shas[-1],"final_parameter_sha256":adapter_sha,"task_order_sha256":task_orders[-1],
                "training_forwards":after["forwards"]-before["forwards"],
                "training_processed_tokens":after["processed_tokens"]-before["processed_tokens"]})
        finally:
            adapter.close()
    if len(set(initial_shas)) != 1 or len(set(task_orders)) != 1 or len(all_records) != 768 or parameter_sha(host.model) != backbone:
        raise ValueError("Paired trainability contract differs")
    store.write("summary.json",summaries)
    store.write("environment.after.json",inventory(ROOT))
    return {"status":"COMPLETED_PAIRED_TRAINABILITY_CONTROL","optimizer_updates":128,"decisions":768,
        "train_tasks":128,"dev_tasks":64,"logical_donors":24,"unique_donor_captures":len(cached),
        "backbone_sha256_before":backbone,"backbone_sha256_after":parameter_sha(host.model),
        "engineering_trainability_positive":{key:value["engineering_trainability_positive"] for key,value in summaries.items()},
        "qualification_content_loaded":False,"G_trained":False,"compute":host.meter.snapshot()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--baseline-fit",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.set_defaults(backend="hf",cpu_threads=1,device="cuda:0",dtype="bfloat16",slots=8,cut=None,writer=None,
        max_context=1024,max_new_tokens=None,until_eos=True,transport=None,encoder_bridge=None,
        reviewed_local_code=None,wall_seconds=900,profile=ROOT/"configs/models/gemma2b.yaml")
    args = p.parse_args()
    store,start = RunStore(args.out),time.monotonic()
    try:
        result = run(args,store)
    except Exception as exc:
        store.write("terminal.json",{"status":"INVALID_STOPPED","error":str(exc),"type":type(exc).__name__,"elapsed_seconds":time.monotonic()-start})
        raise
    store.write("terminal.json",{**result,"elapsed_seconds":time.monotonic()-start})


if __name__ == "__main__":
    main()
