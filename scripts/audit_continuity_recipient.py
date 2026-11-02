"""A zero-update numerical audit of the completed native recipient endpoints."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_continuity_recipient import DATA, STUDY, STAGES, batch_inputs, tasks_for_group
from selffold.carrier_probe import capture_description
from selffold.cli import ROOT, make_host
from selffold.continuity_data import examples, load_groups
from selffold.recipient_learning import RecipientAdapter, batched_recipient_logits, load_recipient
from selffold.records import RunStore, State, code_manifest, digest, file_sha

PATHS = ("raw_bf16", "native_capped", "cap_fp32_from_raw", "projection_fp32", "cap_fp32_from_projection")


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def summarize(rows, pairs):
    output = {}
    for candidate in ("initial", "raw", "standardized"):
        chosen = [x for x in rows if x["candidate"] == candidate]
        if len(chosen) != 384:
            raise ValueError("Incomplete audit candidate")
        groups = {}
        for split in ("all", "train", "dev"):
            subset = [x for x in chosen if split == "all" or x["split"] == split]
            metrics = {}
            for name in PATHS:
                ties = sum(x[name]["A"] == x[name]["B"] for x in subset)
                correct = sum(x[name][x["gold"]] > x[name]["B" if x["gold"] == "A" else "A"] for x in subset)
                metrics[name] = {"n": len(subset), "correct": correct, "ties": ties, "accuracy": correct/len(subset)}
            metrics["native_greedy"] = {"n": len(subset), "correct": sum(x["greedy_label"] == x["gold"] for x in subset),
                "label_coverage": sum(x["greedy_label"] in ("A", "B") for x in subset)}
            groups[split] = metrics
        cp = [x for x in pairs if x["candidate"] == candidate]
        output[candidate] = {"scores": groups, "opposed_pairs": len(cp),
            "identical_hidden_pairs": sum(x["hidden_exact_equal"] for x in cp),
            "native_ties_without_raw_ties": sum(x["native_capped"]["A"] == x["native_capped"]["B"] and x["raw_bf16"]["A"] != x["raw_bf16"]["B"] for x in chosen),
            "native_derivative_zero_labels": sum(v == 0 for x in chosen for v in x["stored_tanh_derivative"].values())}
    return output


def run(args, store):
    seal = code_manifest(ROOT)
    store.write("source.json", seal)
    if seal["git_status"] != "" or not seal["git_commit"]:
        raise ValueError("Clean audit source required")
    fit = args.fit
    prior_source = json.loads((fit/"source.json").read_text())
    terminal = json.loads((fit/"terminal.json").read_text())
    if not prior_source["git_commit"].startswith("77c4ab6") or terminal["status"] != "COMPLETED_RECIPIENT_DEVELOPMENT":
        raise ValueError("Unexpected or incomplete recipient fit")
    request = json.loads((fit/"request.json").read_text())
    if request["data_sha256"] != file_sha(DATA/"manifest.json"):
        raise ValueError("Fit data binding differs")
    preprocessing = fit/"preprocessing.npz"
    store.write("request.json", {"plan_sha256":file_sha(STUDY/"AUDIT_PLAN.md"), "data_sha256":file_sha(DATA/"manifest.json"),
        "fit_source":prior_source["git_commit"], "fit_source_sha256":file_sha(fit/"source.json"),
        "fit_terminal_sha256":file_sha(fit/"terminal.json"), "preprocessing_sha256":file_sha(preprocessing),
        "optimizer_updates":0, "counterfactual_scores":"two_column_numerical_diagnostics"})
    groups = {s:{g["id"]:g for g in load_groups(DATA,s)[:2]} for s in ("train","dev")}
    cases = {}
    for split in groups:
        for stage in STAGES:
            for row in examples(DATA,split,stage):
                if row["group"] in groups[split] and "-self-" in row["id"]:
                    cases[stage+"/"+row["id"]] = {**row,"split":split}
    if len(cases) != 48:
        raise ValueError("Unexpected audit source count")
    host, checkpoint = make_host(args)
    if checkpoint["checkpoint_sha256"] != CHECKPOINT_SHA or host.identity != json.loads((fit/"host.json").read_text()):
        raise ValueError("Audit base host differs")
    store.write("host.json", host.identity)
    store.write("native_parity.json", parity := host.native_parity())
    if not parity["exact"]:
        raise ValueError("Audit native parity failed")
    cap = host.model.config.final_logit_softcapping
    if cap != 30.:
        raise ValueError("Unexpected final softcap")
    labels = [host.tokenizer.encode(x,add_special_tokens=False) for x in ("A","B")]
    if any(len(x) != 1 for x in labels):
        raise ValueError("Native labels must be single tokens")
    label_ids = torch.tensor([x[0] for x in labels],device=host.device)
    weight = host.model.lm_head.weight.index_select(0,label_ids).detach().float()
    backbone = parameter_sha(host.model)
    previous = {x["key"]:x for x in lines(fit/"donors.jsonl")}
    raw, cached = {}, {}
    for key,row in cases.items():
        if row["source"] not in cached:
            state,_ = capture_description(host,row["source"],"workspace")
            cached[row["source"]] = state
        state = cached[row["source"]]
        if previous[key]["state_sha256"] != state.sha() or previous[key]["source_sha256"] != digest(row["source"]):
            raise ValueError("Audit source or donor differs from fit")
        raw[key] = state.slots
        store.append("donors.jsonl",{"key":key,"split":row["split"],"source_sha256":digest(row["source"]),"state_sha256":state.sha()})
    with np.load(preprocessing,allow_pickle=False) as z:
        if set(z.files) != {"mean","std","row_std"}:
            raise ValueError("Unexpected preprocessing arrays")
        stats = {k:torch.from_numpy(z[k].copy()).to(host.device) for k in z.files}
    if (stats["mean"].shape != (host.slots,host.width) or stats["std"].shape != (host.slots,host.width)
            or stats["row_std"].shape != (host.slots,1) or torch.any(stats["std"] < 0)
            or any(not torch.isfinite(v).all() for v in stats.values()) or torch.any(stats["row_std"] <= 0)):
        raise ValueError("Invalid train preprocessing statistics")
    def standardized(value):
        return (stats["mean"] + stats["row_std"]*(value.float()-stats["mean"])/stats["std"].clamp_min(stats["row_std"]*.01)).to(host.dtype)
    tasks = [t for stage in STAGES for split in groups for group in groups[split].values()
             for t in tasks_for_group(group,cases,stage,frame="self",wording=0)]
    if len(tasks) != 384:
        raise ValueError("Unexpected audit task count")
    fits = {x["candidate"]:x for x in lines(fit/"fits.jsonl")}
    records,pairs = [],[]
    for candidate in ("initial","raw","standardized"):
        artifact = None
        if candidate == "initial":
            adapter = RecipientAdapter(host,rank=16,last_layers=4,seed=47,scale=1.)
            if parameter_sha(adapter) != fits["raw"]["initial_parameters_sha256"]:
                adapter.close()
                raise ValueError("Initial recipient differs")
        else:
            artifact = fit/("recipient_"+candidate+".npz")
            adapter,metadata,sha = load_recipient(artifact,host)
            if sha != fits[candidate]["sha256"] or metadata["training"]["preprocessing_sha256"] != file_sha(preprocessing):
                adapter.close()
                raise ValueError("Recipient artifact binding differs")
        try:
            adapter_before = parameter_sha(adapter)
            states = {key:(standardized(value) if candidate == "standardized" else value) for key,value in raw.items()}
            hidden_rows = {}
            for start in range(0,len(tasks),8):
                batch = tasks[start:start+8]
                inputs = batch_inputs(host,batch,states,"matched")
                score_at = torch.tensor(inputs[-1],device=host.device)
                row_ids = torch.arange(len(batch),device=host.device)
                tapped = {}
                def tap(module,arguments,output):
                    tapped["hidden"] = arguments[0][row_ids,score_at,:].detach().clone()
                    tapped["raw"] = output[row_ids,score_at,:].index_select(-1,label_ids).detach().clone()
                hook = host.model.lm_head.register_forward_hook(tap)
                try:
                    with torch.no_grad():
                        logits = batched_recipient_logits(host,adapter,*inputs)
                finally:
                    hook.remove()
                if set(tapped) != {"hidden","raw"}:
                    raise ValueError("Missing head observation")
                original = tapped["raw"]
                native_tanh = torch.tanh(original/cap)
                float_tanh = torch.tanh(original.float()/cap)
                with torch.no_grad():
                    projected = F.linear(tapped["hidden"].float(),weight)
                values = {"raw_bf16":original.float(), "native_capped":logits.index_select(-1,label_ids),
                    "cap_fp32_from_raw":cap*float_tanh,"projection_fp32":projected,
                    "cap_fp32_from_projection":cap*torch.tanh(projected/cap)}
                if any(not torch.isfinite(v).all() for v in values.values()):
                    raise ValueError("Nonfinite audit head values")
                best = logits.argmax(-1).tolist()
                for i,t in enumerate(batch):
                    split = cases[t["key"]]["split"]
                    rec = {k:t[k] for k in ("case","stage","group","pole","frame","source_order","world","wording","label_order","gold")}
                    rec.update(candidate=candidate,split=split,question_sha256=digest(t["question"]),
                        state_sha256=State(slots=states[t["key"]]).sha(),raw_dtype=str(original.dtype),
                        hidden_dtype=str(tapped["hidden"].dtype),hidden_norm=tapped["hidden"][i].float().norm().item(),
                        greedy_token=best[i],greedy_label=("A" if best[i] == labels[0][0] else "B" if best[i] == labels[1][0] else None))
                    rec.update({name:dict(zip(("A","B"),value[i].tolist())) for name,value in values.items()})
                    rec["stored_tanh_derivative"] = dict(zip(("A","B"),(1-native_tanh[i].float().square()).tolist()))
                    rec["fp32_tanh_derivative"] = dict(zip(("A","B"),(1-float_tanh[i].square()).tolist()))
                    reference = []
                    for value in original[i].float().tolist():
                        e = math.exp(-2*abs(value/cap)); reference.append(4*e/(1+e)**2)
                    rec["float64_reference_derivative"] = dict(zip(("A","B"),reference))
                    key = (t["stage"],t["group"],t["source_order"],t["world"],t["label_order"],t["pole"])
                    if key in hidden_rows:
                        raise ValueError("Duplicate audit task")
                    hidden_rows[key] = (tapped["hidden"][i].float().cpu(),rec)
                    records.append(rec)
                    store.append("cases.jsonl",rec)
            for key,(hidden,rec) in hidden_rows.items():
                if key[-1] != "plus":
                    continue
                other,opposed = hidden_rows[(*key[:-1],"minus")]
                if rec["question_sha256"] != opposed["question_sha256"] or rec["gold"] == opposed["gold"]:
                    raise ValueError("Misaligned opposed audit pair")
                delta = hidden-other
                pair = {"candidate":candidate,"stage":key[0],"group":key[1],"source_order":key[2],"world":key[3],
                    "label_order":key[4],"split":rec["split"],"hidden_exact_equal":torch.equal(hidden,other),
                    "hidden_delta_l2":delta.norm().item(),"hidden_delta_max":delta.abs().max().item(),
                    "plus_state_sha256":rec["state_sha256"],"minus_state_sha256":opposed["state_sha256"],
                    "margin_shift":{p:(rec[p]["A"]-rec[p]["B"])-(opposed[p]["A"]-opposed[p]["B"]) for p in PATHS}}
                pairs.append(pair)
                store.append("opposed_hidden.jsonl",pair)
            if parameter_sha(adapter) != adapter_before:
                raise ValueError("Audit changed recipient parameters")
            if artifact is not None and file_sha(artifact) != fits[candidate]["sha256"]:
                raise ValueError("Audit artifact bytes changed")
            store.append("artifacts.jsonl",{"candidate":candidate,"parameter_sha256_before":adapter_before,
                "parameter_sha256_after":parameter_sha(adapter),"artifact_sha256":file_sha(artifact) if artifact else None})
        finally:
            adapter.close()
    if len(records) != 1152 or len(pairs) != 576 or parameter_sha(host.model) != backbone:
        raise ValueError("Incomplete audit or changed backbone")
    store.write("summary.json",summarize(records,pairs))
    return {"status":"COMPLETED_READ_ONLY_RECIPIENT_NUMERICAL_AUDIT","optimizer_updates":0,
        "cases":len(records),"opposed_pairs":len(pairs),"donor_rows":len(raw),"unique_donor_captures":len(cached),
        "backbone_sha256_before":backbone,"backbone_sha256_after":parameter_sha(host.model),
        "checkpoint_sha256":checkpoint["checkpoint_sha256"],"final_softcap":cap,
        "float32_matmul_precision":torch.get_float32_matmul_precision(),"compute":host.meter.snapshot(),
        "qualification_content_loaded":False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--fit",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.set_defaults(backend="hf",cpu_threads=1,device="cuda:0",dtype="bfloat16",slots=8,cut=None,writer=None,
        max_context=1024,max_new_tokens=None,until_eos=True,transport=None,encoder_bridge=None,
        reviewed_local_code=None,wall_seconds=900,profile=ROOT/"configs/models/gemma2b.yaml")
    args = p.parse_args()
    store,start = RunStore(args.out),time.monotonic()
    try:
        result = run(args,store)
    except Exception as exc:
        store.write("terminal.json",{"status":"INVALID_STOPPED","optimizer_updates":0,"error":str(exc),
            "type":type(exc).__name__,"elapsed_seconds":time.monotonic()-start})
        raise
    store.write("terminal.json",{**result,"elapsed_seconds":time.monotonic()-start})


if __name__ == "__main__":
    main()
