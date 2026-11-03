"""Calibrate the existing native adapter on raw teachers and evaluate all three stages."""
import gzip
import json
from collections import defaultdict
from pathlib import Path

import torch

from conditional_curriculum_learning import weighted_native_loss
from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_conditional_curriculum import main
from fit_continuity_recipient import batch_inputs, model_scores
from fit_stage_grammar import ACTIONS, BASE, DESIGN, PREVIOUS, STUDY, read
from fit_stage_transitions import bank, records
from selffold.cli import ROOT, make_host
from selffold.conditional_curriculum import STAGES, tasks_for
from selffold.recipient_learning import batched_recipient_logits, load_recipient, save_recipient
from selffold.records import code_manifest, digest, file_sha
from selffold.stage_grammar import fixture

CONFIG = ROOT / "data/preprint_v1_20261104.json"
FOCUS = STUDY / "literal/focus.npz"
FOCUS_SHA = "bf4d476cf50825a92ce41d2e8430336b7a0998d31143abe03ba3a38b11e168d3"


def questions(c, groups, f):
    return [t for w in range(4) for t in tasks_for(c, groups[c["topic"]], f, w)]


def schedule(cases):
    rng = torch.Generator().manual_seed(1481)
    blocks = []
    for stage in STAGES:
        keys = sorted(k for k in cases if k[3] == 0 and k[4] == stage)
        blocks.append([keys[i] for i in torch.randperm(len(keys), generator=rng)[:64]])
    return [k for step in zip(*blocks, strict=True) for k in step]


def summarize(rows):
    by = {(r["arm"], r["key"], r["world"], r["order"], r["route"]): r for r in rows}
    if len(by) != len(rows):
        raise ValueError("Duplicate evaluation")
    summary = {}
    for arm in ("parent", "trained"):
        stages = {}
        for stage in STAGES:
            units = defaultdict(list)
            sums = dict(matched=0, opposite_reversal=0, zero=0, literal=0)
            for r in rows:
                if (r["arm"], r["route"], r["stage"]) != (arm, "matched", stage):
                    continue
                key = arm, r["key"], r["world"], r["order"]
                opp, zero, lit = [by[(*key, k)] for k in ("opposed", "zero", "literal")]
                ok = r["greedy_label"] == r["gold"]
                reverse = opp["greedy_label"] == ("B" if r["gold"] == "A" else "A")
                sign = (r["A"] - r["B"]) * (opp["A"] - opp["B"]) < 0
                sums["matched"] += ok
                sums["opposite_reversal"] += ok and reverse and sign
                sums["zero"] += zero["greedy_label"] == r["gold"]
                sums["literal"] += lit["greedy_label"] == r["gold"]
                units[r["topic"], r["pair"]].append(ok)
            n = sum(map(len, units.values()))
            if len(units) != 16 or n != (256 if stage == "rule" else 512):
                raise ValueError("Incomplete stage panel")
            strict = sum(all(v) for v in units.values())
            passed = (10*sums["matched"] >= 9*n and 10*sums["opposite_reversal"] >= 9*n
                      and strict >= 12 and 4*(sums["matched"] - sums["zero"]) >= n)
            stages[stage] = dict(**sums, n=n, strict=[strict, 16], passed=passed)
        summary[arm] = dict(stages=stages, passed=all(s["passed"] for s in stages.values()))
    return summary


def run(args, store):
    seal = code_manifest(ROOT)
    if seal["git_status"] or file_sha(FOCUS) != FOCUS_SHA:
        raise ValueError("Need a clean source and the original focus reader")
    store.write("source.json", seal)
    host, ident = make_host(args)
    if ident["checkpoint_sha256"] != CHECKPOINT_SHA or not host.native_parity()["exact"]:
        raise ValueError("Pinned native host required")
    backbone = parameter_sha(host.model)
    store.write("host.json", host.identity)
    base, old, design, actions, cfg = map(read, (BASE, PREVIOUS, DESIGN, ACTIONS, CONFIG))
    groups = actions["groups"]
    f = fixture(base, old, design, "train")
    cases = records(groups, f, "fit")
    order = schedule(cases)
    store.write("schedule.json", order)
    keys = {(*k[:3], pole, *k[4:]) for k in order for pole in (0, 1)}
    values = bank(host, {k:c for k,c in cases.items() if k in keys}, store, "fit")
    reader, _, _ = load_recipient(FOCUS, host, trainable=True)
    labels = [host.tokenizer.encode(x, add_special_tokens=False)[0] for x in ("A","B")]
    opt = torch.optim.AdamW(reader.parameters(), lr=.0002, weight_decay=0)
    output = store.path / "reader.npz"
    try:
        for step, k in enumerate(order):
            opt.zero_grad(set_to_none=True)
            loss_total = 0.0
            for pole in (0, 1):
                pk = (*k[:3], pole, *k[4:])
                c = cases[pk]
                jobs = questions(c, groups, f)
                for j in (0,4):
                    ts = jobs[j:j+4]
                    inputs = batch_inputs(host, ts, {c["key"]: values[pk].to(host.device)}, "matched")
                    scores = batched_recipient_logits(host, reader, *inputs)
                    target = torch.tensor([labels[int(t["gold"]=="B")] for t in ts], device=host.device)
                    loss = 2 * weighted_native_loss(scores, target, [t["world"] for t in ts],
                                                    margin=2., label_ids=labels)
                    loss.backward()
                    loss_total += loss.item()
            norm = torch.nn.utils.clip_grad_norm_(reader.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            store.append("training.jsonl", dict(step=step+1, loss=loss_total, grad_norm=float(norm)))
        fitted = save_recipient(output, reader, host.identity, dataset_sha256=file_sha(CONFIG),
                                source_sha256=seal["source_sha256"], training=dict(parent=FOCUS_SHA, steps=192))
    finally:
        reader.close()
    store.write("fit_closed.json", dict(weights=fitted, updates=len(order), labels=16*len(order),
                                        fresh_captures=0, parent=FOCUS_SHA, fit_states=len(values)))
    del values
    fresh = fixture(base, old, design, "names")
    fresh["styles"]["dispatch"] = cfg["raw_style"]
    fresh["fresh"].update(flag_pairs=cfg["raw_flags"], styles=["dispatch"])
    cases = records(groups, fresh, "fresh")
    values = bank(host, cases, store, "test")
    rows = []
    with torch.no_grad(), gzip.open(store.path / "decisions.jsonl.gz", "wt") as stream:
        for arm, path in (("parent", FOCUS), ("trained", output)):
            reader, _, sha = load_recipient(path, host)
            if sha != (FOCUS_SHA if arm == "parent" else fitted):
                raise ValueError("Changed endpoint")
            try:
                for k, c in cases.items():
                    ts = questions(c, groups, fresh)
                    z = values[k].to(host.device)
                    opposite = (*k[:3], 1-k[3], *k[4:])
                    for route, supplied in (("matched",z),("opposed",values[opposite].to(host.device)),
                                            ("zero",torch.zeros_like(z)),("literal",None)):
                        kind = "literal_base" if route == "literal" else "matched"
                        answers = model_scores(host, reader, ts, {c["key"]:supplied}, kind)
                        for t, ans in zip(ts, answers, strict=True):
                            row = dict(arm=arm,route=route,key=c["key"],stage=k[4],topic=k[0],
                                       pair=k[1],world=t["world"],order=t["label_order"],
                                       gold=t["gold"],question=digest(t["question"]),**ans)
                            rows.append(row)
                            stream.write(json.dumps(row,separators=(",",":"))+"\n")
            finally:
                reader.close()
    result=summarize(rows)
    store.write("summary.json", result)
    if parameter_sha(host.model) != backbone:
        raise ValueError("Frozen backbone changed")
    return dict(status="RAW_TEACHER_SCREEN_COMPLETE", passed=result["trained"]["passed"],
                backbone=backbone, rows=len(rows), compute=host.meter.snapshot(), qualification_opened=False)


if __name__ == "__main__":
    main(run)
