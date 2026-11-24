"""Prepared follow-up to PR #42. --check uses archived inputs and CPU only."""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import fit_paired_role_writer as prior
import numpy as np
import torch
from audit_paired_writer_rows import NEXT_ARMS, PAIR_METRICS, audit

from selffold.paired_row_objectives import ARMS, batch_objective
from selffold.records import RunStore, digest, file_sha
from selffold.structured_report import (
    Midpoint,
    final_choices,
    read_fields,
    reencode,
    report_text,
)
from selffold.structured_report_writer import (
    attach_writer,
    detach_writer,
    free_reports,
    pad_batch,
    training_record,
)

STUDY = prior.ROOT / "provenance/paired_row_objectives_20261124"
ARCHIVE = prior.STUDY / "produce"
SEEDS = (257, 258, 259)
CLOSED_SHA = "cef4f90b17eb4799742e1d69ef9d256d9e296ce1a2d7ed8d601a13d8048cf747"
PARENT = prior.PARENT / "writer.npz"


def schedule(twins, cfg, seed):
    rng = np.random.default_rng(seed)
    return [
        [i for pair in group for i in twins[pair]]
        for _ in range(cfg["epochs"])
        for group in rng.permutation(len(twins)).reshape(-1, cfg["pairs_per_batch"])
    ]


def checked_inputs():
    if file_sha(ARCHIVE / "closed.json") != CLOSED_SHA:
        raise ValueError("PR #42 archive manifest changed")
    files = prior.read(ARCHIVE / "closed.json")["files"]
    for name in (
        "binding.json",
        "host.json",
        "individual_training.jsonl",
        *(k for k in files if k.startswith("native_")),
    ):
        if file_sha(ARCHIVE / name) != files[name]:
            raise ValueError(f"Frozen input differs: {name}")
    cfg, actor, _, names, train, test = prior.prepare()
    binding = prior.read(ARCHIVE / "binding.json")
    if (file_sha(prior.CFG), digest(train), digest(test), file_sha(PARENT)) != (
        binding["design"],
        binding["train"],
        binding["test"],
        binding["parent"],
    ):
        raise ValueError("Panel, parent or original design changed")
    twins = prior.twin_indices(train)
    schedules = {s: schedule(twins, cfg, s) for s in SEEDS}
    if any(len(steps) != 256 for steps in schedules.values()):
        raise ValueError("Incorrect training budget")
    if schedules[257] != [
        r["indices"] for r in prior.read_lines(ARCHIVE / "individual_training.jsonl")
    ]:
        raise ValueError("Control minibatch order differs from PR #42")
    return cfg, actor, names, train, test, schedules


def fit_arm(host, adapters, samples, steps, cfg, arm, store):
    prior.restore(adapters, PARENT)
    params = list(prior.weights(adapters).values())
    if sum(p.numel() for p in params) != 245_760 or {id(p) for p in params} != {
        id(p) for p in host.model.parameters() if p.requires_grad
    }:
        raise ValueError("Unexpected trainable parameters")
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    for step, indices in enumerate(steps, 1):
        batch = [samples[i] for i in indices]
        ids, mask = pad_batch(
            [s[0] for s in batch], host.tokenizer.pad_token_id, host.device, left=False
        )
        logits = host.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        loss, parts = batch_objective(logits, ids, batch, arm, cfg["margin"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        opt.step()
        store.append(
            arm + "_training.jsonl",
            dict(
                step=step,
                indices=indices,
                loss=loss.item(),
                **{k: v.item() for k, v in parts.items()},
            ),
        )
    path = store.path / (arm + ".npz")
    np.savez_compressed(
        path, **{k: v.detach().cpu().numpy() for k, v in prior.weights(adapters).items()}
    )
    return file_sha(path)


def generate(host, adapters, ordered, store, endpoints):
    records = {}
    for arm in ("parent", *ARMS):
        path = PARENT if arm == "parent" else store.path / (arm + ".npz")
        if arm in endpoints and file_sha(path) != endpoints[arm]:
            raise ValueError(f"Endpoint changed: {arm}")
        prior.restore(adapters, path)
        for start in range(0, len(ordered), 8):
            batch = ordered[start : start + 8]
            outputs = free_reports(host, [q["prompt"] for q in batch], 160)
            for q, r in zip(batch, outputs, strict=True):
                record = dict(arm=arm, id=q["id"], prompt_sha256=digest(q["prompt"]), **r)
                store.append("reports.jsonl", record)
                records[arm, q["id"]] = record
    return records


def evaluate(cfg, actor, names, test, records, banks):
    if set(records) != {(arm, q["id"]) for arm in ("parent", *ARMS) for q in test}:
        raise ValueError("Incomplete writer coverage")
    outcomes = []
    for q in test:
        split = "fresh" if q["menu"] in cfg["fresh"] else "previous"
        direct = Midpoint(tuple(q["rows"]), q["bit"])
        gold = list(direct.fields())
        for arm in NEXT_ARMS:
            if arm == "L":
                values, state = gold, direct
            else:
                if arm == "oracle":
                    text = report_text([names[i] for i in gold])
                else:
                    record = records[arm, q["id"]]
                    if record["prompt_sha256"] != digest(q["prompt"]):
                        raise ValueError("Report bound to a different prompt")
                    text = record["text"]
                values = read_fields(text, names)
                state = reencode(values)
            native = final_choices(
                actor, state, q["menu"], q["default"], cfg["next_masks"], banks[split]
            )
            outcomes.append(
                dict(
                    id=q["id"],
                    split=split,
                    arm=arm,
                    menu=q["menu"],
                    pattern=q["pattern"],
                    rotation=q["rotation"],
                    default=q["default"],
                    fields=values,
                    rows_correct=[v == g for v, g in zip(values[:4], gold[:4])],
                    role=values[4] == gold[4],
                    **native,
                )
            )
    return outcomes


def contrast(panel, new, old):
    a, b = panel[new], panel[old]
    key = lambda p: (p["menu"], p["pattern"], p["rotation"])
    a_pairs, b_pairs = ({key(p): p for p in r["presentations"]} for r in (a, b))
    if a_pairs.keys() != b_pairs.keys():
        raise ValueError("Unmatched comparison")
    changes = {
        k: {
            "gained": sum(a_pairs[i][k] and not b_pairs[i][k] for i in a_pairs),
            "lost": sum(b_pairs[i][k] and not a_pairs[i][k] for i in a_pairs),
        }
        for k in PAIR_METRICS
    }
    menu_gains = sum(a["by_menu"][m]["joint"] > b["by_menu"][m]["joint"] for m in a["by_menu"])
    gates = {
        "anchors": min(panel[k]["joint"] for k in ("L", "oracle")) >= 92,
        "retention": a["four_row_reports"] >= max(164, panel["parent"]["four_row_reports"] - 9),
        "joint": a["joint"] >= b["joint"] + 5 and menu_gains >= 3,
        "role": a["role"] >= b["role"],
        "final": a["final"] >= b["final"],
        "shared_wrong": a["any_shared_wrong"] <= b["any_shared_wrong"],
        "all_orders": a["rotation_histogram"]["joint"][4] >= b["rotation_histogram"]["joint"][4],
    }
    return {
        "gates": gates,
        "passed": all(gates.values()),
        "changes": changes,
        "menu_gains": menu_gains,
    }


def run(folder, checkpoint, seal):
    folder = folder.resolve()
    if folder.is_relative_to(prior.ROOT):
        raise ValueError("Choose a new output directory outside the source checkout")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(prior.ROOT), *args], text=True).strip()

    if git("rev-parse", "HEAD") != seal or git("status", "--porcelain"):
        raise ValueError("Run from a clean checkout at the supplied full source seal")
    cfg, actor, names, train, test, schedules = checked_inputs()
    store = RunStore(folder)
    store.write(
        "binding.json",
        {
            "source": seal,
            "plan": file_sha(STUDY / "PLAN.md"),
            "archive": CLOSED_SHA,
            "parent": file_sha(PARENT),
            "train": digest(train),
            "test": digest(test),
            "coefficients": ARMS,
            "schedules": {s: digest(v) for s, v in schedules.items()},
        },
    )
    host, parity = prior.native_host(cfg, checkpoint)
    base = prior.parameter_sha(host.model)
    old_host = prior.read(ARCHIVE / "host.json")
    if host.identity != old_host["identity"] or base != old_host["base_sha256"]:
        raise ValueError("PR #42 host or backbone mismatch")
    store.write("host.json", {"identity": host.identity, "parity": parity, "base_sha256": base})
    samples = [
        training_record(
            host,
            q["prompt"],
            [names[i] for i in Midpoint(tuple(q["rows"]), q["bit"]).fields()],
            prior.options(q, names),
        )
        for q in train
    ]
    adapters = attach_writer(host, cfg)
    stores, endpoints = {}, {}
    for seed, steps in schedules.items():
        stores[seed] = RunStore(folder / str(seed))
        endpoints[seed] = {
            a: fit_arm(host, adapters, samples, steps, cfg, a, stores[seed]) for a in ARMS
        }
    store.write("fits_closed.json", endpoints)
    ordered = sorted(test, key=lambda q: digest(["paired-role-v1", q["id"]]))
    banks = prior.banks_from(ARCHIVE, cfg)
    decisions = {}
    for seed, dest in stores.items():
        records = generate(host, adapters, ordered, dest, endpoints[seed])
        outcomes = evaluate(cfg, actor, names, test, records, banks)
        dest.write("outcomes.json", outcomes)
        panels = audit(outcomes, NEXT_ARMS, details=True)
        decisions[seed] = {
            new + "_vs_" + old: contrast(panels["fresh"], new, old)
            for new, old in (
                ("combined", "mean"),
                ("row_ce", "mean"),
                ("agreement", "mean"),
                ("combined", "row_ce"),
                ("combined", "agreement"),
            )
        }
        dest.write("audit.json", panels)
        dest.write("stops.json", dict(Counter(r["stop"] for r in records.values())))
    detach_writer(host, adapters)
    if base != prior.parameter_sha(host.model):
        raise ValueError("Frozen Gemma backbone changed")
    store.write(
        "decisions.json",
        {
            "by_seed": decisions,
            "combined_passes_all_seeds": all(
                d["combined_vs_mean"]["passed"] for d in decisions.values()
            ),
        },
    )
    store.write(
        "closed.json",
        {str(p.relative_to(folder)): file_sha(p) for p in sorted(folder.rglob("*")) if p.is_file()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--run", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seal")
    args = parser.parse_args()
    if args.check:
        cfg, _, _, train, test, schedules = checked_inputs()
        print(
            json.dumps(
                {
                    "train": len(train),
                    "test": len(test),
                    "arms": list(ARMS),
                    "schedules": {
                        s: {"updates": len(v), "sha256": digest(v)} for s, v in schedules.items()
                    },
                }
            )
        )
    else:
        if not args.checkpoint or not args.seal:
            parser.error("--run requires --checkpoint and --seal")
        run(args.run, args.checkpoint, args.seal)
