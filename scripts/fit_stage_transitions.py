"""Fit a case expansion and rule reintegration from eight-slot states, then test a frozen native reader."""

import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_conditional_curriculum import parser
from fit_continuity_recipient import model_scores
from fit_stage_grammar import ACTIONS, BASE, DESIGN, PREVIOUS, STUDY, read
from selffold.carrier_probe import capture_description
from selffold.cli import ROOT, make_host
from selffold.conditional_curriculum import make_cases, tasks_for
from selffold.records import RunStore, State, code_manifest, digest, file_sha
from selffold.recipient_learning import load_recipient
from selffold.stage_grammar import fixture
from selffold.stage_transitions import StageMap, fit_stage_map, target_preference


def records(groups, f, split):
    return {
        (
            c["topic"],
            c["flag_pair"],
            c["style"],
            c["pole"],
            c["stage"],
            c["source_order"],
        ): c
        for c in make_cases(groups, f, split)
    }


def bank(host, cases, store, split):
    result = {}
    for key, c in cases.items():
        state, meta = capture_description(
            host, c["source"], "workspace", forward_length=256
        )
        result[key] = state.slots.cpu().contiguous()
        store.append(
            "captures.jsonl",
            dict(
                split=split,
                key=c["key"],
                source=digest(c["source"]),
                state=meta["state_sha256"],
            ),
        )
    return result


def keys_for(cases, stage):
    return [k for k in cases if k[4] == stage]


def tensors(bank, keys, stage=None, order=None, opp=False):
    return torch.stack(
        [
            bank[t, f, s, p ^ int(opp), stage or st, o if order is None else order]
            for t, f, s, p, st, o in keys
        ]
    )


def fit_maps(values, keys):
    x = tensors(values, keys)
    y = tensors(values, keys, "cases", 0)
    z = tensors(values, keys, "rule", 0)
    g1 = fit_stage_map(x, y)
    predicted = g1.slots(x)
    g2 = fit_stage_map(torch.cat([predicted, y]), torch.cat([z, z]))
    rk = [k for k in values if k[4] == "rule"]
    zr = tensors(values, rk)
    target = torch.nn.functional.one_hot(
        torch.tensor([2 * k[0] + k[3] for k in rk]), 16
    ).float()
    dec = fit_stage_map(zr, target, rank=16)
    return {"g1": g1, "g2": g2, "decoder": dec}, zr.float().mean(0).to(zr.dtype)


def invariant_state_bank(path, maps):
    arrays = {
        f"{name}_{key}": v.numpy()
        for name, model in maps.items()
        for key, v in vars(model).items()
    }
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as f:
        return {
            name: StageMap(
                **{key: torch.from_numpy(f[f"{name}_{key}"]) for key in vars(model)}
            )
            for name, model in maps.items()
        }


@torch.no_grad()
def run(args, store):
    design, base, prior, action = map(read, (DESIGN, BASE, PREVIOUS, ACTIONS))
    groups = action["groups"]
    seal = code_manifest(ROOT)
    if seal["git_status"] or not seal["git_commit"]:
        raise ValueError("Clean execution source required")
    focus = args.literal / "focus.npz"
    fit_by = {x["arm"]: x for x in read(args.literal / "fit_closed.json")["candidates"]}
    if (
        read(args.literal / "source.json")["git_commit"]
        != "5fac1bd10591b6227ec1c6fb6cdfcef989a7cb30"
        or fit_by["focus"]["weights"] != file_sha(focus)
        or read(args.literal / "terminal.json")["status"]
        != "LITERAL_STAGE_RUN_COMPLETE"
    ):
        raise ValueError("Need the fixed step-400 focus reader")
    store.write("source.json", seal)
    store.write(
        "request.json",
        dict(
            reader=file_sha(focus),
            plan=file_sha(STUDY / "CHAIN_PLAN.md"),
            scorer=file_sha(STUDY / "score_chain.py"),
            rank=32,
            ridge=0.001,
            training_labels_for_decoder_only=True,
        ),
    )
    host, identity = make_host(args)
    if (
        identity["checkpoint_sha256"] != CHECKPOINT_SHA
        or not host.native_parity()["exact"]
    ):
        raise ValueError("Invalid Gemma host")
    store.write("host.json", host.identity)
    backbone = parameter_sha(host.model)
    train = records(groups, fixture(base, prior, design, "train"), "fit")
    values = bank(host, train, store, "fit")
    trainkeys = keys_for(train, "source")
    fitted, mean = fit_maps(values, trainkeys)
    models = invariant_state_bank(store.path / "maps.npz", fitted)
    x = tensors(values, trainkeys[:4])
    if not all(torch.equal(m(x), models[name](x)) for name, m in fitted.items()):
        raise ValueError("Saved functional maps differ")
    store.write(
        "maps_closed.json",
        dict(
            source_cases=len(trainkeys),
            g2_pairs=2 * len(trainkeys),
            rule_decoder_cases=len(keys_for(train, "rule")),
            rank={k: m.down.shape[-1] for k, m in models.items()},
            fresh_captures=0,
            artifact=file_sha(store.path / "maps.npz"),
        ),
    )
    del values, fitted
    f = fixture(base, prior, design, "fresh")
    cases = records(groups, f, "fresh")
    values = bank(host, cases, store, "fresh")
    keys = keys_for(cases, "source")
    raw = tensors(values, keys)
    opp = {k: i for i, k in enumerate(keys)}
    inverse = [opp[t, i, s, 1 - p, st, o] for t, i, s, p, st, o in keys]
    same = [opp[t, i, s, p, st, 1 - o] for t, i, s, p, st, o in keys]
    g1, g2 = models["g1"], models["g2"]
    middle = g1.slots(raw)
    tables = {
        "source": raw,
        "case_teacher": tensors(values, keys, "cases"),
        "rule_teacher": tensors(values, keys, "rule", 0),
        "g1": middle,
        "g2_only": g2.slots(raw),
        "both": g2.slots(middle),
        "opposed": g2.slots(middle[inverse]),
        "equivalent": g2.slots(middle[same]),
        "mean": mean.expand_as(raw),
        "zero": torch.zeros_like(raw),
    }
    canonical_cases = tensors(values, keys, "cases", 0)
    rule = tables["rule_teacher"]
    distances = {}
    for tag, z, target in (
        ("g1_case", middle, canonical_cases),
        ("z0_case", raw, canonical_cases),
        ("full_rule", tables["both"], rule),
        ("g1_rule", middle, rule),
        ("g2_only_rule", tables["g2_only"], rule),
    ):
        distances[tag] = target_preference(z, target, target[inverse]).tolist()
    predicted = {k: models["decoder"](z).argmax(-1).tolist() for k, z in tables.items()}
    store.write(
        "targets.json",
        dict(
            keys=[cases[k]["key"] for k in keys],
            inverse=inverse,
            equivalent=same,
            target_gap=distances,
            predicted=predicted,
        ),
    )
    # Unique actual BF16 tensors plus a lossless route/row index, including the canonical case reference.
    pool = {}
    saved = []
    addresses = {}
    for name, z in [*tables.items(), ("canonical_case", canonical_cases)]:
        addresses[name] = []
        for row in z:
            st = State(slots=row)
            tag = st.sha()
            if tag not in pool:
                pool[tag] = len(pool)
                saved.append(row.contiguous().view(torch.uint16).numpy())
            addresses[name].append(pool[tag])
    np.savez_compressed(store.path / "fresh_states.npz", states=np.stack(saved))
    store.write(
        "state_index.json",
        dict(
            routes=addresses,
            sha256=list(pool),
            dtype="bfloat16",
            shape=[host.slots, host.width],
        ),
    )
    rule_book = {
        2 * t + p: c["source"]
        for (t, i, s, p, st, o), c in train.items()
        if i == 0 and s == base["fit"]["styles"][0] and st == "rule"
    }
    if set(rule_book) != set(range(16)):
        raise ValueError("Incomplete frozen compact rule book")
    reader, _, _ = load_recipient(focus, host)
    reader_before = parameter_sha(reader)
    cache = {}
    logical = 0
    try:
        with gzip.open(store.path / "decisions.jsonl.gz", "wt", encoding="utf8") as out:
            for mode in ("raw", "decoded"):
                names = list(tables) if mode == "raw" else [*tables, "oracle"]
                for name in names:
                    for i, k in enumerate(keys):
                        c = cases[k]
                        topic, _, _, pole, _, _ = k
                        if mode == "raw":
                            state = tables[name][i].to(host.device)
                            statekey = pool[State(slots=tables[name][i]).sha()]
                            prompt_case = c
                            sid = None
                            route = "matched"
                        else:
                            state = None
                            sid = (
                                (2 * topic + pole)
                                if name == "oracle"
                                else predicted[name][i]
                            )
                            prompt_case = c | dict(
                                flags=base["fit"]["flag_pairs"][0],
                                source=rule_book[sid],
                            )
                            statekey = sid
                            route = "literal_base"
                        jobs = [
                            t
                            for w in range(4)
                            for t in tasks_for(prompt_case, groups[topic], f, w)
                        ]
                        questions = tuple(t["question"] for t in jobs)
                        ck = (mode, statekey, questions)
                        if ck not in cache:
                            cache[ck] = model_scores(
                                host, reader, jobs, {c["key"]: state}, route
                            )
                        for t, ans in zip(jobs, cache[ck], strict=True):
                            item = [
                                mode,
                                name,
                                c["key"],
                                t["world"],
                                t["label_order"],
                                t["gold"],
                                ans["greedy_token"],
                                ans["A"],
                                ans["B"],
                                digest(t["question"]),
                                statekey,
                            ]
                            out.write(json.dumps(item, separators=(",", ":")) + "\n")
                            logical += 1
    finally:
        unchanged = parameter_sha(reader) == reader_before
        reader.close()
    if not unchanged or parameter_sha(host.model) != backbone:
        raise ValueError("Frozen reader or backbone changed")
    return dict(
        status="DEVELOPMENT_G_CHAIN_COMPLETE",
        backbone=backbone,
        reader_sha256=file_sha(focus),
        reader_parameters=reader_before,
        captures=len(train) + len(cases),
        logical_choices=logical,
        distinct_native_batches=len(cache),
        compute=host.meter.snapshot(),
        qualification_loaded=False,
    )


if __name__ == "__main__":
    p = parser()
    p.add_argument("--literal", required=True, type=Path)
    a = p.parse_args()
    store = RunStore(a.out)
    start = time.monotonic()
    try:
        end = run(a, store)
    except Exception as exc:
        store.write("terminal.json", dict(status="TECHNICAL_STOP", error=str(exc)))
        raise
    store.write("terminal.json", end | dict(elapsed_seconds=time.monotonic() - start))
