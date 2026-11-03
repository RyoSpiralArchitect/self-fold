"""Fit two- and three-bit role memories on native Gemma choice rewards."""

from itertools import product

import numpy as np
import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_conditional_curriculum import main
from fit_preprint_reader import CONFIG
from fit_stage_grammar import ACTIONS, read
from role_memory_native import PARENT, READER, READER_SHA
from selffold.cli import ROOT, make_host
from selffold.recipient_learning import load_recipient
from selffold.records import code_manifest, digest, file_sha
from selffold.state_only import ClauseReader
from selffold.three_role_capacity import (
    ASSIGNMENTS,
    PAIRS,
    evaluate,
    exact_code,
    learn_code,
    native_rewards,
    triplet,
)
from state_only_native import NativeChoice, capture_clauses


@torch.no_grad()
def score_tables(host, compiler, reader, menus, clause, topics, names, store, split):
    actions = [a for menu in menus for a in menu]
    pairs = [
        [actions[triplet(t)[a]], actions[triplet(t)[b]]]
        for t in topics
        for a, b in PAIRS
    ]
    native = NativeChoice(host, pairs, actions=actions)
    tables, clauses = [], []
    keys = list(product(range(len(topics)), range(len(names))))
    for menu, name in keys:
        a, b = names[name]
        clauses.append(
            [
                clause.format(a=a, b=b, action=actions[index])
                for index in triplet(topics[menu])
            ]
        )
    features = capture_clauses(host, sorted({s for row in clauses for s in row}))
    for texts in clauses:
        tables.append(
            compiler.value(torch.stack([features[s] for s in texts])).softmax(-1)
        )
    tables = torch.stack(tables)
    inputs = list(product(range(len(keys)), range(3), range(3), range(2)))
    winners, scores = [], []
    for start in range(0, len(inputs), 12):
        jobs = inputs[start : start + 12]
        values = torch.stack([tables[i, row] for i, row, p, o in jobs])
        menus_idx = [3 * keys[i][0] + pair for i, row, pair, o in jobs]
        logits = native.logits(reader, values, menus_idx, [o for i, r, p, o in jobs])
        winners.extend(logits.argmax(-1).tolist())
        scores.extend(logits.log_softmax(-1)[:, native.labels].tolist())
    dims = (len(keys), 3, 3, 2)
    top = np.asarray(winners, dtype=np.int32).reshape(dims)
    np.savez_compressed(
        store.path / (split + "_native.npz"),
        tables=tables.numpy(),
        top=top,
        ab=np.asarray(scores, dtype=np.float32).reshape(*dims, 2),
    )
    store.write(
        split + "_inputs.json",
        dict(
            topics=topics,
            names=names,
            keys=keys,
            clauses=clauses,
            actions=actions,
            prompts=native.binding["prompts_sha256"],
            labels=native.labels,
            table_sha256=[digest(row.numpy().tobytes().hex()) for row in tables],
        ),
    )
    return native_rewards(top, native.labels)


def run(args, store):
    cfg, design, actions = map(
        read, (CONFIG, ROOT / "data/state_only_20261104.json", ACTIONS)
    )
    seal = code_manifest(ROOT)
    compiler_sha = file_sha(PARENT / "compiler.npz")
    if (
        seal["git_status"]
        or file_sha(READER) != READER_SHA
        or compiler_sha
        != "ab3dd606f01630e202c72b382d51b74fe317b611604b2229d7c6d82d52e3b935"
    ):
        raise ValueError("Frozen source and native reader required")
    store.write("source.json", seal)
    host, identity = make_host(args)
    if (
        identity["checkpoint_sha256"] != CHECKPOINT_SHA
        or not host.native_parity()["exact"]
    ):
        raise ValueError("Pinned capture required")
    backbone = parameter_sha(host.model)
    store.write("host.json", host.identity)
    compiler = ClauseReader(2 * host.width, 16).requires_grad_(False)
    with np.load(PARENT / "compiler.npz", allow_pickle=False) as raw:
        compiler.load_state_dict({k: torch.from_numpy(raw[k]) for k in raw.files})
    reader, _, _ = load_recipient(READER, host)
    before = parameter_sha(reader)
    menus = [g["options"] for g in actions["groups"]]
    clause = design["styles"]["when"][1]
    try:
        fit = score_tables(
            host,
            compiler,
            reader,
            menus,
            clause,
            [0, 2, 4, 6],
            [cfg["capacity_fit_flags"]],
            store,
            "fit",
        )
        models = {
            "supplied_3bit": (
                np.arange(6),
                np.vstack([ASSIGNMENTS, np.zeros((2, 3), dtype=int)]),
            )
        }
        params = {}
        traces = {}
        for bits in (2, 3):
            models[f"exact_{bits}bit"] = exact_code(fit, bits)
            for seed in (17, 41, 73):
                tag = f"gradient_{bits}bit_seed{seed}"
                (w, r), trace = learn_code(fit, bits, seed)
                params[tag + "_writer"] = w
                params[tag + "_reader"] = r
                models[tag] = (w.argmax(-1), r.argmax(-1))
                traces[tag] = trace
        np.savez_compressed(store.path / "parameters.npz", **params)
        store.write(
            "fit_closed.json",
            dict(
                models={k: [w.tolist(), r.tolist()] for k, (w, r) in models.items()},
                parameters=file_sha(store.path / "parameters.npz"),
                traces=traces,
                updates=512,
                test_tables_compiled=0,
                compiler=compiler_sha,
                reader=READER_SHA,
            ),
        )
        # Reload hard endpoints before constructing any held-out table.
        stored = read(store.path / "fit_closed.json")["models"]
        models = {k: (np.array(w), np.array(r)) for k, (w, r) in stored.items()}
        held = score_tables(
            host,
            compiler,
            reader,
            menus,
            clause,
            [1, 3, 5, 7],
            cfg["capacity_test_flags"],
            store,
            "test",
        )
        results = {
            tag: {
                "fit": evaluate(w, r, fit),
                "test": evaluate(w, r, held),
                "test_erased": evaluate(w, r, held, erase=True),
            }
            for tag, (w, r) in models.items()
        }
        store.write("results.json", results)
    finally:
        after = parameter_sha(reader)
        reader.close()
    if before != after or parameter_sha(host.model) != backbone:
        raise ValueError("Frozen weights changed")

    def fraction(values):
        return values[0] / values[1]

    supplied = results["supplied_3bit"]["test"]
    native_pass = (
        fraction(supplied["native_histories"]) >= 0.9
        and fraction(supplied["native_decisions"]) >= 0.95
    )
    learned = all(
        results[f"gradient_3bit_seed{s}"]["test"]["address_histories"] == [6, 6]
        and fraction(results[f"gradient_3bit_seed{s}"]["test"]["native_histories"])
        >= 0.9
        for s in (17, 41, 73)
    )
    return dict(
        status="CAPACITY_COMPLETE",
        supplied_pass=native_pass,
        learned_pass=learned and native_pass,
        backbone=backbone,
        reader_parameter_sha=before,
        compute=host.meter.snapshot(),
    )


if __name__ == "__main__":
    main(run)
