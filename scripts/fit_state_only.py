"""Fit and freeze all modules, then compile a blind numeric packet for a second process."""

from itertools import product
import json

import numpy as np
import torch
import torch.nn.functional as F

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_conditional_curriculum import main
from selffold.cli import ROOT, make_host
from selffold.records import RunStore, code_manifest, digest, file_sha
from selffold.recipient_learning import load_recipient, save_recipient
from selffold.state_only import (
    ClauseReader,
    OPERATIONS,
    PolicyUpdate,
    ROUTES,
    donor_states,
    selected_action,
    source_bank,
    training_clauses,
)
from state_only_native import NativeChoice, capture_clauses, fingerprint

DATA = ROOT / "data/state_only_20261104.json"
ACTIONS = ROOT / "data/within_topic_wording_20261103.json"
STUDY = ROOT / "provenance/state_only_20261104"
PARENT = ROOT / "provenance/two_flags_20261104/run/parts.npz"
PARENT_SHA = "fa1258b01b165959c346e9f409d0cad43543e1d48bd0c48a9b8917cb08c0ec59"


def mean_trace(store, name, step, loss):
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    store.append(
        "training.jsonl", dict(module=name, step=step + 1, loss=float(loss.detach()))
    )


def fit_compiler(host, design, actions, store):
    clauses = training_clauses(design, actions)
    cache = capture_clauses(host, [r[0] for r in clauses])
    features = torch.stack([cache[c] for c, _, _ in clauses])
    values, roles = [
        torch.tensor(r) for r in zip(*[r[1:] for r in clauses], strict=True)
    ]
    head = ClauseReader(features.shape[1], len(actions))
    opt = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=0.01)
    for step in range(design["compiler_steps"]):
        opt.zero_grad()
        loss = F.cross_entropy(head.value(features), values) + F.cross_entropy(
            head.role(features), roles
        )
        loss.backward()
        opt.step()
        mean_trace(store, "bind_clauses", step, loss)
    return head.requires_grad_(False), cache


def memories(head, sources, cache):
    with torch.no_grad():
        return {
            key: head.compile(torch.stack([cache[c] for c in text.splitlines()])[None])[
                0
            ].contiguous()
            for key, text in sources.items()
        }


def fit_update(design, fit_states, store):
    model = PolicyUpdate()
    opt = torch.optim.AdamW(model.parameters(), lr=0.12, weight_decay=0)
    states = torch.stack(list(fit_states.values()))
    targets = torch.tensor(
        [
            [[selected_action(k[0], k[3], op, w) for w in range(4)] for k in fit_states]
            for op in OPERATIONS
        ]
    )
    for step in range(design["update_steps"]):
        opt.zero_grad()
        tables = model(states[None], torch.arange(len(OPERATIONS))[:, None])
        loss = -tables.gather(-1, targets[..., None]).clamp_min(1e-12).log().mean()
        loss.backward()
        opt.step()
        mean_trace(store, "revise_condition", step, loss)
    return model.requires_grad_(False)


def fit_native(host, menus, design, packet, store, seal):
    native = NativeChoice(host, menus)
    reader, _, parent = load_recipient(PARENT, host, trainable=True)
    if parent != PARENT_SHA:
        raise ValueError("Unexpected starting reader")
    rng = torch.Generator().manual_seed(design["seed"])
    calibration = list(product(range(len(menus)), (0, 1), (0, 1)))
    opt = torch.optim.AdamW(reader.parameters(), lr=0.0002, weight_decay=0)
    try:
        if not native.lexical_parity(reader):
            raise ValueError("Lexical embedding parity failed")
        for step in range(design["reader_steps"]):
            if step % 2 == 0:
                perm = torch.randperm(len(calibration), generator=rng).tolist()
            rows = [calibration[i] for i in perm[(step % 2) * 16 : (step % 2 + 1) * 16]]
            groups, values, orders = zip(*rows)
            weights = F.one_hot(
                torch.tensor([2 * g + v for g, v in zip(groups, values)]), 16
            ).float()
            gold = torch.tensor(
                [native.labels[v ^ o] for v, o in zip(values, orders)],
                device=host.device,
            )
            opt.zero_grad()
            logits = native.logits(reader, weights, groups, orders)
            loss = F.cross_entropy(logits, gold)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                reader.parameters(), 1.0, error_if_nonfinite=True
            )
            if any(p.grad is not None for p in host.model.parameters()):
                raise RuntimeError("Backbone gradient")
            opt.step()
            mean_trace(store, "read_action", step, loss)
        with torch.no_grad():
            check = native.logits(reader, weights[:2], groups[:2], orders[:2]).clone()
        path = packet.path / "reader.npz"
        sha = save_recipient(
            path,
            reader,
            host.identity,
            dataset_sha256=file_sha(DATA),
            source_sha256=seal["source_sha256"],
            training=dict(parent=parent, steps=design["reader_steps"]),
        )
    finally:
        reader.close()
    restored, _, _ = load_recipient(path, host)
    try:
        with torch.no_grad():
            if not torch.equal(
                check, native.logits(restored, weights[:2], groups[:2], orders[:2])
            ):
                raise ValueError("Native reader replay failed")
    finally:
        restored.close()
    return sha, native.binding


def write_packet(design, groups, head, host, packet, store):
    states, state_ids, rows, cases = [], {}, [], []
    for split in ("primary", "challenge"):
        sources = dict(source_bank(design, groups, split))
        texts = sorted(
            {clause for text in sources.values() for clause in text.splitlines()}
        )
        mid = memories(head, sources, capture_clauses(host, texts))
        for (topic, flag, style, pole, order), matched in mid.items():
            key = topic, flag, style, pole, order
            opp = topic, flag, style, 1 - pole, order
            eq = topic, 1 - flag, style, pole, 1 - order
            oracle = F.one_hot(
                torch.tensor(
                    [selected_action(topic, pole, OPERATIONS[0], w) for w in range(4)]
                ),
                16,
            ).float()
            cases.append(
                dict(
                    split=split,
                    topic=topic,
                    flag=flag,
                    style=style,
                    pole=pole,
                    order=order,
                    source_sha256=digest(sources[key]),
                    opposed_sha256=digest(sources[opp]),
                    equivalent_sha256=digest(sources[eq]),
                    mid_sha256=fingerprint(matched),
                )
            )
            routes = donor_states(matched, mid[opp], mid[eq], oracle)
            for route in ROUTES:
                state = routes[route]
                sha = fingerprint(state)
                if sha not in state_ids:
                    state_ids[sha] = len(states)
                    states.append(state.numpy())
                for op, w, label_order in product(
                    range(len(OPERATIONS)), range(4), range(2)
                ):
                    rows.append(
                        (
                            state_ids[sha],
                            op,
                            w,
                            topic,
                            label_order,
                            int(route == "omit"),
                        )
                    )
    # The consumer sees shuffled numeric requests. The inverse order and source descriptions stay outside its packet.
    perm = np.random.default_rng(design["seed"]).permutation(len(rows))
    np.savez_compressed(
        packet.path / "inputs.npz",
        states=np.stack(states),
        requests=np.array(rows, dtype=np.int32)[perm],
    )
    np.savez_compressed(store.path / "grading_order.npz", restore=np.argsort(perm))
    store.write("cases.json", cases)
    return len(cases), len(rows)


def run(args, store):
    seal = code_manifest(ROOT)
    design, actions = (json.loads(p.read_text()) for p in (DATA, ACTIONS))
    if (
        seal["git_status"]
        or not seal["git_commit"]
        or file_sha(ACTIONS) != design["action_bank_sha256"]
    ):
        raise ValueError("Need clean pinned implementation and action bank")
    if file_sha(PARENT) != PARENT_SHA:
        raise ValueError("Parent bytes differ")
    store.write("source.json", seal)
    store.write(
        "request.json",
        dict(
            data=file_sha(DATA),
            plan=file_sha(STUDY / "PLAN.md"),
            scorer=file_sha(STUDY / "summarize.py"),
            parent=PARENT_SHA,
        ),
    )
    host, ident = make_host(args)
    parity = host.native_parity()
    if ident["checkpoint_sha256"] != CHECKPOINT_SHA or not parity["exact"]:
        raise ValueError("Native parity or backbone identity failed")
    store.write("host.json", host.identity)
    store.write("parity.json", parity)
    start = parameter_sha(host.model)
    torch.manual_seed(design["seed"])
    groups = actions["groups"]
    menus = [g["options"] for g in groups]
    packet = RunStore(store.path / "packet")
    packet.write("menus.json", menus)
    head, cache = fit_compiler(host, design, sum(menus, []), store)
    np.savez_compressed(
        store.path / "compiler.npz",
        **{k: v.numpy() for k, v in head.state_dict().items()},
    )
    replay = ClauseReader(host.width * 2, 16)
    with np.load(store.path / "compiler.npz", allow_pickle=False) as f:
        replay.load_state_dict({k: torch.from_numpy(f[k]) for k in f.files})
    fit = dict(source_bank(design, groups, "fit"))
    fit_states = memories(replay, fit, cache)
    if not all(
        torch.equal(v, fit_states[k]) for k, v in memories(head, fit, cache).items()
    ):
        raise ValueError("Compiler replay failed")
    update = fit_update(design, fit_states, store)
    np.savez_compressed(
        packet.path / "update.npz", routing=update.routing.detach().numpy()
    )
    sha, binding = fit_native(host, menus, design, packet, store, seal)
    store.write(
        "fit_closed.json",
        dict(
            compiler=file_sha(store.path / "compiler.npz"),
            update=file_sha(packet.path / "update.npz"),
            reader=sha,
            fresh_captures=0,
            native_logits_replayed=True,
            lexical_one_hot_parity=True,
            fit_tables_correct=sum(
                int(
                    all(
                        t.argmax(-1)[w] == selected_action(k[0], k[3], OPERATIONS[0], w)
                        for w in range(4)
                    )
                )
                for k, t in fit_states.items()
            ),
            fit_tables=len(fit_states),
        ),
    )
    cases, paths = write_packet(design, groups, replay, host, packet, store)
    packet.write(
        "binding.json",
        dict(
            files={p.name: file_sha(p) for p in sorted(packet.path.iterdir())},
            host=digest(host.identity),
            native=binding,
        ),
    )
    if start != parameter_sha(host.model):
        raise RuntimeError("Backbone changed")
    return dict(
        status="STATE_PACKET_CLOSED",
        cases=cases,
        paths=paths,
        backbone_sha256=start,
        compute=host.meter.snapshot(),
        qualification_content_loaded=False,
    )


if __name__ == "__main__":
    main(run)
