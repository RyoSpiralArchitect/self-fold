"""Resolve a veto stated in the original description with a learned map on current memory."""

import gzip
import json
from typing import NamedTuple

import numpy as np
import torch
from torch.nn import functional as F

from fit_codec import parameter_sha
from fit_conditional_curriculum import main
from fit_continuity_recipient import model_scores
from fit_stage_grammar import ACTIONS, read
from raw_supported_boundary import CONFIG, new_host, save_npz
from role_memory_native import READER, READER_SHA
from selffold.cli import ROOT
from selffold.records import code_manifest, digest, file_sha
from selffold.recipient_learning import load_recipient
from selffold.role_memory import RoleMemory, truth_bits
from selffold.stage_transitions import StageMap, fit_stage_map
from selffold.supported_boundary import (
    cases,
    condition,
    questions,
    render_cases,
    row_text,
)
from state_only_native import NativeChoice, capture_clauses

OLD = ROOT / "provenance/role_memory_20261104/run/produce"
LITERAL = ROOT / "provenance/two_flags_20261104/run/parts.npz"
BASE_MASKS = torch.tensor(
    [sum(1 << w for w in range(4) if condition(p, w)) for p in range(4)]
)
GUARD_TRUTH = torch.tensor([[condition(g, w) for w in range(4)] for g in range(-1, 4)])


class Policy(NamedTuple):
    table: torch.Tensor
    bit: torch.Tensor
    guard: torch.Tensor


def ordinary_rows(policy, actor):
    ix = actor.read(policy.table, policy.bit).argmax(-1)[:, 0]
    return policy.table[torch.arange(len(ix)), ix]


def resolve(policy, gates, actor):
    overwrite = gates[policy.guard]
    rows = torch.where(
        overwrite[:, :, None], ordinary_rows(policy, actor)[:, None, :], policy.table
    )
    # A consumed guard cannot fire again. Replacing the reference row makes it regular.
    return Policy(
        rows, policy.bit * (~overwrite[:, 0]).long(), torch.zeros_like(policy.guard)
    )


def fit_heads(host, menus, train, styles):
    labelled = {k: {} for k in ("action", "role", "base", "guard")}

    def add(k, s, y):
        if s in labelled[k] and labelled[k][s] != y:
            raise ValueError("Conflicting source labels")
        labelled[k][s] = y

    for c in train:
        fields = c.fields(menus, styles)
        for i, s in enumerate(fields[:2]):
            r = i ^ c.order
            add("action", s, 2 * c.topic + (c.pole ^ r))
            add("role", s, r)
        add("base", fields[2], c.base)
        add("guard", fields[3], c.veto + 1)
    for t, menu in enumerate(menus):
        for j, a in enumerate(menu):
            for w in range(4):
                add("action", row_text(w, a), 2 * t + j)
    features = capture_clauses(host, sorted({s for l in labelled.values() for s in l}))
    heads = {}
    for kind, labels in labelled.items():
        texts = sorted(labels)
        target = F.one_hot(torch.tensor([labels[s] for s in texts])).float()
        heads[kind] = fit_stage_map(
            torch.stack([features[s] for s in texts]), target, rank=target.shape[1]
        )
    return heads, features


def encode_rules(host, heads, actor, fields, features=None):
    if features is None:
        features = capture_clauses(host, sorted({s for f in fields for s in f}))
    x = torch.stack([torch.stack([features[s] for s in f]) for f in fields])
    n = len(fields)
    words = x[:, :2].flatten(0, 1)
    actions = (12 * heads["action"](words)).softmax(-1).reshape(n, 2, 16)
    roles = heads["role"](words).reshape(n, 2, 2)
    regular = (roles[:, :, 0] - roles[:, :, 1]).argmax(-1)
    base = heads["base"](x[:, 2]).argmax(-1)
    guard = heads["guard"](x[:, 3]).argmax(-1)
    mask = BASE_MASKS[base]
    selector = regular[:, None] ^ truth_bits(mask).long()
    table = actions.gather(1, selector[:, :, None].expand(-1, -1, 16)).contiguous()
    return Policy(table, (actor.write(mask) > 0).long(), guard), dict(
        base=base, regular=regular, actions=actions.argmax(-1), guard=guard
    )


def fit_gate(policy, gold, actor, store):
    regular = ordinary_rows(policy, actor)
    params = torch.nn.Parameter(torch.zeros(5, 4))
    opt = torch.optim.Adam([params], lr=0.1)
    for step in range(128):
        opt.zero_grad(set_to_none=True)
        weight = params[policy.guard].sigmoid()[:, :, None]
        updated = weight * regular[:, None, :] + (1 - weight) * policy.table
        loss = -updated.gather(-1, gold[:, :, None]).clamp_min(1e-12).log().mean()
        loss.backward()
        if not torch.isfinite(params.grad).all():
            raise ValueError("Invalid G gradient")
        opt.step()
        if step in (0, 31, 63, 127):
            store.append(
                "training.jsonl",
                dict(
                    step=step + 1,
                    loss=float(loss.detach()),
                    gradient=float(params.grad.norm()),
                ),
            )
    return params.detach()


def rendered(table, actions):
    indices = table.argmax(-1).tolist()
    return [render_cases([actions[i] for i in ids]) for ids in indices]


def encode_rendered(host, head, texts):
    """Accept just the four emitted lines. No source state, policy field or gold is available here."""
    split = [s.splitlines() for s in texts]
    if any(len(v) != 4 for v in split):
        raise ValueError("Invalid four-row description")
    features = capture_clauses(host, sorted({s for lines in split for s in lines}))
    lines = sorted(features)
    predictions = (12 * head(torch.stack([features[s] for s in lines]))).softmax(-1)
    lookup = dict(zip(lines, predictions, strict=True))
    return torch.stack([torch.stack([lookup[s] for s in row]) for row in split])


@torch.no_grad()
def no_injection(host, reader, native, topic, order):
    ids, _ = native.prompts[topic, order]
    data = torch.tensor([ids], device=host.device)
    host._budget(data.numel())
    with reader.active():
        logits = (
            host.model(
                input_ids=data,
                attention_mask=torch.ones_like(data),
                position_ids=torch.arange(len(ids), device=host.device)[None],
                use_cache=False,
                logits_to_keep=1,
            )
            .logits[0, -1]
            .float()
        )
    host.meter.forwards += 1
    host.meter.processed_tokens += data.numel()
    return int(logits.argmax()), logits.log_softmax(-1)[native.labels].tolist()


def route_bank(host, native, reader, panels, topics, routes, store):
    dispatch = {}
    addresses = {}
    entries = []
    for name, tables in routes.items():
        ids = np.empty((len(topics), 4), np.int32)
        for i, topic in enumerate(topics):
            for w in range(4):
                row = None if tables is None else tables[i, w].contiguous()
                k = (topic, None if row is None else row.numpy().tobytes())
                if k not in addresses:
                    addresses[k] = len(entries)
                    entries.append((topic, row))
                ids[i, w] = addresses[k]
        dispatch[name] = ids
    native_rows = [
        (i, t, r, o)
        for i, (t, r) in enumerate(entries)
        if r is not None
        for o in (0, 1)
    ]
    best = np.empty((len(entries), 2), np.int32)
    ab = np.empty((len(entries), 2, 2), np.float32)
    with torch.no_grad():
        for j in range(0, len(native_rows), 8):
            group = native_rows[j : j + 8]
            ix, t, rows, orders = zip(*group, strict=True)
            scores = native.logits(reader, torch.stack(rows), t, orders)
            best[list(ix), list(orders)] = scores.argmax(-1).cpu().numpy()
            ab[list(ix), list(orders)] = (
                scores.log_softmax(-1)[:, native.labels].cpu().numpy()
            )
        for i, (t, r) in enumerate(entries):
            if r is None:
                for o in (0, 1):
                    best[i, o], ab[i, o] = no_injection(host, reader, native, t, o)
    save_npz(store.path / "dispatch.npz", **dispatch)
    save_npz(
        store.path / "native.npz",
        top=best,
        ab=ab,
        topics=np.array([t for t, _ in entries]),
        missing=np.array([r is None for _, r in entries]),
        rows=torch.stack([r if r is not None else torch.zeros(16) for _, r in entries]),
    )
    store.write(
        "native.json",
        dict(
            labels=native.labels,
            decisions=2 * len(entries),
            prompts=native.binding["prompts_sha256"],
        ),
    )
    return len(entries) * 2


@torch.no_grad()
def literal_scores(host, panel, menus, texts, store):
    reader, _, sha = load_recipient(LITERAL, host)
    if sha != "fa1258b01b165959c346e9f409d0cad43543e1d48bd0c48a9b8917cb08c0ec59":
        raise ValueError("Wrong literal reader")
    cache = {}
    addresses = {}
    try:
        for route, outputs in texts.items():
            ix = []
            for c, text in zip(panel, outputs, strict=True):
                k = (c.topic, text)
                if k not in cache:
                    jobs = questions(c, menus, text=text)
                    cache[k] = (
                        len(cache),
                        model_scores(
                            host, reader, jobs, {"current": None}, "literal_base"
                        ),
                    )
                ix.append(cache[k][0])
            addresses[route] = np.array(ix)
        groups = sorted(cache, key=lambda k: cache[k][0])
        scores = [cache[k][1] for k in groups]
        save_npz(
            store.path / "literal.npz",
            top=np.array(
                [[x["greedy_token"] for x in row] for row in scores], np.int32
            ).reshape(-1, 4, 2),
            ab=np.array(
                [[[x["A"], x["B"]] for x in row] for row in scores], np.float32
            ).reshape(-1, 4, 2, 2),
            **addresses,
        )
        with gzip.open(store.path / "rendered.jsonl.gz", "wt") as f:
            for topic, text in groups:
                f.write(json.dumps(dict(topic=topic, text=text)) + "\n")
    finally:
        reader.close()
    return len(cache)


def run(args, store):
    seal = code_manifest(ROOT)
    cfg = read(CONFIG)
    menus = [g["options"] for g in read(ACTIONS)["groups"]]
    actions = [a for m in menus for a in m]
    path = OLD / "fits.npz"
    parent = read(OLD / "fit_closed.json")
    if (
        seal["git_status"]
        or file_sha(path) != parent["weights"]
        or file_sha(READER) != READER_SHA
    ):
        raise ValueError("Dirty or unbound input")
    store.write("source.json", seal)
    host = new_host(args)
    backbone = parameter_sha(host.model)
    parity = host.native_parity()
    if not parity["exact"]:
        raise ValueError("Native parity")
    actor = RoleMemory(17).requires_grad_(False)
    with np.load(path, allow_pickle=False) as z:
        actor.load_state_dict(
            {k: torch.from_numpy(z["reward_17_" + k]) for k in actor.state_dict()}
        )
    train = cases(cfg, "train")
    heads, fitfeatures = fit_heads(host, menus, train, cfg["train"])
    p, _ = encode_rules(
        host, heads, actor, [c.fields(menus, cfg["train"]) for c in train], fitfeatures
    )
    gold = torch.tensor(
        [[2 * c.topic + r["choice"] for r in c.ledger()] for c in train]
    )
    gated = fit_gate(p, gold, actor, store)
    saved = store.path / "heads.npz"
    save_npz(
        saved,
        gates=gated,
        **{k + "_" + n: t for k, h in heads.items() for n, t in vars(h).items()},
    )
    with np.load(saved, allow_pickle=False) as z:
        heads = {
            k: StageMap(
                **{
                    n: torch.from_numpy(z[k + "_" + n])
                    for n in StageMap.__annotations__
                }
            )
            for k in heads
        }
        gated = torch.from_numpy(z["gates"])
    gate = gated > 0
    store.write(
        "fit_closed.json",
        dict(
            weights=file_sha(saved),
            one_bit_source=file_sha(path),
            one_bit_endpoint="reward_17",
            gate_logits=gated.tolist(),
            fit_sources=len(train),
            fresh_sources=0,
            updates=128,
            learned_gate_parameters=20,
            source_head_parameters=sum(
                v.numel() for h in heads.values() for v in vars(h).values()
            ),
        ),
    )
    panel = cases(cfg, "fresh")
    policy, fields = encode_rules(
        host, heads, actor, [c.fields(menus, cfg["fresh"]) for c in panel]
    )
    prefix, _ = encode_rules(
        host,
        heads,
        actor,
        [c.fields(menus, cfg["fresh"], cfg["neutral_prefix"]) for c in panel],
    )
    before = digest(policy.table.numpy().tobytes().hex())
    out = resolve(policy, gate, actor)
    if not torch.equal(out.table, resolve(out, gate, actor).table):
        store.write("second_application.json", dict(idempotent=False))
    index = {c: i for i, c in enumerate(panel)}

    def partner(k):
        return torch.tensor([index[c.donor(k)] for c in panel])

    twin = policy._replace(bit=policy.bit[partner("twin")])
    erased = policy._replace(bit=torch.zeros_like(policy.bit))
    flip = policy._replace(bit=1 - policy.bit)
    full = out.table
    rtexts = {
        k: rendered(tab, actions)
        for k, tab in (("D_initial", policy.table), ("D_updated", full))
    }
    count = len(panel)
    rng = torch.Generator().manual_seed(4719)
    rnd = torch.rand(16, generator=rng)
    rnd /= rnd.sum()
    rows = dict(
        identity=policy.table,
        updated=full,
        analytic=resolve(policy, GUARD_TRUTH, actor).table,
        opposed=full[partner("opposed")],
        equivalent=full[partner("equivalent")],
        reordered=full[partner("reordered")],
        source_prefix=resolve(prefix, gate, actor).table,
        erased=resolve(erased, gate, actor).table,
        complemented=resolve(flip, gate, actor).table,
        twin=resolve(twin, gate, actor).table,
        twin_erased=resolve(
            twin._replace(bit=torch.zeros_like(twin.bit)), gate, actor
        ).table,
        unrelated=resolve(Policy(*(x[:1] for x in p)), gate, actor).table.expand(
            count, -1, -1
        ),
        zero=torch.zeros_like(full),
        empty=None,
        random=rnd.expand_as(full),
        norm_only=F.normalize(rnd, dim=0) * full.norm(dim=-1, keepdim=True),
        permuted=full.flip([1]),
        ED_initial=encode_rendered(host, heads["action"], rtexts["D_initial"]),
        ED_updated=encode_rendered(host, heads["action"], rtexts["D_updated"]),
    )
    native = NativeChoice(host, menus)
    reader, _, _ = load_recipient(READER, host)
    frozen = parameter_sha(reader)
    try:
        exact = native.lexical_parity(reader)
        if not exact:
            raise ValueError("Native lexical parity")
        scored = route_bank(
            host, native, reader, panel, [c.topic for c in panel], rows, store
        )
        unchanged = parameter_sha(reader) == frozen
    finally:
        reader.close()
    scored_text = literal_scores(host, panel, menus, rtexts, store)
    isolation = digest(policy.table.numpy().tobytes().hex()) == before
    if not isolation or not unchanged or parameter_sha(host.model) != backbone:
        raise ValueError("Mutation")
    store.write(
        "parity.json",
        dict(
            native=parity,
            lexical=exact,
            unchanged_reader=unchanged,
            state_isolation=isolation,
        ),
    )
    store.write(
        "binding.json",
        dict(
            host=host.identity,
            source=seal["source_sha256"],
            config=file_sha(CONFIG),
            backbone=backbone,
            reader=READER_SHA,
            literal_reader=file_sha(LITERAL),
            writer=file_sha(path),
            writer_name="reward_17",
            source_and_update=file_sha(saved),
            qualification_content_loaded=False,
        ),
    )
    keys = np.array([c.key for c in panel])
    save_npz(
        store.path / "states.npz",
        keys=keys,
        table=policy.table,
        bit=policy.bit,
        guard=policy.guard,
        updated=full,
        next_bit=out.bit,
        next_guard=out.guard,
        prefix_table=prefix.table,
        prefix_bit=prefix.bit,
        prefix_guard=prefix.guard,
        field_base=fields["base"],
        field_regular=fields["regular"],
        field_actions=fields["actions"],
    )
    return dict(
        status="STRUCTURED_DEVELOPMENT_COMPLETE",
        sources=count,
        physical_native_queries=scored,
        distinct_literal_batches=scored_text,
        compute=host.meter.snapshot(),
        qualification_opened=False,
    )


if __name__ == "__main__":
    main(run)
