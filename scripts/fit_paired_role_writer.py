"""Matched continuations of the structured Gemma writer on opposed-role donors."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fit_codec import parameter_sha
from fit_structured_report import (
    ROOT,
    GROUPS,
    checked_source,
    native_host,
    native_receipt,
    options,
    read,
    read_lines,
    score_rows,
)
from selffold.records import RunStore, digest, file_sha
from selffold.structured_report import (
    Midpoint,
    controller,
    final_choices,
    panel,
    read_fields,
    reencode,
    report_text,
    paired_change,
)
from selffold.structured_report_writer import (
    attach_writer,
    detach_writer,
    free_reports,
    pad_batch,
    training_record,
)

STUDY = ROOT / "provenance/paired_writer_20261123"
PARENT = ROOT / "provenance/structured_report_20261122/produce"
CFG = STUDY / "design.json"
ARMS = ("parent", "individual", "paired")
ROUTES = (*ARMS, "L", "oracle")


def twin_indices(queries):
    pairs = defaultdict(dict)
    for i, q in enumerate(queries):
        key = q["menu"], q["pattern"], q["rotation"]
        if q["default"] in pairs[key]:
            raise ValueError("Repeated donor")
        pairs[key][q["default"]] = i
    result = []
    for group in pairs.values():
        a, b = group[0], group[1]
        if (
            queries[a]["rows"] != queries[b]["rows"]
            or queries[a]["bit"] == queries[b]["bit"]
        ):
            raise ValueError(
                "An opposed pair must differ only in its role at the midpoint"
            )
        result.append((a, b))
    return result


def prepare():
    cfg = read(CFG)
    if file_sha(PARENT / "writer.npz") != cfg["parent_sha256"]:
        raise ValueError("Parent writer changed")
    actor = controller(ROOT, 17)
    menus = [g["options"] for g in read(GROUPS)["groups"]]
    names = [name for menu in menus for name in menu]
    train, test = [panel(actor, cfg, names, split) for split in ("train", "test")]
    fresh = [q for q in test if q["menu"] in cfg["fresh"]]
    seen = read(PARENT / "train_states.json") + read(PARENT / "test_states.json")
    key = lambda q: (q["menu"], tuple(q["rows"]))
    if (
        {key(q) for q in train} & {key(q) for q in test}
        or {key(q) for q in seen} & {key(q) for q in fresh}
        or (len(train), len(fresh), len(test)) != (512, 192, 384)
    ):
        raise ValueError("Assignment split leaked or has wrong size")
    return cfg, actor, menus, names, train, test


def weights(adapters):
    return {
        f"{i}_{name}_{direction}": getattr(adapter, direction).weight
        for (i, name), adapter in adapters.items()
        for direction in ("down", "up")
    }


def restore(adapters, path):
    params = weights(adapters)
    with torch.no_grad(), np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != params.keys():
            raise ValueError("Unexpected writer parameters")
        for name, p in params.items():
            value = torch.from_numpy(archive[name]).to(p)
            if value.shape != p.shape or not value.isfinite().all():
                raise ValueError("Invalid parameter")
            p.copy_(value)


def role_loss(margins, paired, target):
    losses = F.softplus(target - margins.reshape(-1, 2))
    return losses.amax(1).mean() if paired else losses.mean()


def fit(host, adapters, samples, twin_ids, cfg, arm, store):
    params = list(weights(adapters).values())
    if {id(p) for p in params} != {
        id(p) for p in host.model.parameters() if p.requires_grad
    }:
        raise ValueError("Unexpected trainable parameter")
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    rng = np.random.default_rng(cfg["seed"])
    for epoch in range(cfg["epochs"]):
        groups = rng.permutation(len(twin_ids)).reshape(-1, cfg["pairs_per_batch"])
        for group in groups:
            ix = [i for g in group for i in twin_ids[g]]
            batch = [samples[i] for i in ix]
            ids, mask = pad_batch(
                [s[0] for s in batch],
                host.tokenizer.pad_token_id,
                host.device,
                left=False,
            )
            logits = host.model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits
            active = torch.zeros_like(mask, dtype=torch.bool)
            for j, (seq, n, _) in enumerate(batch):
                active[j, n - 1 : len(seq) - 1] = True
            ce = F.cross_entropy(logits[active].float(), ids.roll(-1, dims=1)[active])
            choices = [c for _, _, fields in batch for c in fields]
            pos = torch.tensor([c[0] for c in choices], device=host.device)
            ids2 = torch.tensor([c[1] for c in choices], device=host.device)
            target = torch.tensor([c[2] for c in choices], device=host.device)
            which = torch.arange(len(batch), device=host.device).repeat_interleave(5)
            scores = logits[which[:, None], pos[:, None], ids2].float()
            binary = F.cross_entropy(scores, target)
            # The fifth field uses the same four teacher-forced rows in each twin.
            default = scores[4::5]
            signed = (default[:, 0] - default[:, 1]) * (1 - 2 * target[4::5])
            penalty = role_loss(signed, arm == "paired", cfg["margin"])
            loss = ce + binary + penalty
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            opt.step()
            store.append(
                arm + "_training.jsonl",
                dict(
                    epoch=epoch,
                    indices=ix,
                    ce=float(ce.detach()),
                    binary=float(binary.detach()),
                    role=float(penalty.detach()),
                ),
            )
        print(arm, "epoch", epoch + 1, flush=True)
    np.savez_compressed(
        store.path / (arm + ".npz"),
        **{k: v.detach().cpu().numpy() for k, v in weights(adapters).items()},
    )


def score(cfg, actor, names, test, records, banks):
    indexed = {(r["id"], r["arm"]): r for r in records}
    expected = {(q["id"], arm) for q in test for arm in ARMS}
    if set(indexed) != expected or len(indexed) != len(records):
        raise ValueError("Incomplete writer report panel")
    outcomes = []
    for q in test:
        z = Midpoint(tuple(q["rows"]), q["bit"])
        fields = list(z.fields())
        split = "fresh" if q["menu"] in cfg["fresh"] else "previous"
        for arm in ROUTES:
            if arm == "L":
                values, state = fields, z
            else:
                if arm == "oracle":
                    text = report_text([names[i] for i in fields])
                else:
                    record = indexed[q["id"], arm]
                    if record["prompt_sha256"] != digest(q["prompt"]):
                        raise ValueError("Report belongs to a different state")
                    text = record["text"]
                values = read_fields(text, names)
                state = reencode(values)
            result = final_choices(
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
                    rows_correct=[x == y for x, y in zip(values[:4], fields[:4])],
                    role=values[-1] == fields[-1],
                    **result,
                )
            )
    return outcomes


def summarize(rows):
    result = {}
    for split in ("fresh", "previous"):
        arms, unit = {}, {}
        for arm in ROUTES:
            subset = [r for r in rows if (r["split"], r["arm"]) == (split, arm)]
            groups = defaultdict(list)
            for r in subset:
                groups[r["menu"], r["pattern"], r["rotation"]].append(r)
            if len(groups) != 96 or any(
                sorted(r["default"] for r in p) != [0, 1] for p in groups.values()
            ):
                raise ValueError("Wrong donor pairing")
            units = {}
            for k, pair in groups.items():
                exact = all(all(r["rows_correct"]) for r in pair)
                role = all(r["role"] for r in pair)
                final = all(r["complete"] for r in pair)
                units[k] = dict(
                    rows=exact, role=role, final=final, joint=exact and role and final
                )
            unit[arm] = units
            arms[arm] = dict(
                rows=sum(sum(r["rows_correct"]) for r in subset),
                four_rows=sum(all(r["rows_correct"]) for r in subset),
                named_default=sum(r["role"] for r in subset),
                same_default=sum(
                    p[0]["fields"][4] is not None
                    and p[0]["fields"][4] == p[1]["fields"][4]
                    for p in groups.values()
                ),
                correct_choices=sum(r["correct"] for r in subset),
                twins={
                    key: sum(u[key] for u in units.values())
                    for key in ("rows", "role", "final", "joint")
                },
                by_menu={
                    str(m): sum(u["joint"] for k, u in units.items() if k[0] == m)
                    for m in sorted({k[0] for k in units})
                },
            )
        delta = {}
        for old, new in (
            ("parent", "paired"),
            ("individual", "paired"),
            ("parent", "individual"),
        ):
            delta[new + "_minus_" + old] = {
                key: paired_change(
                    {k: v[key] for k, v in unit[old].items()},
                    {k: v[key] for k, v in unit[new].items()},
                )
                for key in ("rows", "role", "final", "joint")
            }
        pa, ind, t = arms["parent"], arms["individual"], arms["paired"]
        improve = lambda a: sum(t["by_menu"][m] > a["by_menu"][m] for m in a["by_menu"])
        anchors = min(arms[r]["twins"]["joint"] for r in ("L", "oracle")) >= 92
        retention = t["four_rows"] >= 164 and t["four_rows"] >= pa["four_rows"] - 9
        positive = (
            anchors
            and retention
            and t["twins"]["joint"] >= pa["twins"]["joint"] + 10
            and improve(pa) >= 3
        )
        specific = (
            positive
            and t["twins"]["joint"] >= ind["twins"]["joint"] + 5
            and improve(ind) >= 3
        )
        result[split] = dict(
            arms=arms,
            changes=delta,
            anchors=anchors,
            row_retention=retention,
            transfer=positive,
            paired_specific=specific,
        )
    return result


def banks_from(root, cfg):
    return {
        s: native_receipt(root / ("native_" + s), cfg[s]) for s in ("fresh", "previous")
    }


def run(folder, checkpoint, seal):
    store = RunStore(folder)
    cfg, actor, menus, names, train, test = prepare()
    store.write("source.json", checked_source(seal))
    store.write(
        "binding.json",
        dict(
            train=digest(train),
            test=digest(test),
            plan=file_sha(STUDY / "PLAN.md"),
            design=file_sha(CFG),
            parent=cfg["parent_sha256"],
        ),
    )
    host, parity = native_host(cfg, checkpoint)
    base = parameter_sha(host.model)
    if host.identity != read(
        ROOT / "provenance/role_memory_20261104/run/produce/host.json"
    ):
        raise ValueError("Recipient host mismatch")
    store.write(
        "host.json", dict(identity=host.identity, base_sha256=base, parity=parity)
    )
    # Bind the final reader before fitting; its choices are private to evaluation.
    for s in ("fresh", "previous"):
        topics = cfg[s]
        score_rows(
            host,
            menus,
            torch.eye(16).reshape(8, 2, 16).repeat(len(topics), 1, 1),
            [m for m in topics for _ in range(8)],
            RunStore(folder / ("native_" + s)),
        )
    banks = banks_from(folder, cfg)
    samples = [
        training_record(
            host,
            q["prompt"],
            [names[i] for i in Midpoint(tuple(q["rows"]), q["bit"]).fields()],
            options(q, names),
        )
        for q in train
    ]
    groups = twin_indices(train)
    adapters = attach_writer(host, cfg)
    for arm in ARMS[1:]:
        restore(adapters, PARENT / "writer.npz")
        fit(host, adapters, samples, groups, cfg, arm, store)
    store.write(
        "fits_closed.json", {a: file_sha(folder / (a + ".npz")) for a in ARMS[1:]}
    )
    records = []
    ordered = sorted(test, key=lambda q: digest(["paired-role-v1", q["id"]]))
    for arm in ARMS:
        restore(
            adapters,
            PARENT / "writer.npz" if arm == "parent" else folder / (arm + ".npz"),
        )
        for start in range(0, len(ordered), 8):
            batch = ordered[start : start + 8]
            outputs = free_reports(host, [q["prompt"] for q in batch], 160)
            for q, r in zip(batch, outputs, strict=True):
                row = dict(arm=arm, id=q["id"], prompt_sha256=digest(q["prompt"]), **r)
                store.append("reports.jsonl", row)
                records.append(row)
        print(arm, "generation complete", flush=True)
    detach_writer(host, adapters)
    if base != parameter_sha(host.model):
        raise ValueError("Backbone changed")
    rows = score(cfg, actor, names, test, records, banks)
    store.write("outcomes.json", rows)
    store.write("summary.json", summarize(rows))
    store.write(
        "closed.json",
        dict(
            files={
                str(p.relative_to(folder)): file_sha(p)
                for p in sorted(folder.rglob("*"))
                if p.is_file()
            },
            stops=dict(Counter(r["stop"] for r in records)),
            reports=len(records),
        ),
    )


def replay(folder):
    cfg, actor, menus, names, train, test = prepare()
    closed = read(folder / "closed.json")
    required = {
        "binding.json",
        "source.json",
        "host.json",
        "fits_closed.json",
        "individual.npz",
        "paired.npz",
        "individual_training.jsonl",
        "paired_training.jsonl",
        "reports.jsonl",
        "outcomes.json",
        "summary.json",
    }
    required.update(
        f"native_{s}/{n}"
        for s in ("fresh", "previous")
        for n in ("native.json", "calls.json", "scores.npz")
    )
    if set(closed["files"]) != required or any(
        file_sha(folder / k) != v for k, v in closed["files"].items()
    ):
        raise ValueError("Incomplete or changed archived run")
    binding = read(folder / "binding.json")
    if binding != dict(
        train=digest(train),
        test=digest(test),
        plan=file_sha(STUDY / "PLAN.md"),
        design=file_sha(CFG),
        parent=cfg["parent_sha256"],
    ):
        raise ValueError("Study binding changed")
    trace = [read_lines(folder / (arm + "_training.jsonl")) for arm in ARMS[1:]]
    schedule = lambda tr: [(r["epoch"], r["indices"]) for r in tr]
    if schedule(trace[0]) != schedule(trace[1]) or len(trace[0]) != 256:
        raise ValueError("Mismatched training budget")
    for epoch in range(cfg["epochs"]):
        pairs = [
            tuple(r["indices"][i : i + 2])
            for r in trace[0]
            if r["epoch"] == epoch
            for i in range(0, 8, 2)
        ]
        if sorted(pairs) != sorted(twin_indices(train)):
            raise ValueError("Incorrect twin exposures")
    rows = score(
        cfg,
        actor,
        names,
        test,
        read_lines(folder / "reports.jsonl"),
        banks_from(folder, cfg),
    )
    if rows != read(folder / "outcomes.json") or summarize(rows) != read(
        folder / "summary.json"
    ):
        raise ValueError("Replay differs")
    return dict(
        exact=True,
        reports=closed["reports"],
        route_presentations=len(rows),
        steps_per_fit=len(trace[0]),
        new_model_calls=0,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--run", type=Path)
    modes.add_argument("--reduce", type=Path)
    modes.add_argument("--check", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--seal")
    a = p.parse_args()
    if a.check:
        *_, train, test = prepare()
        print(
            json.dumps(
                dict(
                    train=len(train),
                    test=len(test),
                    train_twins=len(twin_indices(train)),
                )
            )
        )
    elif a.reduce:
        print(json.dumps(replay(a.reduce)))
    else:
        if not a.checkpoint or not a.seal:
            p.error("a fit requires --checkpoint and --seal")
        run(a.run, a.checkpoint, a.seal)
