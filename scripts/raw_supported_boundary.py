"""Fit a paired boundary transport and exercise the native E/G/D path on development material."""

import gzip
import shutil
import json
import time

import numpy as np
import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_conditional_curriculum import parser
from fit_continuity_recipient import model_scores
from fit_stage_grammar import ACTIONS, read
from selffold import prompts
from selffold.carrier_probe import capture_description, description_layout
from selffold.cli import ROOT, make_host
from selffold.host import SlotHost
from selffold.records import (
    RunStore,
    State,
    TextOutput,
    code_manifest,
    digest,
    file_sha,
)
from selffold.recipient_learning import load_recipient
from selffold.supported_boundary import cases, questions
from selffold.transport import Transport, fit_ridge

CONFIG = ROOT / "data/supported_g_20261104.json"
READER = ROOT / "provenance/preprint_v1_20261104/raw/reader.npz"
OPS = tuple(prompts.OPERATORS)
PREFIX = "For this decision, use the stated policy. "


def new_host(args, context=512):
    args.max_context = context
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.34, device=args.device)
    host, ident = make_host(args)
    if ident["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Wrong host")
    host.max_forwards = 40000
    host.max_processed_tokens = 12000000
    return host


def save_npz(path, **arrays):
    np.savez_compressed(
        path,
        **{
            k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
            for k, v in arrays.items()
        },
    )


@torch.no_grad()
def geometry(host, ids, slots):
    """Full native logits, observed logits and matched-slot self-replay at the identical geometry."""
    x = torch.tensor([ids], device=host.device)
    host._budget(len(ids))
    ref = host.model(
        input_ids=x,
        attention_mask=torch.ones_like(x),
        use_cache=False,
        return_dict=True,
    ).logits
    host.meter.forwards += 1
    host.meter.processed_tokens += len(ids)
    observed, cap = host.forward(ids, capture_positions=slots, instrument=True)
    native = torch.equal(ref, observed)
    del ref
    replay, _ = host.forward(
        ids, state=State(slots=cap["pre_cut"][0]), input_positions=slots
    )
    return dict(
        native=native,
        neutral=torch.equal(observed, replay),
        width=len(ids),
        positions=slots,
        input_sha256=digest(ids),
    )


def fit_transport(host, menus, cfg, store):
    train = cases(cfg, "train", raw=True)
    writers, targets = [], []
    target_cache = {}
    first = None
    for c in train:
        source = c.source(menus, cfg["train"])
        text = c.successor(menus, cfg["train"])
        current, _ = capture_description(host, source, "workspace", forward_length=256)
        if text not in target_cache:
            target_cache[text] = capture_description(
                host, text, "workspace", forward_length=256
            )[0]
        for op in OPS:
            state = host.update_latent(current, op)
            writers.append(state.slots.cpu())
            targets.append(target_cache[text].slots.cpu())
            store.append(
                "pairs.jsonl",
                dict(
                    key=c.key,
                    op=op,
                    source=digest(source),
                    target=digest(text),
                    writer=state.sha(),
                    aligned=target_cache[text].sha(),
                ),
            )
        if first is None:
            first = current
    tr = fit_ridge(torch.cat(writers), torch.cat(targets), ridge=0.001)
    # Reload the artifact itself; the runtime G performs the writer-to-cut transform.
    path = store.path / "transport.npz"
    save_npz(path, weight=tr.weight, bias=tr.bias)
    host.transport = Transport.load(path)
    host.transport.weight = host.transport.weight.to(host.device)
    host.transport.bias = host.transport.bias.to(host.device)
    host.identity["transport_sha256"] = file_sha(path)
    store.write(
        "fit_closed.json",
        dict(
            train_sources=len(train),
            pairs=len(writers),
            rows=8 * len(writers),
            ridge=0.001,
            transport=file_sha(path),
            fresh_sources=0,
        ),
    )
    return first


def free_subset(c):
    return c.topic in (0, 1) and c.base == c.topic and c.order == c.style


@torch.no_grad()
def run(args, store):
    cfg = read(CONFIG)
    menus = [g["options"] for g in read(ACTIONS)["groups"]]
    seal = code_manifest(ROOT)
    if seal["git_status"] or file_sha(READER) != cfg["raw_reader"]:
        raise ValueError("Source or task reader mismatch")
    store.write("source.json", seal)
    host = new_host(args)
    backbone = parameter_sha(host.model)
    base_identity = dict(host.identity)
    if host.transport.weight is not None:
        raise ValueError("Fitting needs native post-writer activations")
    # A bounded generation host owns G and D. The task adapter keeps its independently
    # validated native host, with the same frozen model but no writer transport.
    task_host = SlotHost(
        host.model,
        host.tokenizer,
        slots=host.slots,
        cut=host.cut,
        writer=host.writer,
        max_context=1024,
        max_new_tokens=None,
        wall_seconds=host.wall_seconds,
        max_forwards=host.max_forwards,
        max_processed_tokens=host.max_processed_tokens,
    )
    task_host.identity.update(
        {
            k: base_identity[k]
            for k in (
                "profile",
                "checkpoint_sha256",
                "torch_version",
                "transformers_version",
            )
        }
    )
    task_host.meter = host.meter
    task_host.started = host.started
    if task_host.identity != read(
        ROOT / "provenance/preprint_v1_20261104/raw/host.json"
    ):
        raise ValueError("The scoring interface differs from the archived one")
    continued = getattr(args, "continue_capture", None)
    previous_free = {}
    if continued:
        if (
            read(continued / "terminal.json")
            != {
                "error": "Recipient artifact belongs to another host",
                "status": "TECHNICAL_STOP",
            }
            or read(continued / "source.json")["git_commit"]
            != "42de7977995ea39bc49ff4568fc38b4f1bdd04ed"
            or file_sha(continued / "transport.npz")
            != read(continued / "fit_closed.json")["transport"]
            or (continued / "decisions.jsonl.gz").exists()
        ):
            raise ValueError(
                "Only the recorded pre-score binding failure can be continued"
            )
        for name in ("transport.npz", "fit_closed.json", "pairs.jsonl"):
            shutil.copyfile(continued / name, store.path / name)
        host.transport = Transport.load(store.path / "transport.npz")
        host.transport.weight = host.transport.weight.to(host.device)
        host.transport.bias = host.transport.bias.to(host.device)
        host.identity["transport_sha256"] = host.transport.artifact_sha256
        first = cases(cfg, "train", raw=True)[0]
        unrelated = capture_description(
            host, first.source(menus, cfg["train"]), "workspace", forward_length=256
        )[0]
        previous_free = {
            (tuple(r["key"]), r["stage"]): r
            for r in map(
                json.loads, (continued / "free_text.jsonl").read_text().splitlines()
            )
        }
        store.write(
            "continued.json",
            dict(
                source=file_sha(continued / "source.json"),
                terminal=read(continued / "terminal.json"),
                free_text=file_sha(continued / "free_text.jsonl"),
                sources=file_sha(continued / "sources.jsonl"),
                refits=0,
            ),
        )
    else:
        unrelated = fit_transport(host, menus, cfg, store)
    unrelated = host.update_latent(unrelated, "boundary")
    fresh = cases(cfg, "fresh", raw=True)
    sample = fresh[0]
    source = sample.source(menus, cfg["fresh"])
    checks = {}
    for name, text in (
        ("source", source),
        ("prefixed_source", sample.source(menus, cfg["fresh"], cfg["neutral_prefix"])),
    ):
        ids, positions, _ = description_layout(
            host, text, "workspace", forward_length=256
        )
        checks[name] = geometry(host, ids, positions)
    for op in OPS:
        ids, pos = host.render(prompts.update_prompt(op))
        checks["update_" + op] = geometry(host, ids, pos[prompts.INPUT_MARKER])
    for name, content in (
        ("free_read", prompts.read_prompt()),
        ("task", prompts.task_prompt(questions(sample, menus)[0]["question"])),
        (
            "prefixed_task",
            prompts.task_prompt(questions(sample, menus, prefix=PREFIX)[0]["question"]),
        ),
    ):
        ids, pos = host.render(content)
        checks[name] = geometry(host, ids, pos[prompts.INPUT_MARKER])
    store.write("parity.json", checks)
    if not all(x["native"] and x["neutral"] for x in checks.values()):
        raise ValueError("Native geometry or self-replay mismatch")
    store.write(
        "binding.json",
        dict(
            host=host.identity,
            task_host=task_host.identity,
            base_host=base_identity,
            reader=file_sha(READER),
            source=seal["source_sha256"],
            config=file_sha(CONFIG),
            capture=256,
            template=prompts.BASE,
            backbone=backbone,
            qualification_content_loaded=False,
        ),
    )
    encoded = {}
    updated = {}
    extra = {}
    teacher = {}
    state_before = {}
    for c in fresh:
        text = c.source(menus, cfg["fresh"])
        z, meta = capture_description(host, text, "workspace", forward_length=256)
        encoded[c] = z
        prefixed = c.source(menus, cfg["fresh"], cfg["neutral_prefix"])
        zp, mp = capture_description(host, prefixed, "workspace", forward_length=256)
        extra[c] = host.update_latent(zp, "boundary")
        teacher[c] = capture_description(
            host, c.successor(menus, cfg["fresh"]), "workspace", forward_length=256
        )[0]
        updated[c] = {op: host.update_latent(z, op) for op in OPS}
        state_before[c] = (z.sha(), updated[c]["boundary"].sha())
        store.append(
            "sources.jsonl",
            dict(
                key=c.key,
                source=digest(text),
                hashes=state_before[c],
                target=teacher[c].sha(),
                aligned_positions=meta["capture_positions"],
                prefixed_positions=mp["capture_positions"],
                gold=c.ledger(),
            ),
        )
    free = {}
    for c in fresh:
        if free_subset(c):
            for stage, z in (
                ("initial", encoded[c]),
                ("updated", updated[c]["boundary"]),
            ):
                if continued:
                    original = previous_free.pop((c.key, stage))
                    if original["state"] != z.sha():
                        raise ValueError("Reconstructed free-readout state differs")
                    answer = TextOutput(
                        **{k: original[k] for k in TextOutput.__dataclass_fields__}
                    )
                else:
                    answer = host.decode(z.clone())
                free[c, stage] = answer
                store.append(
                    "free_text.jsonl",
                    dict(key=c.key, stage=stage, state=z.sha(), **answer.receipt()),
                )
    if continued and (
        previous_free
        or file_sha(store.path / "free_text.jsonl")
        != file_sha(continued / "free_text.jsonl")
        or file_sha(store.path / "sources.jsonl")
        != file_sha(continued / "sources.jsonl")
    ):
        raise ValueError("Captured sources or original generations differ")
    reader, _, _ = load_recipient(READER, task_host)
    rng = torch.Generator().manual_seed(4719)
    random_dir = torch.randn(host.slots, host.width, generator=rng).to(host.device)
    random_dir = torch.nn.functional.normalize(random_dir, dim=-1)
    # One fixed unrelated norm sets the random control magnitude independently of the tested donor.
    random_state = State(
        slots=(random_dir * unrelated.slots.float().norm(dim=-1, keepdim=True)).to(
            host.dtype
        )
    )
    cache = {}

    def score(jobs, z, kind):
        key = (
            tuple(t["question"] for t in jobs),
            jobs[0]["source"] if kind == "literal_base" else None,
            None if z is None else z.sha(),
            kind,
        )
        if key not in cache:
            cache[key] = model_scores(
                task_host,
                reader,
                jobs,
                {"current": None if z is None else z.slots},
                kind,
            )
        return cache[key]

    def emit(stream, c, tag, jobs, answers):
        for t, r in zip(jobs, answers, strict=True):
            stream.write(
                json.dumps(
                    [
                        *c.key,
                        tag,
                        t["world"],
                        t["label_order"],
                        t["gold"],
                        r["greedy_label"],
                        r["greedy_token"],
                        r["A"],
                        r["B"],
                        digest(t["question"]),
                    ],
                    separators=(",", ":"),
                )
                + "\n"
            )

    missing = [dict(greedy_label=None, greedy_token=None, A=None, B=None)] * 8
    try:
        with gzip.open(store.path / "decisions.jsonl.gz", "wt") as out:
            for c in fresh:
                jobs = questions(c, menus)
                z = updated[c]["boundary"]
                norms = z.slots.float().norm(dim=-1, keepdim=True)
                normonly = State(slots=(random_dir * norms).to(host.dtype))
                routes = dict(
                    identity=encoded[c],
                    teacher=teacher[c],
                    **updated[c],
                    opposed=updated[c.donor("opposed")]["boundary"],
                    equivalent=updated[c.donor("equivalent")]["boundary"],
                    unrelated=unrelated,
                    zero=State(slots=torch.zeros_like(z.slots)),
                    empty=None,
                    random=random_state,
                    permuted=State(slots=z.slots.flip([0])),
                    norm_only=normonly,
                    source_prefix=extra[c],
                    reordered=updated[c.donor("reordered")]["boundary"],
                )
                for name, state in routes.items():
                    emit(out, c, name, jobs, score(jobs, state, "matched"))
                shifted = questions(c, menus, prefix=PREFIX)
                emit(out, c, "question_prefix", shifted, score(shifted, z, "matched"))
                literal = questions(c, menus, text=c.source(menus, cfg["fresh"]))
                emit(out, c, "literal", literal, score(literal, None, "literal_base"))
                for stage in ("initial", "updated"):
                    if (c, stage) not in free:
                        continue
                    ans = free[c, stage]
                    literal = questions(c, menus, text=ans.text)
                    literal_ok = bool(ans.text.strip())
                    if literal_ok:
                        # Check exact rendered length too; record a failure for unscorable text.
                        try:
                            fits = all(
                                len(
                                    host.render(
                                        prompts.task_prompt(
                                            t["question"], text=ans.text
                                        )
                                    )[0]
                                )
                                + 1
                                <= host.max_context
                                for t in jobs
                            )
                            value = (
                                score(literal, None, "literal_base")
                                if fits
                                else missing
                            )
                        except ValueError as e:
                            if "exceeds limit" not in str(e):
                                raise
                            value = missing
                    else:
                        value = missing
                    emit(out, c, "D_" + stage, literal, value)
                    roundtrip = None
                    if ans.text.strip():
                        try:
                            roundtrip = capture_description(
                                host, ans.text, "workspace", forward_length=256
                            )[0]
                        except ValueError as e:
                            if "exceeds" not in str(e):
                                raise
                    emit(
                        out,
                        c,
                        "ED_" + stage,
                        jobs,
                        score(jobs, roundtrip, "matched")
                        if roundtrip is not None
                        else missing,
                    )
    finally:
        reader.close()
    unchanged = all(
        state_before[c] == (encoded[c].sha(), updated[c]["boundary"].sha())
        for c in fresh
    )
    if not unchanged or parameter_sha(host.model) != backbone:
        raise ValueError("State or backbone changed")
    return dict(
        status="RAW_DEVELOPMENT_COMPLETE",
        fresh_states=len(fresh),
        operators=len(OPS),
        free_outputs=len(free),
        isolated=unchanged,
        backbone=backbone,
        distinct_score_batches=len(cache),
        compute=host.meter.snapshot(),
    )


if __name__ == "__main__":
    p = parser()
    from pathlib import Path

    p.add_argument("--continue-capture", type=Path)
    args = p.parse_args()
    store = RunStore(args.out)
    start = time.monotonic()
    try:
        result = run(args, store)
    except Exception as exc:
        store.write("terminal.json", dict(status="TECHNICAL_STOP", error=str(exc)))
        raise
    store.write(
        "terminal.json", result | dict(elapsed_seconds=time.monotonic() - start)
    )
