"""A fixed E-to-lexical ridge fit followed by one bounded residual fit."""

import argparse
import json
from pathlib import Path
import time

import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_continuity_recipient import batch_inputs, model_scores, training_loss
from probe_slot_readout import update
from selffold import prompts
from selffold.cli import ROOT, make_host
from selffold.recipient_learning import RecipientAdapter, batched_recipient_logits
from selffold.records import RunStore, State, code_manifest, digest, file_sha
from selffold.replay_probe import _capture
from selffold.source_mapping import (
    fit_source_ridge,
    load_lexical_interface,
    save_source_mapper,
    load_source_mapper,
)

STUDY = ROOT / "provenance/source_lexical_20261103"
DATA = ROOT / "data/source_lexical_20261103.json"
PRIOR = ROOT / "provenance/slot_readout_20261103/run/lexical"
LEXICAL_SHA = "fcedb9502207336dc00fd307b82d2c6632aa7078d392e9322f8cebb2a9e2b096"
ROUTES = ("matched", "opposed", "mean", "zero", "reverse", "norm")


def source_examples(data):
    result = []
    for group in data["groups"]:
        for pole in ("plus", "minus"):
            for phrasing, source in enumerate(group[pole]):
                split = (
                    ("seen_sources" if phrasing < 2 else "new_style")
                    if group["split"] == "train"
                    else group["exposure"]
                )
                result.append(
                    {
                        "key": f"{group['id']}-{pole}-p{phrasing}",
                        "group": group["id"],
                        "pole": pole,
                        "phrasing": phrasing,
                        "style": phrasing // 2,
                        "source_order": phrasing % 2,
                        "split": split,
                        "source": source,
                    }
                )
    return result


def question_tasks(data, examples):
    groups = {g["id"]: g for g in data["groups"]}
    result = []
    for example in examples:
        group = groups[example["group"]]
        for wording, stem in enumerate(group["questions"]):
            for order, options in enumerate((group["options"], group["options"][::-1])):
                question = f"{stem}\nA = {options[0]}. B = {options[1]}. Use the supplied description."
                gold = "A" if (example["pole"] == "plus") == (order == 0) else "B"
                result.append(
                    example
                    | {
                        "wording": wording,
                        "label_order": order,
                        "gold": gold,
                        "question": question,
                    }
                )
    return result


def capture_lexical(host, words, *, forward_length=256):
    template, markers = host.render(prompts.task_prompt(""))
    positions = markers[prompts.INPUT_MARKER]
    tokens = host.tokenizer.encode(words, add_special_tokens=False)
    if not 1 <= len(tokens) <= host.slots or positions != list(
        range(positions[0], positions[0] + 8)
    ):
        raise ValueError("Invalid lexical teacher geometry")
    prefix = (
        template[: positions[0]] + tokens + [host.placeholder_id] * (8 - len(tokens))
    )
    if len(prefix) > forward_length:
        raise ValueError("Lexical prefix exceeds fixed donor length")
    ids = prefix + [host.placeholder_id] * (forward_length - len(prefix))
    _, captured = host.forward(ids, capture_positions=positions)
    return State(slots=captured["pre_cut"][0]), {
        "native_ids": ids,
        "capture_positions": positions,
        "source_words": words,
    }


def replay_previous(host, lexical, adapter, store):
    groups = json.loads((ROOT / "data/slot_contrasts.dev.json").read_text())["groups"]
    expected = {
        (r["group"], r["pole"], r["label_order"]): r
        for r in rows(PRIOR / "decisions.jsonl")
        if r["route"] == "bridge_matched"
    }
    for group in groups:
        for order, options in enumerate((group["options"], group["options"][::-1])):
            question = f"{group['question']}\nA = {options[0]}. B = {options[1]}. Use the supplied description."
            prompt, markers = host.render(prompts.task_prompt(question))
            pos = markers[prompts.INPUT_MARKER]
            ids = prompt + [host.placeholder_id]
            for pole, words in zip(("plus", "minus"), group["options"], strict=True):
                tokens = host.tokenizer.encode(words, add_special_tokens=False)
                donor = (
                    ids[: pos[0]] + tokens + [host.placeholder_id] * (8 - len(tokens))
                )
                donor += [host.placeholder_id] * (len(ids) - len(donor))
                _, captured = _capture(host, donor, pos)
                with torch.no_grad():
                    mapping = {"replay": lexical(captured[0])}
                    score = model_scores(
                        host,
                        adapter,
                        [{"key": "replay", "source": words, "question": question}],
                        mapping,
                        "matched",
                    )[0]
                ref = expected[group["id"], pole, order]
                if any(
                    score[k] != ref[k]
                    for k in ("A", "B", "greedy_token", "greedy_label")
                ):
                    raise ValueError("Frozen lexical endpoint no longer replays")
                store.append(
                    "lexical_replay.jsonl",
                    {"group": group["id"], "pole": pole, "label_order": order, **score},
                )


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def save_checked(
    host, mapper, lexical, adapter, lexical_sha, seal, store, name, source, sample_tasks
):
    metadata = {
        "base_host_sha256": digest(host.identity),
        "lexical_sha256": lexical_sha,
        "source_sha256": seal["source_sha256"],
        "dataset_sha256": file_sha(DATA),
        "plan_sha256": file_sha(STUDY / "PLAN.md"),
        "endpoint": name,
        "capture": "original_host_encode_no_questions",
        "ridge": 0.001,
        "residual_updates": 0 if name == "ridge" else 64,
    }
    with torch.no_grad():
        expected = mapper(source).clone()
        mapping = {t["key"]: lexical(expected) for t in sample_tasks}
        expected_logits = batched_recipient_logits(
            host, adapter, *batch_inputs(host, sample_tasks, mapping, "matched")
        ).clone()
    path = store.path / (name + ".npz")
    sha = save_source_mapper(path, mapper, metadata=metadata)
    restored, _, actual = load_source_mapper(
        path,
        host_identity=host.identity,
        lexical_sha256=lexical_sha,
        device=host.device,
    )
    with torch.no_grad():
        after = restored(source)
        mapping = {t["key"]: lexical(after) for t in sample_tasks}
        actual_logits = batched_recipient_logits(
            host, adapter, *batch_inputs(host, sample_tasks, mapping, "matched")
        )
    if (
        sha != actual
        or not torch.equal(expected, after)
        or not torch.equal(expected_logits, actual_logits)
    ):
        raise ValueError("Source mapper save/load mismatch")
    store.append(
        "fits.jsonl",
        metadata | {"artifact": path.name, "sha256": sha, "serialization_exact": True},
    )
    return restored


def evaluate(host, adapter, lexical, mapper, tasks, sources, teachers, store, endpoint):
    generator = torch.Generator().manual_seed(4711)
    directions = torch.nn.functional.normalize(
        torch.randn(8, host.width, generator=generator), dim=-1
    ).to(host.device)
    conditions = (
        ROUTES if mapper is not None else ("raw_E", "literal", "lexical_teacher")
    )
    for route in conditions:
        with torch.no_grad():
            mapping = {}
            for key, source in sources.items():
                if mapper is None:
                    mapping[key] = (
                        None
                        if route == "literal"
                        else source.slots
                        if route == "raw_E"
                        else lexical(teachers[key].slots)
                    )
                    continue
                value = source.slots
                if route == "opposed":
                    other = (
                        key.replace("-plus-", "-minus-")
                        if "-plus-" in key
                        else key.replace("-minus-", "-plus-")
                    )
                    value = sources[other].slots
                elif route == "mean":
                    value = mapper.source_mean.reshape(8, host.width).to(value.dtype)
                elif route == "zero":
                    value = torch.zeros_like(value)
                elif route == "reverse":
                    value = value.flip(0)
                elif route == "norm":
                    value = (directions * value.float().norm(dim=-1, keepdim=True)).to(
                        value.dtype
                    )
                mapping[key] = lexical(mapper(value))
            for start in range(0, len(tasks), 4):
                batch = tasks[start : start + 4]
                scored = model_scores(
                    host,
                    adapter,
                    batch,
                    mapping,
                    "literal_base" if route == "literal" else "matched",
                )
                for task, score in zip(batch, scored, strict=True):
                    record = {
                        k: task[k]
                        for k in (
                            "key",
                            "group",
                            "pole",
                            "phrasing",
                            "style",
                            "source_order",
                            "split",
                            "wording",
                            "label_order",
                            "gold",
                        )
                    }
                    store.append(
                        "decisions.jsonl",
                        record
                        | {
                            "endpoint": endpoint,
                            "route": route,
                            "source_sha256": digest(task["source"]),
                            "question_sha256": digest(task["question"]),
                            **score,
                        },
                    )


def run(args, store):
    seal = code_manifest(ROOT)
    store.write("source.json", seal)
    if seal["git_status"] or not seal["git_commit"]:
        raise ValueError("Need clean sealed source")
    data = json.loads(DATA.read_text())
    examples = source_examples(data)
    store.write(
        "request.json",
        {
            "plan_sha256": file_sha(STUDY / "PLAN.md"),
            "dataset_sha256": file_sha(DATA),
            "lexical_sha256": LEXICAL_SHA,
            "source_examples": len(examples),
            "updates": 64,
            "seed": 47,
            "loss_weights": {
                "native_binary": 1.0,
                "representation": 0.1,
                "variance_floor": 0.01,
            },
            "ridge": 0.001,
        },
    )
    host, checkpoint = make_host(args)
    store.write("host.json", host.identity)
    store.write("native_parity.json", parity := host.native_parity())
    if (
        checkpoint["checkpoint_sha256"] != CHECKPOINT_SHA
        or not parity["exact"]
        or host.identity != json.loads((PRIOR / "host.json").read_text())
    ):
        raise ValueError("Pinned host/geometry mismatch")
    lexical, _, lexical_sha = load_lexical_interface(
        PRIOR / "lexical_bridge.npz", host.identity, device=host.device
    )
    if lexical_sha != LEXICAL_SHA:
        raise ValueError("Frozen lexical interface changed")
    before, lexical_before = parameter_sha(host.model), parameter_sha(lexical)
    sources, teachers, cache = {}, {}, {}
    # Complete every E capture and lexical teacher before constructing questions.
    for example in examples:
        state = host.encode(example["source"])
        sources[example["key"]] = state
        store.append(
            "sources.jsonl",
            example
            | {"state_sha256": state.sha(), "source_sha256": digest(example["source"])},
        )
    for group in data["groups"]:
        for pole, words in zip(("plus", "minus"), group["options"], strict=True):
            state, layout = capture_lexical(host, words)
            cache[group["id"], pole] = state
            store.append(
                "teachers.jsonl",
                {
                    "group": group["id"],
                    "pole": pole,
                    "state_sha256": state.sha(),
                    **layout,
                },
            )
    for example in examples:
        teachers[example["key"]] = cache[example["group"], example["pole"]]
    tasks = question_tasks(data, examples)
    adapter = RecipientAdapter(host, rank=16, last_layers=4, seed=47).requires_grad_(
        False
    )
    try:
        replay_previous(host, lexical, adapter, store)
        evaluate(
            host, adapter, lexical, None, tasks, sources, teachers, store, "controls"
        )
        training = [e for e in examples if e["split"] == "seen_sources"]
        source_tensor = torch.stack([sources[e["key"]].slots for e in training])
        teacher_tensor = torch.stack([teachers[e["key"]].slots for e in training])
        mapper, stats = fit_source_ridge(source_tensor, teacher_tensor)
        mapper = mapper.to(host.device)
        store.write(
            "ridge_fit.json",
            stats
            | {
                "source_keys": [e["key"] for e in training],
                "source_state_sha256": digest(
                    [sources[e["key"]].sha() for e in training]
                ),
                "teacher_state_sha256": digest(
                    [teachers[e["key"]].sha() for e in training]
                ),
            },
        )
        buffers = {k: v.clone() for k, v in mapper.named_buffers()}
        first = training[0]
        sample = [t for t in tasks if t["key"] == first["key"]][:2]
        ridge = save_checked(
            host,
            mapper,
            lexical,
            adapter,
            lexical_sha,
            seal,
            store,
            "ridge",
            sources[first["key"]].slots,
            sample,
        )
        evaluate(
            host, adapter, lexical, ridge, tasks, sources, teachers, store, "ridge"
        )
        units = sorted({(e["group"], e["phrasing"]) for e in training})
        generator = torch.Generator().manual_seed(47)
        indices = []
        while len(indices) < 64 * 4:
            indices.extend(torch.randperm(len(units), generator=generator).tolist())
        schedule = [
            [units[i] for i in indices[start : start + 4]] for start in range(0, 256, 4)
        ]
        store.write("schedule.json", schedule)
        optim = torch.optim.AdamW(mapper.parameters(), lr=0.0003, weight_decay=0.0)
        labels = [
            host.tokenizer.encode(x, add_special_tokens=False)[0] for x in ("A", "B")
        ]
        meter = host.meter.snapshot()
        for step, chosen in enumerate(schedule, 1):
            keys = [
                f"{group}-{pole}-p{phrasing}"
                for group, phrasing in chosen
                for pole in ("plus", "minus")
            ]
            by_key = {
                key: [t for t in tasks if t["key"] == key and t["wording"] == 0]
                for key in keys
            }
            panel = [t for key in keys for t in by_key[key]]
            if len(panel) != 16:
                raise ValueError("Training batch incomplete")
            optim.zero_grad(set_to_none=True)
            losses = []
            for start in range(0, 16, 4):
                batch = panel[start : start + 4]
                mapping = {
                    t["key"]: lexical(mapper(sources[t["key"]].slots.detach().clone()))
                    for t in batch
                }
                logits = batched_recipient_logits(
                    host, adapter, *batch_inputs(host, batch, mapping, "matched")
                )
                targets = torch.tensor(
                    [labels[t["gold"] == "B"] for t in batch], device=host.device
                )
                loss = training_loss(logits, targets, labels, "native_binary")
                (loss / 4).backward()
                losses.append(loss.item())
            predicted = mapper(torch.stack([sources[key].slots for key in keys]))
            target = torch.stack([teachers[key].slots for key in keys]).float()
            rep = (predicted.float() - target).square().mean() / stats[
                "teacher_variance"
            ]
            predicted_std = (predicted.float().var(0, correction=0) + 1e-8).sqrt()
            target_std = (target.var(0, correction=0) + 1e-8).sqrt()
            variance = (
                torch.relu(target_std - predicted_std).square().mean()
                / stats["teacher_variance"]
            )
            (0.1 * rep + 0.01 * variance).backward()
            if any(p.grad is not None for p in lexical.parameters()) or any(
                p.grad is not None for p in adapter.parameters()
            ):
                raise ValueError("Frozen interface received gradients")
            store.append(
                "training.jsonl",
                {
                    "step": step,
                    "source_keys": keys,
                    "native_losses": losses,
                    "representation_loss": rep.item(),
                    "variance_floor_loss": variance.item(),
                    "gradient_norm": update(mapper, optim, host),
                },
            )
        end = host.meter.snapshot()
        final = save_checked(
            host,
            mapper,
            lexical,
            adapter,
            lexical_sha,
            seal,
            store,
            "final",
            sources[first["key"]].slots,
            sample,
        )
        if any(
            not torch.equal(value, dict(mapper.named_buffers())[key])
            for key, value in buffers.items()
        ):
            raise ValueError("Fixed ridge buffers changed")
        evaluate(
            host, adapter, lexical, final, tasks, sources, teachers, store, "final"
        )
        store.write(
            "training_compute.json",
            {
                "forwards": end["forwards"] - meter["forwards"],
                "processed_tokens": end["processed_tokens"] - meter["processed_tokens"],
            },
        )
    finally:
        adapter.close()
    after, lexical_after = parameter_sha(host.model), parameter_sha(lexical)
    if before != after or lexical_before != lexical_after:
        raise ValueError("Frozen parameters changed")
    return {
        "status": "COMPLETED_SOURCE_LEXICAL_DEVELOPMENT",
        "updates": 64,
        "backbone_sha256_before": before,
        "backbone_sha256_after": after,
        "lexical_parameter_sha256_before": lexical_before,
        "lexical_parameter_sha256_after": lexical_after,
        "compute": host.meter.snapshot(),
        "G_trained": False,
        "qualification_content_loaded": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(
        backend="hf",
        cpu_threads=1,
        device="cuda:0",
        dtype="bfloat16",
        slots=8,
        cut=None,
        writer=None,
        max_context=1024,
        max_new_tokens=None,
        until_eos=True,
        transport=None,
        encoder_bridge=None,
        reviewed_local_code=None,
        wall_seconds=1800,
        profile=ROOT / "configs/models/gemma2b.yaml",
    )
    args = parser.parse_args()
    store, started = RunStore(args.out), time.monotonic()
    try:
        result = run(args, store)
    except Exception as exc:
        store.write(
            "terminal.json",
            {
                "status": "FAILED_NO_RETRY",
                "error": str(exc),
                "type": type(exc).__name__,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise
    store.write(
        "terminal.json", result | {"elapsed_seconds": time.monotonic() - started}
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
