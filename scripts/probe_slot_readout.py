"""Three sealed development arms for order learning and native slot readout."""

import argparse
import json
from pathlib import Path
import time

import torch
import numpy as np

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_continuity_recipient import (
    DATA,
    batch_inputs,
    model_scores,
    tasks_for_group,
    training_loss,
)
from probe_policy_form import material
from probe_recipient_trainability import CONFIG, TRAIN_GROUPS, rows
from selffold import prompts
from selffold.carrier_probe import capture_description
from selffold.cli import ROOT, make_host
from selffold.codec_learning import EncoderBridge, codec_loss
from selffold.continuity_data import load_groups
from selffold.recipient_learning import (
    RecipientAdapter,
    batched_recipient_logits,
    save_recipient,
    load_recipient,
)
from selffold.records import (
    RunStore,
    State,
    code_manifest,
    digest,
    file_sha,
    load_state,
)
from selffold.replay_probe import _capture, _slots, _logprobs

STUDY = ROOT / "provenance/slot_readout_20261103"
PREVIOUS = ROOT / "provenance/policy_form_20261103/run"
ORIGINAL = ROOT / "provenance/slot_content_20261101/trials/contrasts-gemma"
FIXTURE = ROOT / "data/slot_contrasts.dev.json"
ORDERS = {
    "forward": (0, 1, 2, 3),
    "reverse": (3, 2, 1, 0),
    "permutation_1": (1, 3, 0, 2),
    "permutation_2": (2, 0, 3, 1),
}
GAINS = (0, 0.5, 1, 2, 4, 8, 16)


def order_material(groups):
    base = material(groups)["forward"]
    result = {name: {} for name in (*ORDERS, "mixed")}
    for key, row in base.items():
        clauses = row["source"].split("When ")
        if len(clauses) != 5:
            raise ValueError("Expected four exhaustive clauses")
        for name, order in ORDERS.items():
            result[name][key] = {
                "source": clauses[0]
                + " ".join("When " + clauses[i + 1].strip() for i in order)
            }
        reverse_tag = key.endswith("-o1")
        # The native variant id includes its order as the final component.
        if not key.endswith(("-o0", "-o1")):
            raise ValueError("Unexpected native variant id")
        result["mixed"][key] = result["reverse" if reverse_tag else "forward"][key]
    return result


def optimizer(module):
    return torch.optim.AdamW(module.parameters(), lr=0.0003, weight_decay=0.0)


def update(module, optim, host):
    if any(p.grad is not None for p in host.model.parameters()):
        raise ValueError("Frozen base received gradients")
    if not any(p.grad is not None for p in module.parameters()):
        raise ValueError("No learning path")
    norm = torch.nn.utils.clip_grad_norm_(
        module.parameters(), 1.0, error_if_nonfinite=True
    ).item()
    optim.step()
    return norm


def save_task_adapter(host, adapter, store, name, seal, training, inputs):
    with torch.no_grad():
        before = batched_recipient_logits(host, adapter, *inputs).clone()
    path = store.path / (name + ".npz")
    sha = save_recipient(
        path,
        adapter,
        host.identity,
        dataset_sha256=file_sha(DATA / "manifest.json"),
        source_sha256=seal["source_sha256"],
        training=training,
    )
    param = parameter_sha(adapter)
    adapter.close()
    reloaded, _, actual = load_recipient(path, host)
    with torch.no_grad():
        after = batched_recipient_logits(host, reloaded, *inputs)
    if (
        sha != actual
        or parameter_sha(reloaded) != param
        or not torch.equal(before, after)
    ):
        raise ValueError("Saved recipient replay mismatch")
    store.append(
        "fits.jsonl",
        {
            "artifact": path.name,
            "sha256": sha,
            "parameter_sha256": param,
            "serialization_exact": True,
            **training,
        },
    )
    return reloaded


def order_arm(host, store, seal):
    train = {g["id"]: g for g in load_groups(DATA, "train")}
    dev = load_groups(DATA, "dev")
    groups = {
        "train": [train[k] for k in TRAIN_GROUPS],
        "previous_dev": dev[:2],
        "additional_dev": dev[2:4],
    }
    forms = order_material(groups)
    states, cache = {}, {}
    for form, panel in forms.items():
        states[form] = {}
        for key, row in panel.items():
            text = row["source"]
            if text not in cache:
                cache[text] = capture_description(host, text, "workspace")[0]
            state = cache[text]
            states[form][key] = state.slots
            store.append(
                "donors.jsonl",
                {
                    "form": form,
                    "key": key,
                    "source_sha256": digest(text),
                    "state_sha256": state.sha(),
                },
            )
    tasks = {
        f: {
            s: [t for g in gl for t in tasks_for_group(g, panel, "stage2", "self", 0)]
            for s, gl in groups.items()
        }
        for f, panel in forms.items()
    }
    schedule = json.loads(
        (ROOT / "provenance/continuity_20261103/trainability/schedule.json").read_text()
    )
    store.write("schedule.json", schedule)
    label_ids = [
        host.tokenizer.encode(x, add_special_tokens=False)[0] for x in ("A", "B")
    ]

    def evaluate(adapter, condition, endpoint):
        route = "literal_base" if condition == "literal" else "matched"
        for form in ORDERS:
            mapping = (
                states[form] if condition == "latent" else dict.fromkeys(states[form])
            )
            for split, panel in tasks[form].items():
                for start in range(0, len(panel), 8):
                    batch = panel[start : start + 8]
                    with torch.no_grad():
                        scored = model_scores(host, adapter, batch, mapping, route)
                    for task, score in zip(batch, scored, strict=True):
                        store.append(
                            "decisions.jsonl",
                            {
                                k: task[k]
                                for k in (
                                    "case",
                                    "group",
                                    "pole",
                                    "source_order",
                                    "world",
                                    "label_order",
                                    "gold",
                                )
                            }
                            | {
                                "condition": condition,
                                "endpoint": endpoint,
                                "form": form,
                                "split": split,
                                "source_sha256": digest(task["source"]),
                                "question_sha256": digest(task["question"]),
                                **score,
                            },
                        )

    for condition in ("literal", "latent"):
        old, _, _ = load_recipient(PREVIOUS / ("case_" + condition + ".npz"), host)
        try:
            evaluate(old, condition, "forward_fit")
        finally:
            old.close()
        adapter = RecipientAdapter(
            host, **{k: CONFIG[k] for k in ("rank", "last_layers", "seed", "scale")}
        )
        try:
            original = next(
                r for r in rows(PREVIOUS / "fits.jsonl") if r["condition"] == condition
            )
            initial = parameter_sha(adapter)
            if initial != original["initial_parameter_sha256"]:
                raise ValueError("Initial adapter differs")
            route = "literal_base" if condition == "literal" else "matched"
            mapping = (
                states["mixed"]
                if condition == "latent"
                else dict.fromkeys(states["mixed"])
            )
            optim = optimizer(adapter)
            meter = host.meter.snapshot()
            for step, group in enumerate(schedule, 1):
                panel = [t for t in tasks["mixed"]["train"] if t["group"] == group]
                if len(panel) != 32:
                    raise ValueError("Training group incomplete")
                optim.zero_grad(set_to_none=True)
                losses = []
                for start in range(0, 32, 8):
                    batch = panel[start : start + 8]
                    logits = batched_recipient_logits(
                        host, adapter, *batch_inputs(host, batch, mapping, route)
                    )
                    targets = torch.tensor(
                        [label_ids[t["gold"] == "B"] for t in batch], device=host.device
                    )
                    loss = training_loss(logits, targets, label_ids, "native_binary")
                    (loss / 4).backward()
                    losses.append(loss.item())
                store.append(
                    "training.jsonl",
                    {
                        "condition": condition,
                        "step": step,
                        "group": group,
                        "gradient_norm": update(adapter, optim, host),
                        "losses": losses,
                        "tasks_sha256": digest(panel),
                    },
                )
            training = {
                "condition": condition,
                "updates": 64,
                "form": "mixed",
                "initial_parameter_sha256": initial,
                "schedule_sha256": digest(schedule),
                "training_forwards": host.meter.forwards - meter["forwards"],
                "training_processed_tokens": host.meter.processed_tokens
                - meter["processed_tokens"],
            }
            first = tasks["mixed"]["train"][:8]
            adapter = save_task_adapter(
                host,
                adapter,
                store,
                "mixed_" + condition,
                seal,
                training,
                batch_inputs(host, first, mapping, route),
            )
            evaluate(adapter, condition, "mixed_fit")
        finally:
            adapter.close()
    return {"updates": 128, "unique_donors": len(cache)}


def description_rows(groups):
    return [
        {
            "case": f"{g['id']}-{pole}-p{p}",
            "group": g["id"],
            "pole": pole,
            "phrasing": p,
            "source": text,
        }
        for g in groups
        for pole in ("plus", "minus")
        for p, text in enumerate(g[pole])
    ]


def decoder_arm(host, store, seal):
    groups = json.loads(FIXTURE.read_text())["groups"]
    examples = description_rows(groups)
    cached = {}
    for row in examples:
        state = load_state(ORIGINAL / "states" / row["case"] / "E", device=host.device)
        if host.encode(row["source"]).sha() != state.sha():
            raise ValueError("Original E capture no longer exact")
        cached[row["case"]] = state
        store.append("donors.jsonl", row | {"state_sha256": state.sha()})
    for group in groups:
        for row in [r for r in examples if r["group"] == group["id"]]:
            for order, options in enumerate((group["options"], group["options"][::-1])):
                question = f"{group['question']}\nA = {options[0]}. B = {options[1]}. Use the supplied description."
                for control, state in (
                    ("original_text", State(text=row["source"])),
                    ("frozen_slots", cached[row["case"]]),
                ):
                    store.append(
                        "controls.jsonl",
                        {
                            "case": row["case"],
                            "order": order,
                            "control": control,
                            "gold": "A"
                            if (row["pole"] == "plus") == (order == 0)
                            else "B",
                            "scores": host.score_choices(state, question, ["A", "B"]),
                        },
                    )
    adapter = RecipientAdapter(
        host, rank=16, last_layers=host.depth - host.cut, seed=47
    )
    identity = torch.nn.Identity()

    def evaluate(endpoint):
        with adapter.active():
            zero = State(slots=torch.zeros_like(next(iter(cached.values())).slots))
            output = host.decode(zero)
            store.append(
                "generations.jsonl",
                {
                    "endpoint": endpoint,
                    "case": None,
                    "control": "zero",
                    **output.receipt(),
                },
            )
            for row in examples:
                opposite = f"{row['group']}-{'minus' if row['pole'] == 'plus' else 'plus'}-p{row['phrasing']}"
                losses = {}
                for control, state in (
                    ("matched", cached[row["case"]]),
                    ("opposed", cached[opposite]),
                    ("zero", zero),
                ):
                    losses[control] = codec_loss(
                        host, identity, state.slots, row["source"], end_token_id=107
                    ).item()
                store.append(
                    "nll.jsonl", {"endpoint": endpoint, "case": row["case"], **losses}
                )
                output = host.decode(cached[row["case"]])
                store.append(
                    "generations.jsonl",
                    row
                    | {"endpoint": endpoint, "control": "matched", **output.receipt()},
                )
        # The question scorer is the original base model, outside D's active scope.
        generated = {
            r["case"]: r["text"]
            for r in rows(store.path / "generations.jsonl")
            if r["endpoint"] == endpoint and r["control"] == "matched"
        }
        for group in groups:
            for row in [r for r in examples if r["group"] == group["id"]]:
                for order, options in enumerate(
                    (group["options"], group["options"][::-1])
                ):
                    question = f"{group['question']}\nA = {options[0]}. B = {options[1]}. Use the supplied description."
                    gold = "A" if (row["pole"] == "plus") == (order == 0) else "B"
                    text = generated[row["case"]]
                    scores = (
                        host.score_choices(State(text=text), question, ["A", "B"])
                        if text.strip()
                        else None
                    )
                    store.append(
                        "questions.jsonl",
                        {
                            "endpoint": endpoint,
                            "case": row["case"],
                            "group": group["id"],
                            "pole": row["pole"],
                            "phrasing": row["phrasing"],
                            "order": order,
                            "gold": gold,
                            "scores": scores,
                            "question_sha256": digest(question),
                        },
                    )

    try:
        initial = parameter_sha(adapter)
        evaluate("initial")
        optim = optimizer(adapter)
        generator = torch.Generator().manual_seed(47)
        schedule = []
        while len(schedule) < 512:
            schedule.extend(torch.randperm(24, generator=generator).tolist())
        schedule = [schedule[i : i + 8] for i in range(0, 512, 8)]
        store.write("schedule.json", schedule)
        for step, indices in enumerate(schedule, 1):
            optim.zero_grad(set_to_none=True)
            losses = []
            for index in indices:
                row = examples[index]
                with adapter.active():
                    loss = codec_loss(
                        host,
                        identity,
                        cached[row["case"]].slots,
                        row["source"],
                        end_token_id=107,
                    )
                    (loss / 8).backward()
                losses.append(loss.item())
            store.append(
                "training.jsonl",
                {
                    "step": step,
                    "cases": [examples[i]["case"] for i in indices],
                    "losses": losses,
                    "gradient_norm": update(adapter, optim, host),
                },
            )
        # Verify both teacher-forced and free generation paths after serialization.
        first = examples[0]
        with adapter.active():
            expected_nll = codec_loss(
                host,
                identity,
                cached[first["case"]].slots,
                first["source"],
                end_token_id=107,
            ).item()
        path = store.path / "decoder.npz"
        training = {
            "objective": "copy_source_plus_eos",
            "updates": 64,
            "examples": 24,
            "initial_parameter_sha256": initial,
            "plan_sha256": file_sha(STUDY / "PLAN.md"),
        }
        sha = save_recipient(
            path,
            adapter,
            host.identity,
            dataset_sha256=file_sha(FIXTURE),
            source_sha256=seal["source_sha256"],
            training=training,
        )
        final = parameter_sha(adapter)
        adapter.close()
        adapter, _, actual = load_recipient(path, host)
        with adapter.active():
            actual_nll = codec_loss(
                host,
                identity,
                cached[first["case"]].slots,
                first["source"],
                end_token_id=107,
            ).item()
        if (
            sha != actual
            or final != parameter_sha(adapter)
            or expected_nll != actual_nll
        ):
            raise ValueError("Decoder save/load mismatch")
        store.append(
            "fits.jsonl",
            training
            | {
                "artifact": path.name,
                "sha256": sha,
                "serialization_exact": True,
                "final_parameter_sha256": final,
                "active_layers": list(range(host.cut, host.depth)),
            },
        )
        evaluate("final")
    finally:
        adapter.close()
    return {
        "updates": 64,
        "examples": 24,
        "training_scope": "all_24_original_descriptions",
    }


def gain_slots(neutral, lexical, gain):
    return (neutral.float() + gain * (lexical.float() - neutral.float())).to(
        lexical.dtype
    )


def lexical_arm(host, store, seal):
    groups = json.loads(FIXTURE.read_text())["groups"]
    labels = {
        name: host.tokenizer.encode(name, add_special_tokens=False)[0]
        for name in ("A", "B")
    }
    tasks, states, neutral_states = [], {}, {}
    generator = torch.Generator().manual_seed(4711)
    direction = torch.nn.functional.normalize(
        torch.randn(host.slots, host.width, generator=generator), dim=-1
    ).to(host.device)

    def record(logits, meta, route, **extra):
        token = int(logits.argmax().item())
        store.append(
            "decisions.jsonl",
            meta
            | {
                "route": route,
                **_logprobs(logits, labels),
                "greedy_token": token,
                "greedy_label": next(
                    (k for k, v in labels.items() if v == token), None
                ),
                **extra,
            },
        )

    for gi, group in enumerate(groups):
        for order, options in enumerate((group["options"], group["options"][::-1])):
            question = f"{group['question']}\nA = {options[0]}. B = {options[1]}. Use the supplied description."
            prompt, markers = host.render(prompts.task_prompt(question))
            pos = markers[prompts.INPUT_MARKER]
            ids = prompt + [host.placeholder_id]
            native, full = _capture(host, ids, list(range(len(ids))))
            _, neutral = _capture(
                host,
                ids[: pos[-1] + 1] + [host.placeholder_id] * (len(ids) - pos[-1] - 1),
                pos,
            )
            if not torch.equal(native, _slots(host, ids, pos, neutral)):
                raise ValueError("Neutral native replay failed")
            for pole, words in zip(("plus", "minus"), group["options"], strict=True):
                tokens = host.tokenizer.encode(words, add_special_tokens=False)
                if not 1 <= len(tokens) <= 8:
                    raise ValueError("Lexical phrase exceeds eight slots")
                padded = tokens + [host.placeholder_id] * (8 - len(tokens))
                donor_ids = ids[: pos[0]] + padded + ids[pos[-1] + 1 :]
                reference, natural = _capture(host, donor_ids, list(range(len(ids))))
                _, lexical = _capture(
                    host,
                    donor_ids[: pos[-1] + 1]
                    + [host.placeholder_id] * (len(ids) - pos[-1] - 1),
                    pos,
                )
                if not torch.equal(lexical, natural[:, pos]):
                    raise ValueError(
                        "Opaque lexical capture differs from natural words"
                    )
                key = f"{group['id']}-{pole}-o{order}"
                gold = "A" if (pole == "plus") == (order == 0) else "B"
                meta = {
                    "group": group["id"],
                    "pole": pole,
                    "label_order": order,
                    "gold": gold,
                    "split": "train" if gi < 4 else "transfer",
                    "question_sha256": digest(question),
                }
                states[key], neutral_states[key] = lexical[0], neutral[0]
                tasks.append(meta | {"key": key, "source": words, "question": question})
                store.append(
                    "donors.jsonl",
                    meta
                    | {
                        "key": key,
                        "state_sha256": State(slots=lexical[0]).sha(),
                        "neutral_sha256": State(slots=neutral[0]).sha(),
                        "source_words": words,
                        "padded_source_token_ids": padded,
                        "recipient_ids_sha256": digest(ids),
                        "forward_length": len(ids),
                        "slot_positions": pos,
                        "neutral_native_exact": True,
                        "opaque_lexical_exact": True,
                    },
                )
                record(reference, meta, "native_words")
                for gain in GAINS:
                    value = gain_slots(neutral[0], lexical[0], gain)
                    record(
                        _slots(host, ids, pos, value[None]),
                        meta,
                        "gain",
                        gain=gain,
                        row_norms=value.float().norm(dim=-1).tolist(),
                    )
                difference = lexical[0].float() - neutral[0].float()
                random = (
                    neutral[0].float()
                    + 4 * direction * difference.norm(dim=-1, keepdim=True)
                ).to(lexical.dtype)
                for route, value in (
                    ("reverse_gain4", gain_slots(neutral[0], lexical[0].flip(0), 4)),
                    ("random_gain4", random),
                ):
                    record(_slots(host, ids, pos, value[None]), meta, route)
    bridge = EncoderBridge(host.width, 16).to(host.device)
    adapter = RecipientAdapter(host, rank=16, last_layers=4, seed=47)
    adapter.requires_grad_(False)
    try:
        optim = optimizer(bridge)
        panel = [t for t in tasks if t["split"] == "train"]
        if len(panel) != 16:
            raise ValueError("Wrong lexical training denominator")
        for step in range(1, 65):
            optim.zero_grad(set_to_none=True)
            losses = []
            for start in (0, 8):
                batch = panel[start : start + 8]
                transformed = {
                    t["key"]: bridge(states[t["key"]].detach().clone()) for t in batch
                }
                logits = batched_recipient_logits(
                    host, adapter, *batch_inputs(host, batch, transformed, "matched")
                )
                targets = torch.tensor(
                    [labels[t["gold"]] for t in batch], device=host.device
                )
                loss = training_loss(
                    logits, targets, list(labels.values()), "native_binary"
                )
                (loss / 2).backward()
                losses.append(loss.item())
            store.append(
                "training.jsonl",
                {
                    "step": step,
                    "gradient_norm": update(bridge, optim, host),
                    "losses": losses,
                },
            )
        with torch.no_grad():
            first = next(iter(states.values()))
            before = bridge(first).clone()
        path = store.path / "lexical_bridge.npz"
        metadata = {
            "kind": "lexical_slot_low_rank_residual",
            "base_host_sha256": digest(host.identity),
            "dataset_sha256": file_sha(FIXTURE),
            "source_sha256": seal["source_sha256"],
            "updates": 64,
            "plan_sha256": file_sha(STUDY / "PLAN.md"),
            "objective": "lexical_native_binary",
            "capture_geometry": "matched_recipient_length_lexical_slots",
        }
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                down=bridge.down.detach().cpu().numpy(),
                up=bridge.up.detach().cpu().numpy(),
                metadata_utf8=np.frombuffer(
                    json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8
                ),
            )
        sha = file_sha(path)
        restored = EncoderBridge(host.width, 16).to(host.device)
        with np.load(path, allow_pickle=False) as archive:
            if json.loads(archive["metadata_utf8"].tobytes()) != metadata:
                raise ValueError("Bridge metadata mismatch")
            with torch.no_grad():
                restored.down.copy_(torch.from_numpy(archive["down"]).to(host.device))
                restored.up.copy_(torch.from_numpy(archive["up"]).to(host.device))
        restored.requires_grad_(False)
        with torch.no_grad():
            if not torch.equal(before, restored(first)):
                raise ValueError("Bridge save/load mismatch")
        store.append(
            "fits.jsonl",
            {
                "artifact": path.name,
                "sha256": sha,
                "updates": 64,
                "serialization_exact": True,
            },
        )
        for route in ("bridge_matched", "bridge_opposed", "bridge_zero"):
            with torch.no_grad():
                mapping = {}
                for task in tasks:
                    key = task["key"]
                    source = states[key]
                    if route == "bridge_opposed":
                        opposite = (
                            key.replace("-plus-", "-minus-")
                            if task["pole"] == "plus"
                            else key.replace("-minus-", "-plus-")
                        )
                        source = states[opposite]
                    if route == "bridge_zero":
                        source = torch.zeros_like(source)
                    mapping[key] = restored(source)
                for task in tasks:
                    score = model_scores(host, adapter, [task], mapping, "matched")[0]
                    store.append(
                        "decisions.jsonl",
                        {
                            k: task[k]
                            for k in (
                                "group",
                                "pole",
                                "label_order",
                                "gold",
                                "split",
                                "question_sha256",
                            )
                        }
                        | {"route": route, **score},
                    )
    finally:
        adapter.close()
    return {"updates": 64, "lexical_donors": len(states)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("order", "decoder", "lexical"), required=True)
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
        seal = code_manifest(ROOT)
        store.write("source.json", seal)
        if seal["git_status"] or not seal["git_commit"]:
            raise ValueError("Clean committed source required")
        store.write(
            "request.json",
            {
                "arm": args.arm,
                "plan_sha256": file_sha(STUDY / "PLAN.md"),
                "fixture_sha256": file_sha(FIXTURE),
                "policy_manifest_sha256": file_sha(DATA / "manifest.json"),
            },
        )
        host, checkpoint = make_host(args)
        store.write("host.json", host.identity)
        store.write("native_parity.json", parity := host.native_parity())
        if (
            checkpoint["checkpoint_sha256"] != CHECKPOINT_SHA
            or not parity["exact"]
            or host.identity
            != json.loads((ORIGINAL / "host_identity.json").read_text())
        ):
            raise ValueError("Pinned host or native parity mismatch")
        before = parameter_sha(host.model)
        result = {"order": order_arm, "decoder": decoder_arm, "lexical": lexical_arm}[
            args.arm
        ](host, store, seal)
        after = parameter_sha(host.model)
        if before != after:
            raise ValueError("Backbone changed")
        result.update(
            status="COMPLETED_BOUNDED_READOUT",
            arm=args.arm,
            backbone_sha256_before=before,
            backbone_sha256_after=after,
            G_trained=False,
            qualification_content_loaded=False,
            compute=host.meter.snapshot(),
        )
    except Exception as exc:
        result = {
            "status": "FAILED_NO_RETRY",
            "arm": args.arm,
            "error": str(exc),
            "type": type(exc).__name__,
        }
        store.write(
            "terminal.json", result | {"elapsed_seconds": time.monotonic() - started}
        )
        raise
    store.write(
        "terminal.json", result | {"elapsed_seconds": time.monotonic() - started}
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
