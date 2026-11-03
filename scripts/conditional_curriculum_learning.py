"""Scoped literal, frozen-source and joint-source phases for the conditional curriculum."""

import torch

from fit_continuity_recipient import batch_inputs
from fit_joint_continuity import gradient_norm
from selffold import prompts
from selffold.conditional_curriculum import tasks_for
from selffold.recipient_learning import (
    batched_recipient_logits,
    differentiable_source_slots,
)
from selffold.records import digest


def native_source_slots(host, encoder, text):
    """Retain the source gradient using the same native input length as ordinary E."""
    ids, _ = host.render(prompts.encode_prompt(text))
    return differentiable_source_slots(host, encoder, text, forward_length=len(ids))


def weighted_native_loss(logits, targets, worlds):
    """One microbatch's contribution to a four-source, eight-question-per-source step."""
    if (
        logits.ndim != 2
        or targets.shape != (len(logits),)
        or len(worlds) != len(logits)
    ):
        raise ValueError("Need one target and world index per native score")
    if any(world not in range(4) for world in worlds):
        raise ValueError("Unexpected conditional world")
    weights = logits.new_tensor([3.0 if world == 3 else 1.0 for world in worlds])
    losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    # Four sources each contribute six false-world labels and two true-world labels.
    return (weights * losses).sum() / 48.0


def train_phase(
    host,
    task_adapter,
    encoder_or_None,
    cases,
    groups,
    fixture,
    stage_schedule,
    store,
    phase,
    raw_cache=None,
    literal=False,
):
    """Mutate the declared adapters at one fixed endpoint, retaining every training step."""
    if torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
        raise ValueError("Curriculum training requires an enabled gradient path")
    if (
        task_adapter.host is not host
        or task_adapter._closed
        or task_adapter.scope != "task"
    ):
        raise ValueError("Need the active host's task adapter")
    encoder = encoder_or_None
    if literal and encoder is not None:
        raise ValueError("Literal warmup cannot run a source adapter")
    if encoder is not None and (
        encoder.host is not host
        or encoder._closed
        or encoder.scope != "source_encoding"
    ):
        raise ValueError("Need the active host's source-encoding adapter")
    if not literal and encoder is None and raw_cache is None:
        raise ValueError("Frozen-source training needs its captured state bank")
    records = list(cases.values()) if isinstance(cases, dict) else list(cases)
    by_case = {
        (
            c["group"],
            c["stage"],
            c["flag_pair"],
            c["style"],
            c["source_order"],
            c["pole"],
        ): c
        for c in records
        if c["split"] == "fit"
    }
    group_map = groups if isinstance(groups, dict) else {g["id"]: g for g in groups}
    parameters = [*task_adapter.parameters()]
    if encoder is not None:
        parameters.extend(encoder.parameters())
    if not parameters or not all(p.requires_grad for p in parameters):
        raise ValueError("Every selected adapter parameter must be trainable")
    if any(p.requires_grad for p in host.model.parameters()):
        raise ValueError("The backbone must stay frozen")
    label_ids = {
        label: host.tokenizer.encode(label, add_special_tokens=False)
        for label in ("A", "B")
    }
    if any(len(v) != 1 for v in label_ids.values()) or label_ids["A"] == label_ids["B"]:
        raise ValueError("Need distinct single-token native A/B labels")
    optimizer = torch.optim.AdamW(parameters, lr=0.0003, weight_decay=0.0)
    for step, units in enumerate(stage_schedule, 1):
        if len(units) != 2 or any(len(unit) != 5 for unit in units):
            raise ValueError(
                "Each curriculum step requires two five-field source units"
            )
        chosen = [by_case[(*unit, pole)] for unit in units for pole in (0, 1)]
        if len({c["key"] for c in chosen}) != 4:
            raise ValueError("A curriculum step must contain four distinct sources")
        optimizer.zero_grad(set_to_none=True)
        losses, task_bindings, source_layouts = [], [], []
        meter = host.meter.snapshot()
        for case in chosen:
            tasks = [
                task
                for world in range(4)
                for task in tasks_for(case, group_map[case["group"]], fixture, world)
            ]
            if len(tasks) != 8:
                raise ValueError(
                    "Each source must cover four worlds in both label orders"
                )
            task_bindings.extend(
                {key: value for key, value in task.items() if key != "source"}
                for task in tasks
            )
            for start in (0, 4):
                batch = tasks[start : start + 4]
                supplied = None
                if encoder is not None:
                    supplied, layout = native_source_slots(
                        host, encoder, case["source"]
                    )
                    if not supplied.requires_grad:
                        raise ValueError(
                            "The source adapter learning path was detached"
                        )
                    source_layouts.append(
                        {
                            "key": case["key"],
                            "forward_length": layout["forward_length"],
                            "positions": layout["capture_positions"],
                        }
                    )
                elif not literal:
                    supplied = raw_cache[case["key"]]
                    if supplied.requires_grad:
                        raise ValueError("The frozen source bank must be detached")
                state_map = {case["key"]: supplied}
                inputs = batch_inputs(
                    host, batch, state_map, "literal_base" if literal else "matched"
                )
                logits = batched_recipient_logits(host, task_adapter, *inputs)
                target = torch.tensor(
                    [label_ids[task["gold"]][0] for task in batch],
                    dtype=torch.long,
                    device=host.device,
                )
                loss = weighted_native_loss(logits, target, [t["world"] for t in batch])
                loss.backward()
                losses.append(loss.item())
        if any(p.grad is not None for p in host.model.parameters()):
            raise ValueError("The frozen backbone received gradients")
        task_norm = gradient_norm(task_adapter)
        source_norm = gradient_norm(encoder) if encoder is not None else None
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters, 1.0, error_if_nonfinite=True
        ).item()
        optimizer.step()
        after = host.meter.snapshot()
        store.append(
            "training.jsonl",
            {
                "phase": phase,
                "step": step,
                "units": units,
                "source_keys": [c["key"] for c in chosen],
                "weighted_loss": sum(losses),
                "micro_losses": losses,
                "task_gradient_norm": task_norm,
                "source_gradient_norm": source_norm,
                "gradient_norm": total_norm,
                "tasks_sha256": digest(task_bindings),
                "source_layouts": source_layouts,
                "forwards": after["forwards"] - meter["forwards"],
                "processed_tokens": after["processed_tokens"]
                - meter["processed_tokens"],
            },
        )
