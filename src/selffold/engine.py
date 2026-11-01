"""T/L/B matched recurrence with cloned readouts, frozen operator schedules and receipts."""
from __future__ import annotations
from copy import deepcopy
from typing import Callable
from .records import State
from .rng import isolated_rng
from .metrics import transition
from .prompts import operation


def observe(host, state: State) -> dict:
    before = state.sha()
    with isolated_rng():
        output = None if state.text is not None else host.decode(state.clone())
    if state.sha() != before:
        raise RuntimeError("Observer mutated main state")
    if output is None:
        return {"text": state.text, "stop": "state_text", "censored": False}
    return output.receipt()


def step(host, state: State, arm: str, op: str):
    prior_hash = state.sha()
    if arm == "T":
        if state.text is None:
            raise TypeError("T requires text")
        out = host.update_text(state.text, op)
        new = out.as_state()
        action = {"text_update": out.receipt()}
        trace = deepcopy(host.last_trace)
        activations = {k: v.clone() for k, v in host.last_activations.items()}
    elif arm in {"L", "B"}:
        if state.slots is None:
            raise TypeError("L/B require tensor state")
        updated = host.update_latent(state.clone(), op)
        trace = deepcopy(host.last_trace)
        activations = {k: v.clone() for k, v in host.last_activations.items()}
        action = {"updated_sha256_before_roundtrip": updated.sha()}
        if arm == "B":
            readout = host.decode(updated.clone())
            action["roundtrip_text"] = readout.receipt()
            new = host.encode(readout.as_state().text)
        else:
            new = updated
    else:
        raise ValueError("arm must be T, L, or B")
    if state.sha() != prior_hash:
        raise RuntimeError("Update mutated its input state")
    return new, action, trace, activations


def run_trajectory(host, seed_text: str, *, arm: str, regime="plain", steps=2,
                   initial: State | None = None, observer=True,
                   sink: Callable | None = None):
    if steps < 0:
        raise ValueError("negative steps")
    state = initial.clone() if initial is not None else (State(text=seed_text) if arm == "T" else host.encode(seed_text))
    history = [state.clone()]
    # History is experimenter-side only; it is NEVER passed to step or to host prompts.
    rows, seen = [], {state.sha(): 0}
    for n in range(steps):
        op = operation(n, regime)
        before = host.meter.snapshot()
        state, action, trace, activations = step(host, state, arm, op)
        after_main = host.meter.snapshot()
        row = {"step": n+1, "arm": arm, "regime": regime, "operator": op,
               "state_sha256": state.sha(), "transition": transition(history[-1], state),
               "exact_return_to_step": seen.get(state.sha()), "action": action,
               "activations": trace, "main_compute": {k: after_main[k]-before[k] for k in before}}
        if observer:
            # T already holds the generated text. Carry its original stop reason into review.
            row["observation"] = action["text_update"].copy() if arm == "T" else observe(host, state)
        after = host.meter.snapshot()
        row["observer_compute"] = {k: after[k]-after_main[k] for k in after}
        seen.setdefault(state.sha(), n+1)
        rows.append(row)
        history.append(state.clone())
        if sink:
            sink(row, state.clone(), activations)
    return history, rows


def branch_once(host, state: State, *, continuation_steps=2, start_step=0, regime="plain"):
    """Same-state one-time E(D(.)) intervention. Restore RNG before each branch."""
    if state.slots is None:
        raise TypeError("Branch intervention requires latent state")
    original = state.sha()
    branches = {}
    for kind in ("identity", "language_roundtrip"):
        with isolated_rng():
            s = state.clone()
            intervention = None
            if kind == "language_roundtrip":
                output = host.decode(s.clone())
                intervention = output.receipt()
                s = host.encode(output.as_state().text)
            states = [s.clone()]
            for i in range(continuation_steps):
                s, _, _, _ = step(host, s, "L", operation(start_step+i, regime))
                states.append(s.clone())
            branches[kind] = {"states": states, "readout": intervention}
    if state.sha() != original:
        raise RuntimeError("Branch altered parent")
    return branches


def task_readout(host, state: State, task: dict) -> dict:
    original = state.sha()
    with isolated_rng():
        scores = host.score_choices(state.clone(), task["question"], task["labels"])
    if state.sha() != original:
        raise RuntimeError("Task readout altered recurrence state")
    order = sorted(scores, key=lambda x: (-scores[x]["logprob"], x))
    tied = len(order) > 1 and scores[order[0]]["logprob"] == scores[order[1]]["logprob"]
    return {"task_id": task["id"], "group": task["group"], "scores": scores,
            "prediction": None if tied else order[0], "tie": tied,
            "correct": False if tied else order[0] == task["answer"],
            "state_sha256": original}
