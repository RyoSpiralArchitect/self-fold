"""One language insertion at a four-row midpoint and its carried role bit."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import itertools
import re

import numpy as np
import torch
import torch.nn.functional as F

from .records import digest
from .role_memory import RoleMemory

COORDS = ("off/off", "off/on", "on/off", "on/on")
FIELDS = (*COORDS, "default")
FITS = "provenance/role_memory_20261104/run/produce/fits.npz"


@dataclass(frozen=True)
class Midpoint:
    rows: tuple[int, ...]
    bit: int

    def __post_init__(self):
        if len(self.rows) != 4 or self.bit not in (0, 1):
            raise ValueError("Expected four rows and one binary role")
        if any(x not in range(16) for x in self.rows):
            raise ValueError("Action outside the fixed vocabulary")

    def default(self):
        return self.rows[0] ^ self.bit

    def fields(self):
        return (*self.rows, self.default())


def controller(root, seed):
    actor = RoleMemory(seed)
    with np.load(root / FITS, allow_pickle=False) as f:
        actor.load_state_dict({k: torch.from_numpy(f[f"reward_{seed}_{k}"]) for k in actor.state_dict()})
    return actor.eval().requires_grad_(False)


@torch.no_grad()
def first_update(actor, menu, pattern, default):
    """Run the learned G1. The input table is the same starting layout used in §7."""
    pair = [2 * menu + default, 2 * menu + (1 - default)]
    initial = F.one_hot(torch.tensor([[pair[0]] * 3 + [pair[1]]]), 16).float()
    mask = pattern ^ (15 if default else 0)
    updated, bit, *_ = actor.step(initial, torch.zeros(1, dtype=torch.long), torch.tensor([mask]))
    result = Midpoint(tuple(updated[0].argmax(-1).tolist()), int(bit.item()))
    expected = tuple(2 * menu + ((pattern >> i) & 1) for i in range(4))
    if result.rows != expected or result.default() != pair[0]:
        raise ValueError("Frozen first edit did not realize the registered midpoint")
    return result


def report_text(values):
    return "\n".join(f"{key} = {value}" for key, value in zip(FIELDS, values, strict=True))


def writer_prompt(state, rotation, names):
    order = [(rotation + i) % 4 for i in range(4)]
    rows = "\n".join(f"{COORDS[i]} : {names[state.rows[i]]}" for i in order)
    role = "default" if state.bit == 0 else "alternative"
    return (f"The current action record is printed below. Slots are named left/right.\n{rows}\n"
            f"The action currently at off/off has the {role} role. There are two actions: "
            "one is the default and the other is the alternative.\n"
            "Write a reusable five-line report. List the complete action name for off/off, "
            "off/on, on/off, on/on, then the default action. Use the labels off/off, off/on, "
            "on/off, on/on, default, each followed by = and the action name. No commentary.")


def panel(actor, cfg, names, split):
    states = []
    for menu, pattern, default in itertools.product(cfg["menus"], cfg[split + "_patterns"], (0, 1)):
        state = first_update(actor, menu, pattern, default)
        for rot in range(4):
            states.append(dict(id=f"{menu}/{pattern}/{default}/{rot}", menu=menu, pattern=pattern,
                               default=default, rotation=rot, rows=list(state.rows), bit=state.bit,
                               z_sha256=digest([state.rows, state.bit]),
                               prompt=writer_prompt(state, rot, names)))
    return states


def read_fields(text, names):
    """Read labeled lines using only the emitted text and the fixed action dictionary."""
    hits = defaultdict(list)
    pattern = r"^\s*(?:[-*#>]+\s*|\d+[.)]\s*)*(off/off|off/on|on/off|on/on|default(?: action)?)\s*[=:–—-]\s*(.+)$"
    for line in text.casefold().splitlines():
        line = line.replace("**", "").replace("`", "").strip().strip("|").strip()
        match = re.match(pattern, line)
        if match:
            field, value = match.groups()
            field = "default" if field.startswith("default") else field
            mentions = [i for i, name in enumerate(names)
                        if re.search(r"(?<!\w)" + re.escape(name.casefold()) + r"(?!\w)", value)]
            hits[field].append(mentions[0] if len(mentions) == 1 else None)
    return [values[0] if (values := hits[field]) and len(set(values)) == 1 else None
            for field in FIELDS]


def reencode(values):
    if len(values) != 5 or any(x is None for x in values):
        return None
    if len({i // 2 for i in values}) != 1:
        return None
    return Midpoint(tuple(values[:4]), int(values[0] != values[4]))


@torch.no_grad()
def final_choices(actor, state, menu, default, masks, native):
    """Run frozen G2 on reconstructed rows and bit; the final Gemma sees a row and its question."""
    total = len(masks) * 8
    if state is None:
        return dict(correct=0, total=total, complete=False, output_rows=None)
    table = F.one_hot(torch.tensor(state.rows), 16).float().expand(len(masks), -1, -1)
    bits = torch.full((len(masks),), state.bit, dtype=torch.long)
    updated, *_ = actor.step(table, bits, torch.tensor(masks))
    out = updated.argmax(-1).numpy()
    gold = default ^ ((np.asarray(masks)[:, None] >> np.arange(4)) & 1)
    expected_letter = gold[:, :, None] ^ np.arange(2)
    tops = native["top"][menu, out]
    ab = native["ab"][menu, out]
    margin = (ab[..., 0] - ab[..., 1]) * (1 - 2 * expected_letter)
    correct = (tops == native["labels"][expected_letter]) & (margin > 0)
    return dict(correct=int(correct.sum()), total=total, complete=bool(correct.all()),
                output_rows=out.tolist())


def evaluate(actor, cfg, names, queries, records, native):
    expected = {(q["id"], arm, mode) for q in queries for arm in ("before", "after")
                for mode in ("free", "choice")}
    indexed = {(r["id"], r["arm"], r["mode"]): r for r in records}
    if len(indexed) != len(records) or set(indexed) != expected:
        raise ValueError("Reports do not cover exactly the registered arms and presentations")
    scored = []
    for q in queries:
        z = Midpoint(tuple(q["rows"]), q["bit"])
        gold = list(z.fields())
        oracle = read_fields(report_text([names[i] for i in gold]), names)
        routes = {"L": (z, gold), "oracle": (reencode(oracle), oracle)}
        for arm, mode in itertools.product(("before", "after"), ("free", "choice")):
            r = indexed[q["id"], arm, mode]
            if r["prompt_sha256"] != digest(q["prompt"]):
                raise ValueError("Response bound to the wrong prompt")
            values = read_fields(r["text"], names)
            routes[f"{arm}_{mode}"] = (reencode(values), values)
            if mode == "free":
                for suffix, patched in (("oracle_rows", gold[:4] + values[4:]),
                                        ("oracle_default", values[:4] + gold[4:])):
                    routes[f"{arm}_{suffix}"] = (reencode(patched), patched)
        for route, (state, values) in routes.items():
            outcome = final_choices(actor, state, q["menu"], q["default"], cfg["next_masks"], native)
            scored.append(dict(id=q["id"], menu=q["menu"], pattern=q["pattern"], default=q["default"],
                               rotation=q["rotation"], route=route, fields=values,
                               field_correct=[a == b for a, b in zip(values, gold, strict=True)],
                               encodable=state is not None, **outcome))
    return scored


def paired_change(a, b):
    if a.keys() != b.keys():
        raise ValueError("Unmatched presentations")
    return dict(n=len(a), recovered=sum(b[k] and not a[k] for k in a),
                lost=sum(a[k] and not b[k] for k in a))


def summarize(scored, menus):
    routes = defaultdict(list)
    for r in scored:
        routes[r["route"]].append(r)
    twins, summary = {}, {}
    for name, rows in routes.items():
        groups = defaultdict(list)
        for r in rows:
            groups[r["menu"], r["pattern"], r["rotation"]].append(r)
        if any(sorted(r["default"] for r in pair) != [0, 1] for pair in groups.values()):
            raise ValueError("Both role donors are required in every pair")
        twins[name] = {k: all(r["complete"] for r in pair) for k, pair in groups.items()}
        summary[name] = dict(reports=len(rows), encodable=sum(r["encodable"] for r in rows),
                             individual_rows=sum(sum(r["field_correct"][:4]) for r in rows),
                             four_rows=sum(all(r["field_correct"][:4]) for r in rows),
                             default=sum(r["field_correct"][4] for r in rows),
                             all_fields=sum(all(r["field_correct"]) for r in rows),
                             complete=sum(r["complete"] for r in rows),
                             judgments=sum(r["correct"] for r in rows),
                             judgments_total=sum(r["total"] for r in rows),
                             twins=sum(twins[name].values()), twin_total=len(groups))
    changes = {}
    for a, b in (("before_free", "after_free"), ("before_choice", "after_choice"),
                 ("L", "after_free"), ("after_free", "oracle")):
        old, new = twins[a], twins[b]
        by_menu = [float(np.mean([int(new[k]) - int(old[k]) for k in old if k[0] == m])) for m in menus]
        means = [np.mean(sample) for sample in itertools.product(by_menu, repeat=len(menus))]
        change = paired_change(old, new)
        change.update(by_menu=by_menu, delta=float(np.mean(by_menu)),
                      menu_block_interval=np.quantile(means, [0.025, 0.975]).tolist())
        for label, getter in (("four_rows", lambda r: all(r["field_correct"][:4])),
                              ("default", lambda r: r["field_correct"][4]),
                              ("complete", lambda r: r["complete"])):
            change[label] = paired_change({r["id"]: getter(r) for r in routes[a]},
                                          {r["id"]: getter(r) for r in routes[b]})
        changes[f"{b}_minus_{a}"] = change
    gain = changes["after_free_minus_before_free"]
    anchors = all(summary[r]["twins"] / summary[r]["twin_total"] >= .95 for r in ("L", "oracle"))
    delta_l = changes["after_free_minus_L"]["menu_block_interval"]
    return dict(arms=summary, changes=changes, decisions=dict(
        anchors=anchors, training_gain=anchors and gain["delta"] >= .10
        and sum(d > 0 for d in gain["by_menu"]) >= 3,
        beats_L=anchors and delta_l[0] > 0,
        within_five_points_of_L=anchors and delta_l[0] >= -.05 and delta_l[1] <= .05))
