"""On-policy realized-state writer: the registered S-WRITE conditions.

The writer prompt renders the controller's own realized transition —
(actual prior table, applied instruction, realized post-update table) —
generated lazily inside the rollout loop, memoized per realized triple.
The decoder asserts ``off/off -> member1`` with NO role mapping handed in:
member1 is identified as the action the protocol-fixed initial table
(mask 8, layout [m0,m0,m0,m1]) assigns to the on/on row.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .lm_writer import assignment_words, decode_state_slot
from .mixture_boundary import eval_step, init_weights, realize
from .role_memory import truth_bits
from .roundtrip import pattern_of

WORLD = ["off/off", "off/on", "on/off", "on/on"]
INITIAL_PRIOR = 8  # member layout [m0,m0,m0,m1]: on/on row names member1
RW_STYLES = ("rw_directed", "rw_natural", "rw_bare")


def render_table(options, pole, pattern):
    actions = assignment_words(options, pole, int(pattern))
    return "; ".join(f"{w} -> {a}" for w, a in zip(WORLD, actions))


def rw_prompt(style, options, pole, prior_pattern, mask, out_pattern):
    """Prompt over the realized transition; rw_bare drops the prior table."""
    base = "You are the self-report module of a small state machine.\n"
    if style != "rw_bare":
        base += (f"The table previously read: "
                 f"{render_table(options, pole, prior_pattern)}.\n")
    base += (
        f"An instruction was applied with these assignments: "
        f"{render_table(options, pole, mask)}.\n"
        f"The table now reads: {render_table(options, pole, out_pattern)}.\n"
    )
    if style == "rw_directed":
        return base + (
            "Question: which action does the off/off row now select? "
            "Answer with the action name only.")
    return base + "Task: write one short log sentence describing what changed."


def member1_name(options, pole):
    """member1 = the action the fixed INITIAL table assigns to the on/on row."""
    return assignment_words(options, pole, INITIAL_PRIOR)[3]


def asserts_member1(text, member1, other):
    """Pole-free registered decode: does the report bind off/off to member1?

    The decoder receives the two action names only — the role mapping is
    never handed in. member1's name is a protocol constant recovered from
    the initial rendered table, not from `pole`.
    """
    slot, miss = decode_state_slot(text, [other, member1])
    return (0 if miss else int(slot == 1)), miss


class ReportBank:
    """Lazy on-policy generation, memoized per (pair, prior, mask, out).

    Writes each generated record to reports_<style>.jsonl immediately, so
    interrupted runs resume from the file. Generation is deterministic
    (kv_capped_argmax); a memoized lookup is the same object an on-policy
    request would produce.
    """

    def __init__(self, host, style, pairs, groups, outdir, generate=None):
        if style not in RW_STYLES:
            raise ValueError(f"Unknown realized-writer style: {style}")
        self.host, self.style = host, style
        self.host_generate = generate or host.generate
        self.pairs, self.groups = pairs, groups
        self.path = Path(outdir) / f"reports_{style}.jsonl"
        self.cache, self.generated = {}, 0
        if self.path.exists():
            for line in self.path.open():
                r = json.loads(line)
                self.cache[(r["pair"], r["prior"], r["mask"], r["out"])] = r

    def bit(self, pair, prior, mask, out):
        key = (int(pair), int(prior), int(mask), int(out))
        r = self.cache.get(key)
        if r is None:
            gi, pole = self.pairs[pair]
            options = self.groups[gi]["options"]
            prompt = rw_prompt(self.style, options, pole, prior, mask, out)
            res = self.host_generate(prompt)
            m1 = member1_name(options, pole)
            other = [o for o in options if o != m1][0]
            bit, miss = asserts_member1(res.text, m1, other)
            receipt = dict(res.receipt())
            receipt.pop("text", None)
            r = dict(pair=key[0],
                     group=int(gi), prior=key[1], mask=key[2], out=key[3],
                     style=self.style, text=res.text, generation=receipt,
                     bit=bit, missing=miss, decoder="member1_v1")
            with self.path.open("a") as fh:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            self.cache[key] = r
            self.generated += 1
        return r["bit"]

    def matrix(self, n_pairs):
        bits = torch.full((n_pairs, 16, 16, 16), -1, dtype=torch.long)
        for (p, prior, m, o), r in self.cache.items():
            bits[p, prior, m, o] = r["bit"]
        return bits


@torch.no_grad()
def rollout_realized(model, form, members, keys, masks, report, pair_of):
    """rollout_rt with the writer in the loop.

    `report(pair, prior_pattern, cur_mask, out_pattern) -> int` is called
    for every realized transition the controller actually reaches — the
    on-policy request surface. Returns the same endpoint dict layout as
    rollout_rt (depths/second/fourth are member-space w[...,1]).
    """
    from .role_trials import donor_indices

    n = len(members)
    paths = torch.tensor(list(itertools.product(range(len(masks)), repeat=4)))
    num = len(paths)
    instructions = torch.tensor(masks)[paths].repeat(n, 1)
    donors = donor_indices(keys)
    carried = members.repeat_interleave(num, 0)
    weights = init_weights(n * num, 1.0)
    table = realize(weights, carried)
    message = torch.zeros(n * num, dtype=torch.long)
    case_of = torch.arange(n * num) // num

    def rt_step(table, weights, message, cur, cases, mem):
        prior_p = pattern_of(weights)
        table, w2, _, _ = eval_step(
            model, table, weights, message, cur, form, "lm", mem,
            lm_bits=torch.zeros(len(table), 1, dtype=torch.long))
        out_p = pattern_of(w2)
        written = torch.tensor(
            [report(pair_of[int(c)], int(p), int(m), int(o))
             for c, p, m, o in zip(cases, prior_p, cur, out_p)],
            dtype=torch.long)
        return table, w2, written

    table, weights, w0 = rt_step(table, weights, message, instructions[:, 0],
                                 case_of, carried)
    depths = [weights[..., 1].reshape(n, num, 4)]
    second, fourth, written, mismatch = [], [], [w0.reshape(n, num, -1)], []
    for route in ("matched", "opposed", "equivalent", "erase"):
        ids = donors[{"opposed": 1, "equivalent": 2}.get(route, 0)]
        w2 = weights.reshape(n, num, 4, 2)[ids].reshape(-1, 4, 2)
        start_members = carried.reshape(n, num, 2, 16)[ids].reshape(-1, 2, 16)
        msg = w0.reshape(n, num)[ids].reshape(-1)
        if route == "erase":
            msg = torch.zeros_like(msg)
        start_table = realize(w2, start_members)
        ws = []
        route_cases = torch.as_tensor(ids).repeat_interleave(num)
        for t in range(1, 4):
            if route == "matched":
                mismatch.append(
                    (pattern_of(w2) != instructions[:, t - 1]).reshape(n, num))
            start_table, w2, wt = rt_step(
                start_table, w2, msg, instructions[:, t], route_cases, start_members)
            msg = wt
            ws.append(w2)
            if route == "matched" and t < 3:
                written.append(wt.reshape(n, num, -1))
        if route == "matched":
            depths += [w[..., 1].reshape(n, num, 4) for w in ws]
        second.append(ws[0][..., 1].reshape(n, num, 4))
        fourth.append(ws[2][..., 1].reshape(n, num, 4))
    return dict(
        paths=paths.numpy(),
        depths=np.stack([d.numpy() for d in depths], 2).astype(np.float16),
        second=np.stack([s.numpy() for s in second], 1).astype(np.float16),
        fourth=np.stack([s.numpy() for s in fourth], 1).astype(np.float16),
        routes=np.asarray(["matched", "opposed", "equivalent", "erase"]),
        written=torch.stack(written, 2).numpy().astype(np.uint8),
        written_semantics=np.asarray("rt_member1_bits_v1"),
        prior_mismatch=np.stack([m.numpy() for m in mismatch], 1),
    )
