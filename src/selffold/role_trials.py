"""Paired multi-edit evaluations. Compression happens only after the soft tensors are constructed."""

from itertools import product

import numpy as np
import torch

from .role_memory import edit_paths, final_edit, native_row_indices, truth_bits

ROUTES = ("matched", "opposed", "equivalent", "erase", "flip", "omit")


def donor_indices(keys):
    lookup = {tuple(k): i for i, k in enumerate(keys)}
    opposed = [lookup[(g, f, s, 1 - p, o)] for g, f, s, p, o in keys]
    equivalent = [lookup[(g, 1 - f, s, p, 1 - o)] for g, f, s, p, o in keys]
    return np.array([np.arange(len(keys)), opposed, equivalent], dtype=np.int32)


@torch.no_grad()
def rollout(model, arm, rows, keys, masks):
    n = len(rows)
    paths = torch.tensor(edit_paths(masks))
    num = len(paths)
    instructions = paths.repeat(n, 1)
    donors = donor_indices(keys)
    origin = rows.repeat_interleave(num, 0)
    init = origin[:, [0, 0, 0, 1]]
    empty = torch.zeros(len(init), dtype=torch.long)
    table1, bit1 = final_edit(
        model, init, empty, instructions[:, 0], arm == "table_only"
    )
    first = native_row_indices(table1, origin).reshape(n, num, 4)[:, :: len(masks)]
    second, fourth = [], []
    for route in ROUTES:
        ids = donors[{"opposed": 1, "equivalent": 2}.get(route, 0)]
        start = table1.reshape(n, num, 4, 16)[ids].reshape(-1, 4, 16)
        bit = bit1.reshape(n, num)[ids].reshape(-1)
        if route == "erase":
            bit = torch.zeros_like(bit)
        elif route == "flip":
            bit = 1 - bit
        if arm == "table_only":
            bit = torch.zeros_like(bit)
        ref = rows[ids].repeat_interleave(num, 0)
        table2, bit2 = final_edit(
            model, start, bit, instructions[:, 1], arm == "table_only"
        )
        table3, bit3 = final_edit(
            model, table2, bit2, instructions[:, 2], arm == "table_only"
        )
        table4, _ = final_edit(
            model, table3, bit3, instructions[:, 3], arm == "table_only"
        )
        # Each omission is local to the scored edit. The two omissions are separate paths.
        second.append(
            native_row_indices(start if route == "omit" else table2, ref).reshape(
                n, num, 4
            )
        )
        fourth.append(
            native_row_indices(table3 if route == "omit" else table4, ref).reshape(
                n, num, 4
            )
        )
    return dict(
        first=first.numpy(),
        second=torch.stack(second, 1).numpy(),
        fourth=torch.stack(fourth, 1).numpy(),
    )


@torch.no_grad()
def twins(model, arm, rows, masks):
    representatives = masks[::2]
    if any(a ^ b != 15 for a, b in zip(representatives, masks[1::2])):
        raise ValueError("Need complement-paired masks")
    grid = torch.tensor(list(product(range(len(rows)), representatives, masks)))
    ix, prior, current = grid.T
    ref = rows[ix]
    # The two histories use this very same float tensor; their desired next actions are opposite.
    shared = torch.where(truth_bits(prior)[:, :, None], ref[:, 1:2], ref[:, :1])
    results = []
    for history in (prior, 15 - prior):
        bit = (model.write(history) > 0).long() * int(arm != "table_only")
        conditions = []
        for incoming in (bit, torch.zeros_like(bit)):
            revised, _ = final_edit(
                model, shared, incoming, current, arm == "table_only"
            )
            conditions.append(
                native_row_indices(revised, ref).reshape(
                    len(rows), len(representatives), len(masks), 4
                )
            )
        results.append(torch.stack(conditions, -2))
    return torch.stack(results, -3).numpy()
