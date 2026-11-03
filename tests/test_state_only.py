import json
from pathlib import Path

import torch

from selffold.state_only import (
    ClauseReader,
    OPERATIONS,
    PolicyUpdate,
    branch_for,
    donor_states,
    selected_action,
    source_bank,
    training_clauses,
)

ROOT = Path(__file__).resolve().parents[1]
DESIGN = json.loads((ROOT / "data/state_only_20261104.json").read_text())
GROUPS = json.loads((ROOT / "data/within_topic_wording_20261103.json").read_text())[
    "groups"
]


def test_sealed_source_bank_and_independently_swapped_label_truth():
    actions = [a for g in GROUPS for a in g["options"]]
    fit = training_clauses(DESIGN, actions)
    assert len({x[0] for x in fit}) == len(fit) == 288
    for panel, n in [("fit", 288), ("primary", 128), ("challenge", 64)]:
        bank = dict(source_bank(DESIGN, GROUPS, panel))
        assert len(bank) == len(set(bank.values())) == n
        for (t, f, s, p, o), source in bank.items():
            assert source.splitlines()[::-1] == bank[t, f, s, p, 1 - o].splitlines()
            assert all(act in source for act in GROUPS[t]["options"])
            if panel != "fit":
                assert not {x[0] for x in fit}.intersection(source.splitlines())
            for op in OPERATIONS:
                for world in range(4):
                    assert (
                        selected_action(t, p, op, world)
                        ^ selected_action(t, 1 - p, op, world)
                        == 1
                    )
    assert [[branch_for(op, w) for w in range(4)] for op in OPERATIONS] == [
        [0, 0, 0, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 0],
    ]


def test_continuous_memory_is_order_symmetric_and_swaps_are_local():
    torch.manual_seed(631)
    reader = ClauseReader(5, 16)
    with torch.no_grad():
        for p in reader.parameters():
            p.normal_()
    x = torch.randn(2, 2, 5)
    z = reader.compile(x)
    assert z.shape == (2, 4, 16) and torch.allclose(z, reader.compile(x.flip(1)))
    assert torch.allclose(z.sum(-1), torch.ones(2, 4))
    swaps = donor_states(z[0], z[1], z[0], z[1])
    assert torch.equal(swaps["joint_row"][:3], z[0, :3]) and torch.equal(
        swaps["joint_row"][3], z[1, 3]
    )
    assert torch.equal(swaps["default_rows"][:3], z[1, :3]) and torch.equal(
        swaps["default_rows"][3], z[0, 3]
    )
    model = PolicyUpdate()
    with torch.no_grad():
        model.routing.fill_(-60)
        for op_id, op in enumerate(OPERATIONS):
            for w in range(4):
                model.routing[op_id, w, 3 if branch_for(op, w) else 0] = 60
    for op_id, op in enumerate(OPERATIONS):
        updated = model(z, op_id)
        assert torch.equal(
            updated, z[:, [3 if branch_for(op, w) else 0 for w in range(4)]]
        )
