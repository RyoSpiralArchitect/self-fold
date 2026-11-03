"""A supervised two-branch policy memory and a learned world-to-world update."""

from itertools import product

import torch
from torch import nn

OPERATIONS = ("both_on", "either_on", "exactly_one_on")
ROUTES = (
    "matched",
    "opposed",
    "equivalent",
    "joint_row",
    "default_rows",
    "omit",
    "uniform",
    "oracle",
)


def branch_for(operation, world):
    """Independent Boolean reference used for supervision and grading."""
    left, right = (bool(world & 2), bool(world & 1))
    values = {
        "both_on": left and right,
        "either_on": left or right,
        "exactly_one_on": left != right,
    }
    if world not in range(4):
        raise ValueError("Invalid world")
    return int(values[operation])


def source_text(design, flags, style, actions, order):
    clauses = [
        s.format(a=flags[0], b=flags[1], action=act)
        for s, act in zip(design["styles"][style], actions, strict=True)
    ]
    return "\n".join(clauses[:: 1 if order == 0 else -1])


def training_clauses(design, actions):
    rows = []
    for flags, style, value, role in product(
        design["fit_flags"], design["fit_styles"], range(len(actions)), range(2)
    ):
        text = design["styles"][style][role].format(
            a=flags[0], b=flags[1], action=actions[value]
        )
        rows.append((text, value, role))
    return rows


def source_bank(design, groups, split):
    if split == "fit":
        flags, styles = design["fit_flags"], design["fit_styles"]
    elif split in ("primary", "challenge"):
        flags, styles = design["test_flags"], design[split + "_styles"]
    else:
        raise ValueError(split)
    for topic, flag, style, pole, order in product(
        range(len(groups)), range(len(flags)), styles, range(2), range(2)
    ):
        options = groups[topic]["options"]
        actions = [options[pole], options[1 - pole]]
        yield (topic, flag, style, pole, order), source_text(
            design, flags[flag], style, actions, order
        )


class ClauseReader(nn.Module):
    def __init__(self, width, actions):
        super().__init__()
        self.value = nn.Linear(width, actions)
        self.role = nn.Linear(width, 2)
        for p in self.parameters():
            nn.init.zeros_(p)

    def compile(self, features):
        """Two clause vectors in presentation order, with no position, labels or metadata."""
        if features.ndim != 3 or features.shape[1] != 2:
            raise ValueError("A policy requires two observed clause vectors")
        action = self.value(features).softmax(-1)
        assignment = self.role(features).log_softmax(-1).transpose(1, 2).softmax(-1)
        branches = assignment @ action
        return branches[:, [0, 0, 0, 1], :]


class PolicyUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        self.routing = nn.Parameter(torch.zeros(len(OPERATIONS), 4, 4))

    def forward(self, state, operation):
        """The operation is global. It receives the entire memory before a world is selected."""
        return self.routing[operation].softmax(-1) @ state


def donor_states(matched, opposed, equivalent, oracle):
    """Replace continuous rows at the boundary between compiler and update."""
    return {
        "matched": matched,
        "opposed": opposed,
        "equivalent": equivalent,
        "joint_row": torch.cat((matched[:3], opposed[3:]), dim=0),
        "default_rows": torch.cat((opposed[:3], matched[3:]), dim=0),
        "omit": matched,
        "uniform": torch.full_like(matched, 1 / matched.shape[-1]),
        "oracle": oracle,
    }


def selected_action(topic, pole, operation, world):
    return 2 * topic + (pole ^ branch_for(operation, world))
