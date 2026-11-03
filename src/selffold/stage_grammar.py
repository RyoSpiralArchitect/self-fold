"""Matched literal continuation and prespecified panels for three representation stages."""

from copy import deepcopy
from .conditional_curriculum import tasks_for
from .two_flags import branch_probe, world_probe

ARMS = ("repeat", "vary", "focus")
PANELS = ("fit", "names", "grammar", "fresh", "previous")


def fixture(base, previous, design, panel):
    f = deepcopy(base)
    f["styles"].update(design["extra_styles"])
    f["styles"][design["new_style_name"]] = design["new_style"]
    f["styles"][previous["new_style_name"]] = previous["new_style"]
    if panel == "fit":
        return f
    if panel == "train":
        f["fit"]["styles"] += list(design["extra_styles"])
        return f
    d = previous if panel == "previous" else design
    if panel not in PANELS:
        raise ValueError(panel)
    f["fresh"] = {
        "flag_pairs": (
            d["new_flags"] if panel != "grammar" else f["fit"]["flag_pairs"][:2]
        ),
        "styles": [d["new_style_name"]] if panel != "names" else f["fit"]["styles"][:1],
        "world_prefix": (
            d["world_prefix"]
            if panel in ("fresh", "previous")
            else f["fit"]["world_prefix"]
        ),
        "question": (
            d["question"] if panel in ("fresh", "previous") else f["fit"]["question"]
        ),
        "operations": (
            d["operations"] if panel in ("fresh", "previous") else f["operations"]
        ),
    }
    return f


def case_for(lookup, unit, pole, step, slot, arm, base, design):
    group, stage, flag, style, order = unit
    # The stage, semantic binding, names and order are yoked. Only grammar is replaced in half the blocks.
    if arm != "repeat" and (step // 2 + slot) % 2:
        style = list(design["extra_styles"])[base["fit"]["styles"].index(style)]
    return lookup[group, stage, flag, style, order, pole]


def training_pair(c, group, f, previous, design, step, slot, pole, world, arm):
    cycle = (
        design["focus_cycle"]
        if arm == "focus"
        else ["action", "flag", "and", "bind"] * 4
    )
    kind = cycle[(step + slot * 8 + pole * 4 + world) % 16]
    words = (
        previous["fit_probes"]
        if (step // 3 + slot) % 2 == 0
        else design["alternate_probes"]
    )
    if kind == "action":
        pair = tasks_for(c, group, f, world)
    elif kind == "bind":
        pair = branch_probe(c, group, f, words, all(f["worlds"][world]))
    else:
        pair = world_probe(
            c["flags"], world, f, words, kind, (step // 4 + slot + pole) % 2
        )
    return kind, pair


def panel_jobs(groups, f, words, panel):
    from .conditional_curriculum import make_cases

    split = "fit" if panel == "fit" else "fresh"
    cases = list(make_cases(list(groups.values()), f, split))
    for c in cases:
        g = groups[c["group"]]
        for w in range(4):
            for t in tasks_for(c, g, f, w):
                yield "ordinary", c["key"], w, t["label_order"], t
        if panel in ("fresh", "previous"):
            for bit in (False, True):
                for t in branch_probe(c, g, f, words, bit):
                    yield "branch", c["key"], int(bit), t["label_order"], t
    if panel in ("fresh", "previous"):
        for pair, flags in enumerate(f[split]["flag_pairs"]):
            for name, i in (("flag", 0), ("flag", 1), ("and", 0)):
                for w in range(4):
                    for t in world_probe(flags, w, f, words, name, i):
                        yield "world", f"{pair}/{name}/{i}", w, t["label_order"], t
