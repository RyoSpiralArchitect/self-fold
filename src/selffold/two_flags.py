"""Tasks for a matched comparison of final-action and component supervision."""

from copy import deepcopy
from itertools import product

from .conditional_curriculum import action_index, tasks_for, world_text

ARM_NAMES = {
    "direct": "Practise the final choice",
    "parts": "Learn the parts of the choice",
}
TASKS = ("action", "flag", "and", "bind")
FACTORS = ("names", "grammar", "question")
PANELS = {
    "+".join(f for f, b in zip(FACTORS, bits) if b) or "familiar": bits
    for bits in product((False, True), repeat=3)
}
ALL_NEW = "names+grammar+question"


def panel_fixture(base, design, panel):
    fixture = deepcopy(base)
    names, grammar, question = PANELS[panel]
    fit = fixture["fit"]
    fixture["styles"][design["new_style_name"]] = design["new_style"]
    fixture["fresh"] = {
        "flag_pairs": design["new_flags"] if names else fit["flag_pairs"][:2],
        "styles": [design["new_style_name"] if grammar else fit["styles"][0]],
        "world_prefix": design["world_prefix"] if question else fit["world_prefix"],
        "question": design["question"] if question else fit["question"],
        "operations": design["operations"] if question else fixture["operations"],
    }
    return fixture


def label_pair(key, source, stem, options, correct, world):
    result = []
    for order in (0, 1):
        q = f"{stem}\nA = {options[order]}. B = {options[1-order]}. Use the supplied description."
        result.append(
            dict(
                key=key,
                source=source,
                question=q,
                label_order=order,
                world=world,
                gold="A" if correct == order else "B",
            )
        )
    return result


def world_probe(flags, world, fixture, words, task, flag_index=0):
    values = fixture["worlds"][world]
    prefix = words["world_prefix"].format(world=world_text(flags, values))
    if task == "flag":
        stem = words["flag"].format(flag=flags[flag_index])
        options, truth = ("on", "off"), values[flag_index]
    elif task == "and":
        stem = words["and"]
        options, truth = ("yes", "no"), all(values)
    else:
        raise ValueError("Expected a flag or conjunction question")
    return label_pair(
        "world",
        "Two indicators describe the current situation.",
        f"{prefix}\n{stem}",
        options,
        int(not truth),
        world,
    )


def branch_probe(case, group, fixture, words, bit):
    if type(bit) is not bool:
        raise ValueError("The intermediate result must be one Boolean")
    details = fixture[case["split"]]
    operation = details.get("operations", fixture["operations"])[case["group"]]
    question = details["question"].format(operation=operation)
    return label_pair(
        case["key"],
        case["source"],
        words["bit"][int(bit)] + "\n" + question,
        group["options"],
        action_index(case["pole"], (bit, bit)),
        None,
    )


def training_pairs(case, group, fixture, design, step, unit, world):
    action = tasks_for(case, group, fixture, world)
    kind = TASKS[(step + unit + world) % len(TASKS)]
    words = design["fit_probes"]
    if kind == "action":
        parts = action
    elif kind == "bind":
        parts = branch_probe(case, group, fixture, words, all(fixture["worlds"][world]))
    else:
        parts = world_probe(
            case["flags"], world, fixture, words, kind, (step // 4 + unit) % 2
        )
    return kind, action, parts
