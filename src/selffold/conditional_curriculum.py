"""Deterministic conditional teachers with disjoint flag names and wording families."""

import itertools

STAGES = ("source", "cases", "rule")


def action_index(pole, world):
    if pole not in (0, 1) or len(world) != 2 or any(type(v) is not bool for v in world):
        raise ValueError("Expected one of two policies and a two-Boolean world")
    return pole if all(world) else 1 - pole


def world_text(flags, world):
    return " and ".join(
        f"{flag} is {'on' if value else 'off'}"
        for flag, value in zip(flags, world, strict=True)
    )


def make_cases(groups, fixture, split):
    """Yield independent source, exhaustive-case and rule texts before any world is asked."""
    if split not in ("fit", "fresh"):
        raise ValueError("Unknown conditional curriculum split")
    if fixture["worlds"] != [
        list(v) for v in itertools.product((False, True), repeat=2)
    ]:
        raise ValueError("The fixture must cover all four Boolean worlds")
    if {g["id"] for g in groups} != set(fixture["operations"]):
        raise ValueError("Conditional curriculum must retain all eight topics")
    details = fixture[split]
    for topic, group in enumerate(groups):
        if len(group["options"]) != 2:
            raise ValueError("Expected two canonical actions per topic")
        for pair_index, flags in enumerate(details["flag_pairs"]):
            if len(flags) != len(set(flags)) or len(flags) != 2:
                raise ValueError("Need two distinct flag names")
            for style in details["styles"]:
                words = fixture["styles"][style]
                for pole in (0, 1):
                    values = dict(
                        first=flags[0],
                        second=flags[1],
                        selected=group["options"][pole],
                        other=group["options"][1 - pole],
                    )
                    clauses = (
                        words["source_true"].format(**values),
                        words["source_false"].format(**values),
                    )
                    table = [
                        words["case"].format(
                            world=world_text(flags, world),
                            action=group["options"][action_index(pole, world)],
                        )
                        for world in fixture["worlds"]
                    ]
                    for stage in STAGES:
                        for order in (0,) if stage == "rule" else (0, 1):
                            if stage == "source":
                                text = " ".join(
                                    clauses if order == 0 else clauses[::-1]
                                )
                            elif stage == "cases":
                                text = words["cases_prefix"] + " ".join(
                                    table if order == 0 else table[::-1]
                                )
                            else:
                                text = words["rule"].format(**values)
                            yield {
                                "key": f"{split}/{group['id']}/{pair_index}/{style}/{stage}/{pole}/{order}",
                                "split": split,
                                "topic": topic,
                                "group": group["id"],
                                "flag_pair": pair_index,
                                "flags": list(flags),
                                "style": style,
                                "stage": stage,
                                "pole": pole,
                                "source_order": order,
                                "source": words["prefix"] + text,
                            }


def tasks_for(case, group, fixture, world_index):
    """Ask one later world in both label orders without changing the supplied state."""
    if case["group"] != group["id"]:
        raise ValueError("Question topic and source topic differ")
    if world_index not in range(4):
        raise ValueError("World index must identify one of four observations")
    details = fixture[case["split"]]
    world = fixture["worlds"][world_index]
    correct = action_index(case["pole"], world)
    prefix = details["world_prefix"].format(world=world_text(case["flags"], world))
    operation = details.get("operations", fixture["operations"])[group["id"]]
    stem = details["question"].format(operation=operation)
    tasks = []
    for order in (0, 1):
        options = group["options"] if order == 0 else group["options"][::-1]
        tasks.append(
            {
                "key": case["key"],
                "source": case["source"],
                "question": f"{prefix}\n{stem}\nA = {options[0]}. B = {options[1]}. Use the supplied description.",
                "world": world_index,
                "label_order": order,
                "gold": "A" if correct == order else "B",
            }
        )
    return tasks
