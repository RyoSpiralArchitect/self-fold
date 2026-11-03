"""Audit actual conditional source, case and rule states on four later worlds.

Every state is independently encoded before any audit question. No action-row
lookup or topic/pole classifier supplies a world-dependent answer.
"""

from collections import defaultdict
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/conditional_audit_20261103.json"
ACTIONS = ROOT / "data/within_topic_wording_20261103.json"
STAGES = ("source", "cases", "rule")


def world_text(flags, world):
    return " and ".join(
        f"{f} is {'on' if v else 'off'}" for f, v in zip(flags, world, strict=True)
    )


def action_index(pole, world):
    return pole if all(world) else 1 - pole


def audit_cases(groups, fixture):
    if set(fixture["groups"]) != {g["id"] for g in groups}:
        raise ValueError("Conditional audit topics differ")
    if fixture["worlds"] != [
        list(v) for v in itertools.product((False, True), repeat=2)
    ]:
        raise ValueError("Conditional audit must cover all four worlds")
    for topic, g in enumerate(groups):
        flags = fixture["groups"][g["id"]]["flags"]
        if len(set(flags)) != 2:
            raise ValueError("Two distinct flags are required")
        for pole in (0, 1):
            selected, other = g["options"][pole], g["options"][1 - pole]
            clauses = (
                f"If {flags[0]} and {flags[1]} are both on, I choose {selected}.",
                f"If at least one of {flags[0]} and {flags[1]} is off, I choose {other}.",
            )
            table = [
                f"When {world_text(flags, w)}, I choose {g['options'][action_index(pole, w)]}."
                for w in fixture["worlds"]
            ]
            prefix = "My current conditional policy is as follows. "
            rule_condition = (
                f"({world_text(flags, [True, True])})"
                if pole == 0
                else f"({flags[0]} is off) or ({flags[1]} is off)"
            )
            canonical = f"{prefix}I choose {g['options'][0]} exactly when {rule_condition}. In every other case I choose {g['options'][1]}."
            for stage in STAGES:
                for order in (0,) if stage == "rule" else (0, 1):
                    if stage == "source":
                        text = prefix + " ".join(
                            clauses if order == 0 else clauses[::-1]
                        )
                    elif stage == "cases":
                        text = (
                            prefix
                            + "The cases are mutually exclusive and exhaustive. "
                            + " ".join(table if order == 0 else table[::-1])
                        )
                    else:
                        text = canonical
                    yield {
                        "key": f"{g['id']}/{stage}/{pole}/{order}",
                        "topic": topic,
                        "group": g["id"],
                        "stage": stage,
                        "pole": pole,
                        "source_order": order,
                        "source": text,
                    }


def tasks_for(case, group, fixture, world_index):
    details = fixture["groups"][group["id"]]
    world = fixture["worlds"][world_index]
    answer = action_index(case["pole"], world)
    result = []
    for order in (0, 1):
        options = group["options"] if order == 0 else group["options"][::-1]
        question = (
            f"Current indicators: {world_text(details['flags'], world)}.\n{details['operation']}\n"
            f"A = {options[0]}. B = {options[1]}. Use the supplied description."
        )
        result.append(
            {
                "key": case["key"],
                "source": case["source"],
                "question": question,
                "world": world_index,
                "label_order": order,
                "gold": "A" if answer == order else "B",
            }
        )
    return result


def evaluate(host, adapter, groups, store):
    import torch
    from fit_continuity_recipient import model_scores
    from selffold.records import State, digest, file_sha

    with torch.no_grad():
        if any(torch.count_nonzero(f.up).item() for f in adapter.factors) or any(
            p.requires_grad for p in adapter.parameters()
        ):
            raise ValueError("Conditional audit requires an untrained frozen recipient")
        fixture = json.loads(DATA.read_text())
        if file_sha(ACTIONS) != fixture["action_fixture_sha256"]:
            raise ValueError("Conditional audit action fixture changed")
        cases = list(audit_cases(groups, fixture))
        if len(cases) != 80 or len({c["source"] for c in cases}) != 80:
            raise ValueError("Conditional teacher states must all be distinct")
        states = {}
        # All three independently encoded teacher stages precede all audit questions.
        for c in cases:
            z = host.encode(c["source"])
            states[c["key"]] = z
            store.append(
                "audit_sources.jsonl",
                {k: v for k, v in c.items() if k != "source"}
                | {"source_sha256": digest(c["source"]), "state_sha256": z.sha()},
            )
        zero = torch.zeros_like(next(iter(states.values())).slots)
        zero_sha = State(slots=zero).sha()
        cache = {}
        for c in cases:
            for world in range(4):
                tasks = tasks_for(c, groups[c["topic"]], fixture, world)
                for route in ("literal", "raw_E", "zero"):
                    supplied = (
                        None
                        if route == "literal"
                        else zero
                        if route == "zero"
                        else states[c["key"]].slots
                    )
                    donor_sha = (
                        digest(c["source"])
                        if route == "literal"
                        else zero_sha
                        if route == "zero"
                        else states[c["key"]].sha()
                    )
                    cache_key = (
                        route,
                        donor_sha,
                        tuple(digest(t["question"]) for t in tasks),
                    )
                    if cache_key not in cache:
                        cache[cache_key] = model_scores(
                            host,
                            adapter,
                            tasks,
                            {c["key"]: supplied},
                            "literal_base" if route == "literal" else "matched",
                        )
                    for t, score in zip(tasks, cache[cache_key], strict=True):
                        store.append(
                            "audit_decisions.jsonl",
                            {
                                k: c[k]
                                for k in (
                                    "key",
                                    "topic",
                                    "group",
                                    "stage",
                                    "pole",
                                    "source_order",
                                )
                            }
                            | {k: t[k] for k in ("world", "label_order", "gold")}
                            | {
                                "route": route,
                                "source_sha256": digest(c["source"]),
                                "injected_sha256": donor_sha,
                                "question_sha256": digest(t["question"]),
                            }
                            | score,
                        )
        return {
            "fixture_sha256": file_sha(DATA),
            "independent_teacher_states": len(cases),
            "native_decisions": 1920,
            "routes": ["literal", "raw_E", "zero"],
            "G_trained": False,
        }


def summarize(records):
    panels = {}
    gates = {}
    for stage in STAGES:
        total = 128 if stage == "rule" else 256
        for route in ("literal", "raw_E", "zero"):
            data = [r for r in records if r["stage"] == stage and r["route"] == route]
            key = lambda r: (r["key"], r["world"], r["label_order"])
            if len(data) != len({key(r) for r in data}) or len(data) != total:
                raise ValueError("Incomplete conditional audit stage")
            correct = lambda r: r["greedy_label"] == r["gold"]
            groups = defaultdict(list)
            for r in data:
                groups[r["group"]].append(r)
            if len(groups) != 8:
                raise ValueError("Conditional audit dropped a topic")
            panels[stage + "/" + route] = {
                "native": [sum(map(correct, data)), len(data)],
                "forced": [
                    sum(
                        r[r["gold"]] > r["B" if r["gold"] == "A" else "A"] for r in data
                    ),
                    len(data),
                ],
                "coverage": [
                    sum(r["greedy_label"] in ("A", "B") for r in data),
                    len(data),
                ],
                "ties": sum(r["A"] == r["B"] for r in data),
                "strict_topics": [
                    sum(all(map(correct, g)) for g in groups.values()),
                    8,
                ],
            }
        raw, null = (panels[stage + "/" + r] for r in ("raw_E", "zero"))
        gates[stage] = (
            raw["native"][0] * 100 >= 90 * total
            and raw["strict_topics"][0] >= 6
            and (raw["native"][0] - null["native"][0]) * 100 >= 10 * total
        )
    return {
        "panels": panels,
        "stages": gates,
        "native_three_stage_admission": all(gates.values()),
        "distinct_teacher_states": {"source": 32, "cases": 32, "rule": 16},
    }
