"""A source states a default, an exception trigger, and its possible veto."""

from dataclasses import dataclass, replace
from itertools import product

SIDES = ("left", "right")
VALUES = ("off", "on")


def condition(literal, world):
    if literal == -1:
        return False
    if literal not in range(4) or world not in range(4):
        raise ValueError("Unknown binary predicate or world")
    return ((world >> (1 - literal // 2)) & 1) == literal % 2


@dataclass(frozen=True)
class Case:
    topic: int
    base: int
    veto: int
    pole: int
    style: int
    order: int

    @property
    def key(self):
        return (self.topic, self.base, self.veto, self.pole, self.style, self.order)

    def donor(self, kind):
        if kind == "opposed":
            return replace(self, pole=1 - self.pole)
        if kind == "equivalent":
            return replace(self, style=1 - self.style)
        if kind == "reordered":
            return replace(self, order=1 - self.order)
        if kind == "twin":
            return replace(self, base=self.base ^ 1, pole=self.pole ^ 1)
        raise ValueError(kind)

    def fields(self, menus, styles, prefix=""):
        words = styles[self.style]
        actions = [words[r].format(a=menus[self.topic][self.pole ^ r]) for r in (0, 1)]
        base = words[2].format(s=SIDES[self.base // 2], v=VALUES[self.base % 2])
        guard = (
            words[4]
            if self.veto < 0
            else words[3].format(s=SIDES[self.veto // 2], v=VALUES[self.veto % 2])
        )
        return [
            prefix + s for s in [*actions[:: 1 if self.order == 0 else -1], base, guard]
        ]

    def ledger(self):
        rows = []
        for w in range(4):
            trigger = condition(self.base, w)
            veto = condition(self.veto, w)
            rows.append(
                dict(
                    world=w,
                    trigger=trigger,
                    veto=veto,
                    affected=trigger and veto,
                    base_choice=self.pole ^ int(trigger),
                    choice=self.pole ^ int(trigger and not veto),
                )
            )
        return rows

    def source(self, menus, styles, prefix=""):
        return "\n".join(self.fields(menus, styles, prefix))

    def successor(self, menus, styles):
        if not any(r["affected"] for r in self.ledger()):
            return self.source(menus, styles)
        return render_cases([menus[self.topic][r["choice"]] for r in self.ledger()])


def cases(config, split, raw=False):
    if split not in ("train", "fresh"):
        raise ValueError("Only development material can be loaded")
    topics = range(8) if split == "train" or not raw else config["raw_topics"]
    bases = range(2) if raw else range(4)
    vetoes = (-1, 3) if raw else range(-1, 4)
    styles = range(1) if raw and split == "train" else range(len(config[split]))
    orders = (0,) if raw and split == "train" else range(2)
    return [Case(*x) for x in product(topics, bases, vetoes, range(2), styles, orders)]


def world_text(world):
    return f"the left signal is {VALUES[world // 2]} and the right signal is {VALUES[world % 2]}"


def row_text(world, action):
    return f"When {world_text(world)}, I choose {action}."


def render_cases(actions):
    if len(actions) != 4:
        raise ValueError("The case representation needs four world rows")
    return "\n".join(row_text(w, a) for w, a in enumerate(actions))


def questions(case, menus, text=None, prefix=""):
    jobs = []
    for r in case.ledger():
        for order in range(2):
            first, second = menus[case.topic][:: 1 if order == 0 else -1]
            question = (
                f"{prefix}Today {world_text(r['world'])}. Which action follows my policy?\n"
                f"A = {first}. B = {second}. Respond with only A or B."
            )
            jobs.append(
                dict(
                    key="current",
                    source=text,
                    world=r["world"],
                    label_order=order,
                    question=question,
                    gold="AB"[r["choice"] ^ order],
                )
            )
    return jobs
