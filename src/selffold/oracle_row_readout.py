"""Read a named row in a supplied record and in matched multi-record contexts."""
from __future__ import annotations

import hashlib
import itertools
from collections import Counter, defaultdict

from .actual_report import normalize
from .lm_writer import assignment_words
from .lm_writer_reports import fingerprint
from .roundtrip import WORLD, feasible_outs


QUESTIONS = ("instruction", "actual")
ORDERS = ("instruction_then_observation", "observation_then_instruction")
HEADINGS = {"instruction": "INSTRUCTION RECORD:", "actual": "OBSERVATION RECORD (the table now):"}


def table(options, pattern, rotation):
    actions = assignment_words(options, 0, pattern)
    entries = [f"{label} -> {action}" for label, action in zip(WORLD, actions, strict=True)]
    return "; ".join(entries[rotation:] + entries[:rotation])


def requests(design, groups):
    panel = []
    contexts = [("single", "single"), *itertools.product(ORDERS, ("agree", "conflict"))]
    menus = zip(design["topics"], design["previous_patterns"], strict=True)
    for topic, previous in menus:
        options = groups[topic]["options"]
        intro = ("You report records of a small state machine.\n"
                 "Valid action names (copy exactly one complete line, without quotes or explanation):\n"
                 + "\n".join(options) + "\n\n")
        for target, rotation, question, row in itertools.product(
            design["patterns"], design["rotations"], QUESTIONS, range(len(WORLD))
        ):
            label = WORLD[row]
            record_name = "INSTRUCTION RECORD" if question == "instruction" else "OBSERVATION RECORD"
            query = (f"Question: which action did this instruction assign to the {label} row?"
                     if question == "instruction" else
                     f"Question: which action does the table now contain in its {label} row?")
            ending = (f"\n\n{query}\nRequested record: {record_name}. Read the {label} entry of that record."
                      "\nAnswer with one of the valid action names only.")
            for context, relation in contexts:
                # The single-record prompt is unique across virtual partners and record orders.
                instruction = observation = prior = None
                if context == "single":
                    displayed = [HEADINGS[question] + "\n" + table(options, target, rotation)]
                else:
                    other = target if relation == "agree" else 15 - target
                    instruction, observation = (target, other) if question == "instruction" else (other, target)
                    prior = previous
                    if observation not in feasible_outs(prior, instruction):
                        raise ValueError("Transition cannot be rendered by the row-copy task")
                    active = [HEADINGS["instruction"] + "\n" + table(options, instruction, rotation),
                              HEADINGS["actual"] + "\n" + table(options, observation, rotation)]
                    if context == ORDERS[1]:
                        active.reverse()
                    displayed = ["PREVIOUS TABLE:\n" + table(options, prior, rotation), *active]
                key = f"{context}/{relation}/{topic}/{question}/{target}/{rotation}/{row}"
                panel.append(dict(id=key, topic=topic, question=question, target=target, row=row,
                                  rotation=rotation, context=context, relation=relation,
                                  prior=prior, instruction=instruction, observation=observation,
                                  options=options, prompt=intro + "\n\n".join(displayed) + ending))
    if (len(panel) != design["generation_calls"] or len({q["id"] for q in panel}) != len(panel)
            or len({q["prompt"] for q in panel}) != len(panel)):
        raise ValueError("Incorrect or duplicated generation panel")
    def order(q):
        return hashlib.sha256((design["order_seed"] + "/" + q["id"]).encode()).digest()
    return sorted(panel, key=order)


def score(q, answer, max_new_tokens, max_context):
    if answer["id"] != q["id"] or answer["request_sha256"] != fingerprint(q):
        raise ValueError("Answer is not bound to this request")
    gen = answer["generation"]
    length, stop = gen["token_count"], gen["stop"]
    budget = min(max_new_tokens, max_context - gen["prompt_tokens"])
    if (not isinstance(answer["text"], str) or type(length) is not int
            or not 0 <= length <= budget or gen["budget"] != budget
            or stop not in {"eos", "length", "context_limit"}
            or gen["censored"] is not (stop != "eos")
            or (stop == "eos" and (gen["stop_token_id"] is None or length >= budget))
            or (stop != "eos" and (length != budget or gen["stop_token_id"] is not None))):
        raise ValueError("Invalid generation receipt")
    target_slot = (q["target"] >> q["row"]) & 1
    names = [normalize(option) for option in q["options"]]
    if len(set(names)) != 2 or not all(names):
        raise ValueError("Action names are not distinguishable")
    text = answer["text"]
    value = normalize(text)
    picked = names.index(value) if value in names else None
    return dict(id=q["id"], expected_slot=target_slot, picked_slot=picked, correct=picked == target_slot,
                valid_name=picked is not None, literal_correct=text.strip() == q["options"][target_slot],
                position=(q["row"] - q["rotation"]) % len(WORLD), stop=stop)


def count(bits):
    bits = list(bits)
    return [sum(bits), len(bits)]


def meets(pair, percent):
    correct, total = pair
    return total > 0 and 100 * correct >= percent * total


def profile(rows, limits):
    def grouped(key):
        parts = defaultdict(list)
        for item in rows:
            parts[str(key(item))].append(item["correct"])
        return {name: count(values) for name, values in sorted(parts.items())}

    tables = defaultdict(dict)
    swap_sets = defaultdict(dict)
    for item in rows:
        topic, question, pattern, rotation, row = (item[k] for k in
                                                   ("topic", "question", "target", "rotation", "row"))
        tables[topic, question, pattern, rotation][row] = item["correct"]
        swap_sets[topic, question, rotation, row][pattern] = item["correct"]
    if any(len(answers) != 4 for answers in tables.values()):
        raise ValueError("Incomplete four-row block")
    complete = count(all(answers.values()) for answers in tables.values())
    # Each unordered pair exchanges just the queried row and one other row.
    swaps = count(a and b for (*_, row), values in swap_sets.items()
                  for (p, a), (q, b) in itertools.combinations(values.items(), 2)
                  if (p ^ q).bit_count() == 2 and (p ^ q) & (1 << row))
    metrics = dict(overall=count(r["correct"] for r in rows),
                   valid_name=count(r["valid_name"] for r in rows),
                   literal_correct=count(r["literal_correct"] for r in rows),
                   by_question=grouped(lambda r: r["question"]),
                   by_question_row=grouped(lambda r: r["question"] + "/" + WORLD[r["row"]]),
                   by_position=grouped(lambda r: r["position"] + 1), by_menu=grouped(lambda r: r["topic"]),
                   by_rotation=grouped(lambda r: r["rotation"]), all_four_rows=complete,
                   both_sides_of_swap=swaps)
    checks = {key: all(meets(pair, limits[key]) for pair in metrics[metric].values())
              for key, metric in [("each_question", "by_question"), ("each_question_row", "by_question_row"),
                                  ("each_menu", "by_menu"), ("each_position", "by_position")]}
    checks.update({key: meets(metrics[key], limits[key]) for key in ("all_four_rows", "both_sides_of_swap")})
    return dict(metrics=metrics, checks=checks, passed=all(checks.values()))


def analyze(design, reqs, answers):
    indexed = {row["id"]: row for row in answers}
    if len(indexed) != len(answers) or set(indexed) != {q["id"] for q in reqs}:
        raise ValueError("Foreign, duplicate or incomplete answer panel")
    scores = [score(q, indexed[q["id"]], design["max_new_tokens"], design["max_context"]) for q in reqs]
    panels = defaultdict(list)
    singles = {}
    conflicts = defaultdict(dict)
    for q, s in zip(reqs, scores, strict=True):
        panels[q["context"], q["relation"]].append(q | s)
        key = q["topic"], q["target"], q["rotation"], q["row"], q["question"]
        if q["context"] == "single":
            singles[key] = s["correct"]
        elif q["relation"] == "conflict":
            conflicts[q["context"]][key] = s["correct"]
    limits = design["threshold_percent"]
    profiles = {f"{ctx}/{rel}": profile(rows, limits) for (ctx, rel), rows in sorted(panels.items())}
    # For a conflicting transition the instruction and observation differ at every row.
    # Pair the same two targets with their single-record presentations once per order.
    units = [key[:4] for key in singles if key[4] == "instruction"]
    def both(bank, unit):
        topic, pattern, rotation, row = unit
        return bank[topic, pattern, rotation, row, "instruction"] and bank[topic, 15-pattern, rotation, row, "actual"]
    paired = {}
    for layout in ORDERS:
        categories = Counter((both(singles, unit), both(conflicts[layout], unit)) for unit in units)
        gain = categories[True, False] - categories[False, True]
        paired[layout] = dict(both=categories[True, True], single_only=categories[True, False],
                              full_only=categories[False, True], neither=categories[False, False], total=len(units),
                              net_gain=[gain, len(units)], gain_met=meets([gain, len(units)], limits["paired_gain_points"]))
    oracle_pass = profiles["single/single"]["passed"]
    full_pass = {order: profiles[f"{order}/conflict"]["passed"] for order in ORDERS}
    if not oracle_pass:
        decision = "row_readout_first"
    elif all(full_pass.values()):
        decision = "both_contexts_reliable_on_panel"
    elif all(not full_pass[order] and paired[order]["gain_met"] for order in ORDERS):
        decision = "investigate_competing_records"
    else:
        decision = "resolve_order_or_context_dependence"
    summary = dict(profiles=profiles, matched_conflicts=paired,
                   single_both_questions=count(both(singles, u) for u in units),
                   full_both_questions_both_orders=count(all(both(conflicts[order], u) for order in ORDERS)
                                                        for u in units),
                   stops=dict(Counter(s["stop"] for s in scores)), unique_generations=len(scores),
                   single_record_pass=oracle_pass, conflict_full_pass=full_pass, decision=decision)
    return summary, scores
