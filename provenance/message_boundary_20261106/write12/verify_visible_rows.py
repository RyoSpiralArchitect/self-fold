"""Verify saved outcomes using the actual prompt text. Uses only the standard library."""
import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import itertools
import json
from pathlib import Path
import re


def check(condition, message):
    if not condition:
        raise ValueError(message)


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def verify(folder):
    with gzip.open(folder / "requests.jsonl.gz", "rt", encoding="utf8") as stream:
        requests = [json.loads(line) for line in stream]
    reports = [json.loads(line) for line in (folder / "reports.jsonl").read_text().splitlines()]
    scores = json.loads((folder / "scores.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    check(len(requests) == len(reports) == len(scores) == len({q["prompt"] for q in requests}) == 3840,
          "Incomplete or repeated prompts")
    positions = defaultdict(list)
    question_rows = defaultdict(list)
    panels = defaultdict(list)
    complete = defaultdict(dict)
    oracle_swaps = defaultdict(dict)
    oracle_baselines = Counter()
    for q, a, score in zip(requests, reports, scores, strict=True):
        blob = json.dumps(q, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        check(q["id"] == a["id"] == score["id"] and sha256(blob).hexdigest() == a["request_sha256"],
              "Invalid request or output binding")
        prefix, *blocks, ending = q["prompt"].split("\n\n")
        names = prefix.splitlines()[-2:]
        query, cue, _ = ending.splitlines()
        target_label = re.search(r"(off/off|off/on|on/off|on/on) row\?$", query).group(1)
        name = re.match(r"Requested record: (INSTRUCTION RECORD|OBSERVATION RECORD)\.", cue).group(1)
        question = "instruction" if "did this instruction assign" in query else "actual"
        check((question == "instruction") == (name == "INSTRUCTION RECORD"), "Cue and question disagree")
        headings = []
        tables = {}
        for block in blocks:
            heading, lines = block.splitlines()
            heading = heading.removesuffix(":").split(" (")[0]
            headings.append(heading)
            rows = dict(pair.split(" -> ") for pair in lines.split("; "))
            check(set(rows) == {"off/off", "off/on", "on/off", "on/on"} and set(rows.values()) <= set(names),
                  "Malformed rendered table")
            tables[heading] = rows
        target = tables[name]
        if len(blocks) == 1:
            panel = "single/single"
            check(headings == [name], "Single prompt contains another record")
        else:
            check(len(blocks) == 3 and headings[0] == "PREVIOUS TABLE", "Bad full context")
            order = "instruction_then_observation" if headings[1] == "INSTRUCTION RECORD" else "observation_then_instruction"
            relation = "agree" if tables["INSTRUCTION RECORD"] == tables["OBSERVATION RECORD"] else "conflict"
            if relation == "conflict":
                check(all(tables["INSTRUCTION RECORD"][k] != tables["OBSERVATION RECORD"][k] for k in target),
                      "Conflicting records have a shared assignment")
            panel = order + "/" + relation
        gold = target[target_label]
        position = list(target).index(target_label) + 1
        correct = normalized(a["text"]) == normalized(gold)
        check(score["expected_slot"] == names.index(gold) and score["correct"] == correct
              and score["position"] == position - 1, "Scorer disagrees with the visible table")
        panels[panel].append(correct)
        positions[panel, str(position)].append(correct)
        question_rows[panel, question + "/" + target_label].append(correct)
        complete[panel, tuple(blocks), name][target_label] = correct
        if panel == "single/single":
            oracle_baselines["first_named_option"] += names[0] == gold
            oracle_baselines["first_printed_row"] += next(iter(target.values())) == gold
            oracle_baselines["last_printed_row"] += next(reversed(target.values())) == gold
            oracle_swaps[tuple(names), name, tuple(target), target_label][tuple(target.values())] = correct
    def count(values):
        items = list(values)
        return [sum(items), len(items)]
    for panel, bits in panels.items():
        metrics = summary["profiles"][panel]["metrics"]
        check(metrics["overall"] == count(bits), "Overall counts differ")
        for name, value in metrics["by_position"].items():
            check(value == count(positions[panel, name]), "Position count differs")
        for name, value in metrics["by_question_row"].items():
            check(value == count(question_rows[panel, name]), "Coordinate count differs")
        groups = [answers for (p, _, _), answers in complete.items() if p == panel]
        check(all(len(v) == 4 for v in groups), "Four-row group incomplete")
        check(metrics["all_four_rows"] == count(all(v.values()) for v in groups), "Four-row counts differ")
    swap_results = []
    for (_, _, labels, queried), mappings in oracle_swaps.items():
        i = labels.index(queried)
        for (left, l_ok), (right, r_ok) in itertools.combinations(mappings.items(), 2):
            if left[i] != right[i] and sum(x != y for x, y in zip(left, right, strict=True)) == 2:
                swap_results.append(l_ok and r_ok)
    swaps = count(swap_results)
    check(swaps == summary["profiles"]["single/single"]["metrics"]["both_sides_of_swap"], "Swap count differs")
    return dict(exact=True, prompt_derived_scores=len(scores), profiles=len(panels), oracle_swaps=swaps,
                oracle_baselines={key: [n, len(panels["single/single"])] for key, n in oracle_baselines.items()},
                script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                inputs={name: sha256((folder / name).read_bytes()).hexdigest() for name in
                        ("requests.jsonl.gz", "reports.jsonl", "scores.json", "summary.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(json.dumps(verify(parser.parse_args().folder), indent=2, sort_keys=True))
