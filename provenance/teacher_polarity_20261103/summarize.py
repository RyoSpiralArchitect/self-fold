"""Reduce the separated native-teacher and question-free polarity endpoints."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
OLD = ROOT / "provenance/within_topic_20261103/run"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def count(bools):
    values = list(bools)
    return [sum(values), len(values)]


def passes(measure, threshold):
    return measure[1] > 0 and measure[0] * 100 >= measure[1] * threshold


def history(folder=OLD):
    assignments = {
        r["key"]: r["gold_class"] == r["predicted_class"]
        for r in rows(folder / "assignments.jsonl")
        if (r["arm"], r["endpoint"], r["panel"]) == ("many", "final", "joint")
    }
    selected = defaultdict(dict)
    for r in rows(folder / "decisions.jsonl"):
        if (r["arm"], r["endpoint"], r["panel"]) == ("many", "final", "joint"):
            selected[r["route"]][r["key"], r["wording"], r["label_order"]] = r
    table = Counter()
    for k, m in selected["matched"].items():
        oracle = selected["rendered_teacher"][k]
        assert (m["gold"], m["question_sha256"]) == (
            oracle["gold"],
            oracle["question_sha256"],
        )
        table[assignments[k[0]], correct(oracle), correct(m)] += 1
    assert len(assignments) == 128 and sum(table.values()) == 512
    return [
        {
            "assignment_correct": a,
            "rendered_oracle_correct": o,
            "matched_correct": m,
            "n": table[a, o, m],
        }
        for a in (False, True)
        for o in (False, True)
        for m in (False, True)
    ]


def correct(r):
    return r["greedy_label"] == r["gold"]


def native(rows_):
    return {
        "native": count(correct(r) for r in rows_),
        "coverage": count(r["greedy_label"] in ("A", "B") for r in rows_),
        "forced": count(
            r[r["gold"]] > r["B" if r["gold"] == "A" else "A"] for r in rows_
        ),
        "ties": sum(r["A"] == r["B"] for r in rows_),
    }


def teacher_stats(data):
    result = {}
    for split, route in sorted({(r["split"], r["route"]) for r in data}):
        group = [r for r in data if (r["split"], r["route"]) == (split, route)]
        idx = {
            (r["topic"], r["pole"], r["wording"], r["label_order"]): r for r in group
        }
        assert len(idx) == len(group) == 128
        topics = {t: [r for r in group if r["topic"] == t] for t in range(8)}
        assert all(len(v) == 16 for v in topics.values())
        deltas = {k: r["A"] - r["B"] for k, r in idx.items()}
        result[f"{split}/{route}"] = native(group) | {
            "each_wording": [
                native([r for r in group if r["wording"] == q])["native"]
                for q in range(4)
            ],
            "each_topic": [native(v)["native"] for v in topics.values()],
            "strict_topics": count(all(map(correct, v)) for v in topics.values()),
            "opposed_flips": count(
                d * deltas[t, 1, q, o] < 0
                for (t, p, q, o), d in deltas.items()
                if p == 0
            ),
        }
    f = result["fresh/calibrated"]
    gate = (
        passes(result["fit/calibrated"]["native"], 95)
        and passes(f["native"], 95)
        and all(passes(n, 95) for n in f["each_wording"])
        and f["strict_topics"][0] >= 7
        and passes(f["opposed_flips"], 90)
        and result["fresh/mean"]["native"][0] <= 64
    )
    return result, gate


def source_measure(data, method):
    idx = {(r["topic"], r["frame"], r["pair"], r["pole"], r["order"]): r for r in data}
    assert len(idx) == len(data)
    units = defaultdict(list)
    ok = {k: r[method] == 2 * k[0] + k[3] for k, r in idx.items()}
    for (topic, frame, pair, pole, order), good in ok.items():
        units[topic, frame, pair].append(good)
    assert all(len(v) == 4 for v in units.values())
    result = {
        "policy": count(ok.values()),
        "strict": count(all(v) for v in units.values()),
        "by_order": [count(v for k, v in ok.items() if k[4] == o) for o in (0, 1)],
    }
    for label, flip in (
        ("opposed_same_role_order", 0),
        ("opposed_same_action_order", 1),
    ):
        result[label] = count(
            first and ok[t, f, w, 1, o ^ flip]
            for (t, f, w, p, o), first in ok.items()
            if p == 0
        )
    return result


def source_stats(data):
    result = {}
    for split in ("fit", "fresh"):
        group = [r for r in data if r["split"] == split]
        assert len(group) == (512 if split == "fit" else 256)
        sets = {"all": group} | {
            f: [r for r in group if r["frame"] == f]
            for f in sorted({r["frame"] for r in group})
        }
        for frame, subset in sets.items():
            result[f"{split}/{frame}"] = {
                m: source_measure(subset, m)
                for m in ("factor", "direct", "mention", "flat", "oracle_factor")
            } | {
                "topic": count(r["scores"]["topic"] == r["topic"] for r in subset),
                "first_role": count(
                    r["scores"]["role"] * (1 - 2 * r["order"]) > 0 for r in subset
                ),
                "first_action_oracle_topic": count(
                    r["scores"]["first_action"][r["topic"]]
                    * (1 - 2 * (r["pole"] ^ r["order"]))
                    > 0
                    for r in subset
                ),
            }
    fit = result["fit/all"]["factor"]
    gate = passes(fit["policy"], 95) and passes(fit["strict"], 90)
    for frame in ("priority", "verdict"):
        sub = result[f"fresh/{frame}"]["factor"]
        gate = gate and all(
            passes(sub[k], level)
            for k, level in (
                ("policy", 90),
                ("strict", 75),
                ("opposed_same_role_order", 90),
                ("opposed_same_action_order", 90),
            )
        )
    return result, gate


def joint_stats(data):
    result, gate = {}, True
    for frame in ("priority", "verdict"):
        routes = {}
        for route in (
            "matched",
            "opposed_role_order",
            "opposed_action_order",
            "mean_E",
            "zero_E",
            "oracle",
        ):
            sub = [r for r in data if r["frame"] == frame and r["route"] == route]
            assert (
                len(sub)
                == len({(r["key"], r["wording"], r["label_order"]) for r in sub})
                == 1024
            )
            units = defaultdict(list)
            for r in sub:
                units[r["topic"], r["pair"]].append(correct(r))
            assert len(units) == 32 and all(len(v) == 32 for v in units.values())
            routes[route] = native(sub) | {
                "strict": count(all(v) for v in units.values())
            }
        matched = routes["matched"]
        n, den = matched["native"]
        gate = (
            gate
            and passes(matched["native"], 90)
            and passes(matched["strict"], 75)
            and (n - max(routes[k]["native"][0] for k in ("mean_E", "zero_E"))) * 100
            >= 10 * den
            and (
                n
                - max(
                    routes[k]["native"][0]
                    for k in ("opposed_role_order", "opposed_action_order")
                )
            )
            * 100
            >= 30 * den
        )
        result[frame] = routes
    return result, gate


def summary(folder):
    teachers, t = teacher_stats(rows(folder / "teachers.jsonl"))
    sources, s = source_stats(rows(folder / "sources.jsonl"))
    joint = folder / "joint.jsonl"
    assert joint.is_file() == (t and s)
    compound, j = joint_stats(rows(joint)) if joint.exists() else (None, None)
    return {
        "historical_joint": history(),
        "teacher": teachers,
        "source": sources,
        "joint": compound,
        "gates": {"teacher": t, "source": s, "joint": j},
    }


if __name__ == "__main__":
    print(
        json.dumps(
            summary(Path(sys.argv[1]) if len(sys.argv) > 1 else STUDY / "run"), indent=2
        )
    )
