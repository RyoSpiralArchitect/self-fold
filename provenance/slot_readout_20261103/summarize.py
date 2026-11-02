"""Reduce the three readout arms using only public JSON records."""

from collections import defaultdict
import json
from pathlib import Path
import sys


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def ratio(numerator, denominator):
    return {
        "correct": numerator,
        "total": denominator,
        "fraction": numerator / denominator,
    }


def native_metrics(panel, pair_fields):
    pairs = defaultdict(list)
    for row in panel:
        pairs[tuple(row[k] for k in pair_fields)].append(row)
    expected = 2 if "label_order" in pair_fields else 4
    assert pairs and all(len(p) == expected for p in pairs.values())
    return {
        "native": ratio(sum(r["greedy_label"] == r["gold"] for r in panel), len(panel)),
        "forced_ties": sum(r["A"] == r["B"] for r in panel),
        "forced": ratio(
            sum((r[r["gold"]] > r["B" if r["gold"] == "A" else "A"]) for r in panel),
            len(panel),
        ),
        "strict_pairs": ratio(
            sum(all(r["greedy_label"] == r["gold"] for r in p) for p in pairs.values()),
            len(pairs),
        ),
    }


def order_summary(folder):
    panels = defaultdict(list)
    for r in rows(folder / "decisions.jsonl"):
        panels[r["condition"], r["endpoint"], r["split"], r["form"]].append(r)
    result, unique = {}, {}
    assert len(panels) == 2 * 2 * 3 * 4
    for key, panel in sorted(panels.items()):
        dedup = defaultdict(list)
        for r in panel:
            dedup[r["group"], r["pole"], r["world"], r["label_order"]].append(r)
        for copies in dedup.values():
            assert len(copies) == 2 and {r["source_order"] for r in copies} == {0, 1}
            assert all(
                all(
                    r[k] == copies[0][k]
                    for k in (
                        "A",
                        "B",
                        "greedy_token",
                        "gold",
                        "question_sha256",
                        "source_sha256",
                    )
                )
                for r in copies
            )
        unique[key] = [p[0] for p in dedup.values()]
        expected = 64 if key[2] == "train" else 32
        assert len(unique[key]) == expected
        result["/".join(key)] = native_metrics(
            unique[key], ("group", "world", "label_order")
        )
    joint, screens = {}, {}
    for condition in ("literal", "latent"):
        for endpoint in ("forward_fit", "mixed_fit"):
            for split in ("train", "previous_dev", "additional_dev"):
                combined = defaultdict(list)
                for form in ("forward", "reverse", "permutation_1", "permutation_2"):
                    for r in unique[condition, endpoint, split, form]:
                        combined[r["group"], r["world"], r["label_order"]].append(r)
                assert all(len(p) == 8 for p in combined.values())
                name = "/".join((condition, endpoint, split))
                joint[name] = ratio(
                    sum(
                        all(r["greedy_label"] == r["gold"] for r in p)
                        for p in combined.values()
                    ),
                    len(combined),
                )
        trained = [
            result[f"{condition}/mixed_fit/train/{f}"] for f in ("forward", "reverse")
        ]
        screens[condition] = {
            "trainability": all(
                m["native"]["fraction"] >= 0.95
                and m["strict_pairs"]["fraction"] >= 0.90
                for m in trained
            ),
            "permutation": joint[f"{condition}/mixed_fit/train"]["fraction"] >= 0.90,
        }
    return {"by_order": result, "all_four_orders": joint, "screens": screens}


def question_metrics(panel):
    correct, margins = 0, defaultdict(list)
    for row in panel:
        scores = row["scores"]
        margin = (
            None
            if scores is None
            else scores[row["gold"]]["logprob"]
            - scores["B" if row["gold"] == "A" else "A"]["logprob"]
        )
        correct += margin is not None and margin > 0
        margins[row["case"]].append(margin)
    assert all(len(v) == 2 for v in margins.values())
    return {
        "by_order": ratio(correct, len(panel)),
        "order_averaged_donors": ratio(
            sum(all(x is not None for x in v) and sum(v) > 0 for v in margins.values()),
            len(margins),
        ),
    }


def decoder_summary(folder):
    generations = rows(folder / "generations.jsonl")
    nll = rows(folder / "nll.jsonl")
    questions = rows(folder / "questions.jsonl")
    controls = rows(folder / "controls.jsonl")
    result = {}
    for endpoint in ("initial", "final"):
        panel = [
            r
            for r in generations
            if r["endpoint"] == endpoint and r["control"] == "matched"
        ]
        losses = [r for r in nll if r["endpoint"] == endpoint]
        assert len(panel) == len(losses) == 24 and len({r["case"] for r in panel}) == 24
        exact = sum(r["text"].strip() == r["source"].strip() for r in panel)
        pairs = defaultdict(list)
        for r in panel:
            pairs[r["group"], r["phrasing"]].append(r)
        assert len(pairs) == 12 and all(len(p) == 2 for p in pairs.values())
        result[endpoint] = {
            "exact": ratio(exact, 24),
            "eos": sum(r["stop"] == "eos" for r in panel),
            "censored": sum(r["censored"] for r in panel),
            "unique_outputs": len({r["text"] for r in panel}),
            "exact_opposed_pairs": ratio(
                sum(
                    all(r["text"].strip() == r["source"].strip() for r in p)
                    for p in pairs.values()
                ),
                12,
            ),
            "nll": {
                c: sum(r[c] for r in losses) / 24
                for c in ("matched", "opposed", "zero")
            },
            "source_conditioned_references": sum(
                r["matched"] < min(r["opposed"], r["zero"]) for r in losses
            ),
            "reconstructed_text_questions": question_metrics(
                [r for r in questions if r["endpoint"] == endpoint]
            ),
        }
    result["controls"] = {
        c: question_metrics([r for r in controls if r["control"] == c])
        for c in ("original_text", "frozen_slots")
    }
    m = result["final"]
    result["reconstruction_screen"] = (
        m["exact"]["correct"] == m["eos"] == m["source_conditioned_references"] == 24
    )
    result["all_24_are_training_examples"] = True
    return result


def lexical_summary(folder):
    decisions = rows(folder / "decisions.jsonl")
    result = {}
    for split in ("all", "train", "transfer"):
        panel = (
            decisions
            if split == "all"
            else [r for r in decisions if r["split"] == split]
        )
        routes = defaultdict(list)
        for r in panel:
            name = r["route"] + ("_" + str(r["gain"]) if r["route"] == "gain" else "")
            routes[name].append(r)
        assert len(routes) == 13
        result[split] = {
            name: native_metrics(group, ("group",))
            for name, group in sorted(routes.items())
        }
        baseline = max(
            result[split][name]["native"]["fraction"]
            for name in ("gain_0", "random_gain4")
        )
        for name, m in result[split].items():
            null = (
                max(baseline, result[split]["bridge_zero"]["native"]["fraction"])
                if name == "bridge_matched"
                else baseline
            )
            m["screen"] = (
                m["native"]["fraction"] >= 0.8
                and m["native"]["fraction"] - null >= 0.1 - 1e-12
                and m["strict_pairs"]["fraction"] >= 0.75
            )
    return result


def summarize(root):
    output = {}
    for arm, reduce in (
        ("order", order_summary),
        ("decoder", decoder_summary),
        ("lexical", lexical_summary),
    ):
        folder = root / arm
        terminal = json.loads((folder / "terminal.json").read_text())
        output[arm] = {"terminal": terminal}
        if terminal["status"] == "COMPLETED_BOUNDED_READOUT":
            assert (
                not terminal["G_trained"]
                and not terminal["qualification_content_loaded"]
            )
            assert (
                terminal["backbone_sha256_before"] == terminal["backbone_sha256_after"]
            )
            output[arm]["metrics"] = reduce(folder)
    return output


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "run"
    print(json.dumps(summarize(root), indent=2, sort_keys=True))
