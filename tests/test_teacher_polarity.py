from collections import Counter
import json
from pathlib import Path

import numpy as np

from selffold.polarity_readout import fit_axes, predicted_class, read_axes, source_cases
from selffold.records import file_sha

ROOT = Path(__file__).resolve().parents[1]


def test_polarity_fixture_varies_only_grammar_and_balances_both_opposed_exchanges():
    f = json.loads((ROOT / "data/teacher_polarity_20261103.json").read_text())
    p = ROOT / f["prior_fixture"]
    assert file_sha(p) == f["prior_sha256"]
    old = json.loads(p.read_text())
    forms = {s: old["styles"][s] for s in old["fit_styles"]}
    fit, fresh = (
        list(source_cases(old["groups"], forms_, f["familiar_pairs"]))
        for forms_ in (forms, f["fresh_frames"])
    )
    assert len(set(c["source"] for c in fit)) == len(fit) == 512
    assert len(set(c["source"] for c in fresh)) == len(fresh) == 256
    assert {c["source"] for c in fit}.isdisjoint(c["source"] for c in fresh)
    assert set(forms).isdisjoint(f["fresh_frames"]) and len(old["groups"]) == 8
    exposed = {q for g in old["groups"] for q in g["questions"]}
    for g in old["groups"]:
        new = f["generic_questions"] + f["specific_questions"][g["id"]]
        assert len(set(new)) == 4 and exposed.isdisjoint(new)
    for bank in (fit, fresh):
        idx = {
            (c["group"], c["frame"], c["pair"], c["pole"], c["order"]): c["source"]
            for c in bank
        }
        for (g, frame, pair, p, o), text in idx.items():
            assert text.split(". ") != idx[g, frame, pair, p, 1 - o].split(". ")
            assert (
                text[:-1].split(". ")
                == idx[g, frame, pair, p, 1 - o][:-1].split(". ")[::-1]
            )
            bag = lambda s: Counter(s.lower().replace(".", "").replace(",", "").split())
            assert (
                bag(text)
                == bag(idx[g, frame, pair, 1 - p, o])
                == bag(idx[g, frame, pair, 1 - p, 1 - o])
            )
            group = next(x for x in old["groups"] if x["id"] == g)
            words = (group["plus"][pair], group["minus"][pair])
            first_is_plus = lambda s: s.index(words[0]) < s.index(words[1])
            assert first_is_plus(text) == first_is_plus(
                idx[g, frame, pair, 1 - p, 1 - o]
            )
            assert first_is_plus(text) != first_is_plus(idx[g, frame, pair, 1 - p, o])


def test_factor_reader_solves_crossed_role_action_toy_without_topic_or_pair_side_input():
    cs, values = [], []
    for t in (0, 1):
        for p in (0, 1):
            for o in (0, 1):
                cs.append({"topic": t, "pole": p, "order": o})
                values.append(
                    [
                        5 * (1 - 2 * t),
                        (1 - 2 * (p ^ o)) * (t == 0),
                        (1 - 2 * (p ^ o)) * (t == 1),
                        1 - 2 * o,
                    ]
                )
    x = np.array(values, dtype=np.float32)
    axes = fit_axes(x, cs, 2)
    assert set(axes) == {"mean", "scale", "shared", "offset", "slopes", "biases"}
    for c, state in zip(cs, x, strict=True):
        read = read_axes(state, axes)
        assert predicted_class(read) == 2 * c["topic"] + c["pole"]
        assert predicted_class(read, "mention") == 2 * c["topic"] + (
            c["pole"] ^ c["order"]
        )
        assert predicted_class(read | {"role": 0.0}) is None
        assert predicted_class(read | {"topic": None}) is None
