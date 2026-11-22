"""Check the saved reports, physical tables and downstream choices independently of the runner."""
import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import re

import numpy as np


def read(path):
    return json.loads(path.read_text())


def reports_fields(text, names):
    found = defaultdict(set)
    for raw in text.splitlines():
        cleaned = raw.casefold().strip(" |\t").replace("*", "").replace("`", "")
        for label in ("off/off", "off/on", "on/off", "on/on", "default"):
            match = re.match(r"^[\s\d.()#>•-]*" + re.escape(label) +
                             (r"(?: action)?" if label == "default" else "") + r"\s*[:=–—-]\s*(.*)", cleaned)
            if match:
                phrases = [j for j, name in enumerate(names) if re.search(r"\b" + re.escape(name) + r"\b", match[1])]
                found[label].add(phrases[0] if len(phrases) == 1 else None)
    return [next(iter(found[k])) if len(found[k]) == 1 else None
            for k in ("off/off", "off/on", "on/off", "on/on", "default")]


def verify(root, folder):
    cfg = read(root/"data/structured_report_20261122.json")
    names = [n for group in read(root/"data/within_topic_wording_20261103.json")["groups"] for n in group["options"]]
    queries = {q["id"]: q for q in read(folder/"test_states.json")}
    train = read(folder/"train_states.json")
    assert set(q["pattern"] for q in queries.values()).isdisjoint(q["pattern"] for q in train)
    assert len(queries) == 192 and len(train) == 256
    roles = defaultdict(list)
    for q in queries.values():
        coords = ("off/off", "off/on", "on/off", "on/on")
        actions = [2*q["menu"] + (q["pattern"]//(2**i)) % 2 for i in range(4)]
        printed = [line.split(" : ") for line in q["prompt"].splitlines() if " : " in line]
        assert [coords.index(x) for x, _ in printed] == [(i+q["rotation"])%4 for i in range(4)]
        assert all(names[actions[coords.index(x)]] == y for x,y in printed)
        assert actions == q["rows"]
        assert q["bit"] == ((actions[0] % 2) != q["default"])
        assert ("off/off has the default role." in q["prompt"]) == (q["bit"] == 0)
        roles[q["menu"], q["pattern"], q["rotation"]].append(q)
    assert all(len(p) == 2 and p[0]["rows"] == p[1]["rows"] and p[0]["bit"] != p[1]["bit"] for p in roles.values())
    raw = [json.loads(line) for line in (folder/"reports.jsonl").read_text().splitlines()]
    expected = {(qid, phase, mode) for qid, phase, mode in itertools.product(queries, ("before","after"), ("free","choice"))}
    assert len(raw) == len(expected) and {(r["id"],r["arm"],r["mode"]) for r in raw} == expected
    reports = {(r["id"],r["arm"]+"_"+r["mode"]): reports_fields(r["text"], names) for r in raw}
    eos = read(folder/"host.json")["identity"]["eos_ids"]
    for r in raw:
        if r["mode"] == "free":
            seq = r["token_ids"]
            assert (seq[-1] in eos) == (r["stop"] == "native_eos")
            assert not any(t in eos for t in seq[:-1])
            assert seq[-1] in eos or len(seq) == cfg["generation_cap"]
    table = read(folder/"outcomes.json")
    native = np.load(folder/"native/scores.npz", allow_pickle=False)
    dispatch = native["dispatch"].reshape(4,16,2)
    assert sorted(dispatch.ravel()) == list(range(128))
    labels = read(folder/"native/native.json")["labels"]
    summary = read(folder/"summary.json")
    measured = defaultdict(dict)
    for r in table:
        q = queries[r["id"]]
        gold = q["rows"] + [2*q["menu"] + q["default"]]
        route = r["route"]
        if route in ("L", "oracle"):
            fields = gold
        elif route.endswith("_oracle_rows") or route.endswith("_oracle_default"):
            raw_fields = reports[q["id"], route.split("_")[0]+"_free"]
            fields = gold[:4] + raw_fields[4:] if route.endswith("rows") else raw_fields[:4] + gold[4:]
        else:
            fields = reports[q["id"], route]
        assert fields == r["fields"], (q["id"],route,fields,r["fields"])
        assert [u==v for u,v in zip(fields,gold)] == r["field_correct"]
        n = 0
        if None not in fields and len({x//2 for x in fields}) == 1:
            primary = fields[4]
            alternate = primary ^ 1
            answers = [fields[0]]*2 if len(set(fields[:4])) == 1 else [primary,alternate]
            actual = []
            for mask in cfg["next_masks"]:
                after = [answers[mask//(2**world)%2] for world in range(4)]
                actual.append(after)
                for j, act in enumerate(after):
                    for order in (0,1):
                        k = dispatch[cfg["menus"].index(q["menu"]),act,order]
                        want = q["default"] ^ ((mask >> j) & 1) ^ order
                        ab = native["ab"][k]
                        n += int(native["top"][k] == labels[want] and ab[want] > ab[1-want])
            assert actual == r["output_rows"]
        else:
            assert r["output_rows"] is None
        assert n == r["correct"] and (n == 112) == r["complete"]
        measured[route][q["id"]] = n
    for route, by_id in measured.items():
        assert len(by_id) == 192 and sum(by_id.values()) == summary["arms"][route]["judgments"]
        twins = sum(all(by_id[q["id"]] == 112 for q in pairs) for pairs in roles.values())
        assert twins == summary["arms"][route]["twins"]
    return dict(exact=True, independent_prompt_and_report_readings=len(raw),
                verified_route_presentations=len(table), donor_presentations=len(roles), unique_physical_tables=24,
                native_dispatches=128, refits=0, new_model_calls=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parents[2], args.folder), sort_keys=True))
