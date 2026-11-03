"""Check real training, panel/target bindings, ordered stage operations and native readouts."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from selffold.records import State, file_sha
from selffold.stage_transitions import StageMap, target_preference

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/stage_grammar_20261104"


def read(p):
    return json.loads(p.read_text())


def test_registered_stage_reports_recompute_and_share_training_budgets():
    tracked = {
        str((ROOT / n.decode()).relative_to(STUDY))
        for n in subprocess.check_output(
            ["git", "ls-files", "-z", "--", str(STUDY.relative_to(ROOT))], cwd=ROOT
        ).split(b"\0")
        if n
    }
    hashes = {
        n: h
        for h, n in (
            line.split("  ", 1)
            for line in (STUDY / "SHA256SUMS").read_text().splitlines()
        )
    }
    assert set(hashes) == tracked - {"SHA256SUMS"} and all(
        file_sha(STUDY / n) == h for n, h in hashes.items()
    )
    for script, folder, target in (
        ("summarize.py", "literal", "literal-summary.json"),
        ("score_chain.py", "maps", "maps-summary.json"),
        ("diagnose.py", "literal", "diagnostic.json"),
    ):
        assert json.loads(
            subprocess.check_output(
                [sys.executable, str(STUDY / script), str(STUDY / folder)]
            )
        ) == read(STUDY / target)
    run = STUDY / "literal"
    fits = read(run / "fit_closed.json")
    assert fits["fresh_scores"] == 0 and len(fits["candidates"]) == 3
    assert len({x["initial_parameters"] for x in fits["candidates"]}) == 1
    for a in fits["candidates"]:
        assert a["replayed"] and a["weights"] == file_sha(run / (a["arm"] + ".npz"))
    summary = read(STUDY / "literal-summary.json")
    assert len({v["padded"] for v in summary["cost"].values()}) == 1
    for a, v in summary["cost"].items():
        assert v["forwards"] == 3200
        assert v["tasks"] == (
            {"action": 2400, "bind": 5600, "and": 4000, "flag": 800}
            if a == "focus"
            else dict.fromkeys(("action", "bind", "and", "flag"), 3200)
        )
    for folder, seal in (
        ("literal", "5fac1bd10591b6227ec1c6fb6cdfcef989a7cb30"),
        ("maps", "b8c4511b9334b3e4b5dd69235f2e757e4a595b06"),
    ):
        assert read(STUDY / folder / "source.json")["git_commit"] == seal
        assert (
            read(STUDY / folder / "terminal.json")["backbone"]
            == "2690ac5991d97f7caf539b457d3de686949626739480b7d228bcdad8609b9e92"
        )
    assert read(STUDY / "maps/request.json")["reader"] == file_sha(run / "focus.npz")


def test_two_transitions_and_midpoint_donors_reconstruct_from_frozen_arrays():
    run = STUDY / "maps"
    closed = read(run / "maps_closed.json")
    assert (
        closed["source_cases"],
        closed["g2_pairs"],
        closed["rule_decoder_cases"],
        closed["fresh_captures"],
    ) == (384, 768, 192, 0)
    meta = read(run / "state_index.json")
    target = read(run / "targets.json")
    with np.load(run / "fresh_states.npz", allow_pickle=False) as ar:
        rows = torch.from_numpy(ar["states"]).view(torch.bfloat16)
    assert {State(slots=z).sha() for z in rows} == set(meta["sha256"])
    rows_by = {k: rows[v] for k, v in meta["routes"].items()}
    assert torch.equal(rows_by["both"][target["inverse"]], rows_by["opposed"])
    assert torch.equal(rows_by["both"][target["equivalent"]], rows_by["equivalent"])
    with np.load(run / "maps.npz", allow_pickle=False) as ar:
        models = {
            n: StageMap(
                **{
                    k: torch.from_numpy(ar[n + "_" + k])
                    for k in StageMap.__annotations__
                }
            )
            for n in ("g1", "g2", "decoder")
        }
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for model, src, dst in (
            ("g1", "source", "g1"),
            ("g2", "g1", "both"),
            ("g2", "source", "g2_only"),
        ):
            pred = models[model].slots(rows_by[src]).float()
            stored = rows_by[dst].float()
            torch.testing.assert_close(pred, stored, atol=0.035, rtol=0.01)
            assert float((pred - stored).square().mean()) < 1e-5
        for k, p in target["predicted"].items():
            assert models["decoder"](rows_by[k]).argmax(-1).tolist() == p
        for tag, src, dst in (
            ("g1_case", "g1", "canonical_case"),
            ("z0_case", "source", "canonical_case"),
            ("full_rule", "both", "rule_teacher"),
            ("g1_rule", "g1", "rule_teacher"),
            ("g2_only_rule", "g2_only", "rule_teacher"),
        ):
            ref = rows_by[dst]
            gap = target_preference(rows_by[src], ref, ref[target["inverse"]]).numpy()
            np.testing.assert_allclose(
                gap, target["target_gap"][tag], atol=1e-6, rtol=1e-4
            )
    finally:
        torch.set_num_threads(previous)


def test_native_objective_reached_both_maps_and_reader_remained_frozen():
    base = STUDY / "maps"
    g = STUDY / "guided"
    receipt = read(g / "guide_closed.json")
    assert receipt["parent_maps"] == file_sha(base / "maps.npz") and receipt[
        "reader_sha256"
    ] == file_sha(STUDY / "literal/focus.npz")
    assert (
        receipt["added_parameters"] == 589824
        and receipt["stage_labels"] == 12288
        and receipt["updates"] == 192
        and receipt["fresh_captures"] == 0
    )
    assert (
        read(base / "terminal.json")["reader_parameters"]
        == receipt["frozen_reader_parameters"]
        == read(g / "terminal.json")["reader_parameters"]
    )
    assert (
        read(g / "terminal.json")["backbone"]
        == read(base / "terminal.json")["backbone"]
    )
    assert (
        read(g / "source.json")["git_commit"]
        == "beb453d9c6990ab59db9f00c7fd6c3554493286e"
    )
    train = [
        json.loads(line)
        for line in (g / "guide_training.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in train] == list(range(1, 193)) and sum(
        r["forwards"] for r in train
    ) == 3072
    assert all(
        np.isfinite(r["mid_to_final_gradient"])
        and r["mid_to_final_gradient"] > 0
        and min(r["residual_gradients"]) > 0
        for r in train
    )
    assert read(g / "maps_closed.json")["rank"] == {"g1": 40, "g2": 40, "decoder": 15}
    with (
        np.load(g / "maps.npz", allow_pickle=False) as fresh,
        np.load(base / "maps.npz", allow_pickle=False) as oldmaps,
    ):
        for name in ("g1", "g2"):
            for t in ("center", "scale", "target"):
                assert np.array_equal(fresh[name + "_" + t], oldmaps[name + "_" + t])
            assert np.array_equal(
                fresh[name + "_down"][:, :32], oldmaps[name + "_down"]
            )
            assert np.array_equal(fresh[name + "_up"][:32], oldmaps[name + "_up"])
            assert np.any(fresh[name + "_up"][32:] != 0)
    assert json.loads(
        subprocess.check_output([sys.executable, str(STUDY / "score_chain.py"), str(g)])
    ) == read(STUDY / "guided-summary.json")
    for mode in ("maps", "guided"):
        replay = read(STUDY / ("replay-" + mode) / "receipt.json")
        assert replay["exact_native_decisions"] == 1024 and replay[
            "raw_digest"
        ] == file_sha(STUDY / mode / "decisions.jsonl.gz")


def test_crossed_bit_reports_and_historical_batch_replays():
    import gzip
    from collections import Counter, defaultdict
    from selffold.conditional_curriculum import make_cases
    from selffold.stage_grammar import fixture
    from selffold.two_flags import branch_probe
    from selffold.records import digest

    data = ROOT / "data"
    base, prev, design, actions = [
        read(data / x)
        for x in (
            "conditional_margin_20261103.json",
            "two_flags_20261104.json",
            "stage_grammar_20261104.json",
            "within_topic_wording_20261103.json",
        )
    ]
    groups = actions["groups"]
    file = STUDY / "wording"
    summ = read(file / "cross.json")
    labels = {v: k for k, v in read(STUDY / "literal/labels.json").items()}
    with gzip.open(file / "cross.jsonl.gz", "rt") as io:
        for tag in sorted(summ):
            grammar, bitword, question = map(int, tag)
            f = fixture(base, prev, design, "fresh")
            if not grammar:
                f["fresh"]["styles"] = base["fit"]["styles"][:1]
            if not question:
                f["fresh"].update(
                    question=base["fit"]["question"], operations=base["operations"]
                )
            words = design["fresh_probes"] if bitword else prev["fit_probes"]
            by = defaultdict(list)
            bits = defaultdict(list)
            for case in make_cases(groups, f, "fresh"):
                for b in (False, True):
                    for t in branch_probe(case, groups[case["topic"]], f, words, b):
                        r = json.loads(next(io))
                        assert r[:5] == [
                            tag,
                            case["key"],
                            int(b),
                            t["label_order"],
                            t["gold"],
                        ]
                        assert r[5] == labels.get(r[6]) and r[9:] == [
                            digest(t["source"]),
                            digest(t["question"]),
                        ]
                        ok = r[5] == t["gold"]
                        bits[case["stage"], int(b)].append(ok)
                        by[case["stage"], case["group"], case["flag_pair"]].append(ok)
            for st, s in summ[tag].items():
                for b, k in ((0, "false_bit"), (1, "true_bit")):
                    v = bits[st, b]
                    assert s[k] == [sum(v), len(v)]
                assert s["strict"] == [
                    sum(all(v) for k, v in by.items() if k[0] == st),
                    16,
                ]
        assert not io.read()
    batch = read(file / "batch.json")
    counts = Counter()
    with gzip.open(file / "batch.jsonl.gz", "rt") as io:
        for line in io:
            _, _, _, arch, prior, full, thin, new = json.loads(line)
            assert arch == prior and full == thin == new
            counts["n"] += 1
            counts["changed"] += int(prior[0] != full[0])
    assert (
        counts == {"n": 1280, "changed": 8}
        and batch["old4_vs_new8_full"]["winner_changes"] == 8
    )
