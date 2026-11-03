"""Connect the public closing result to its source, training inputs and native scores."""

from pathlib import Path
import json
import math
import subprocess
import sys

import numpy as np

from selffold.conditional_curriculum import make_cases
from selffold.two_flags import ARM_NAMES, training_pairs
from selffold.records import file_sha, digest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "provenance/two_flags_20261104"
RUN = STUDY / "run"


def read(p):
    return json.loads(p.read_text())


def lines(p):
    return [json.loads(l) for l in p.read_text().splitlines()]


def test_public_two_flags_decisions_and_registered_gates_recompute():
    prefix = STUDY.relative_to(ROOT).as_posix()
    files = {
        s[len(prefix) + 1 :]
        for s in subprocess.check_output(
            ["git", "ls-files", "-z", "--", prefix], cwd=ROOT
        )
        .decode()
        .split("\0")
        if s
    }
    checks = (STUDY / "SHA256SUMS").read_text().splitlines()
    hashes = {name: h for h, name in (l.split("  ", 1) for l in checks)}
    assert len(hashes) == len(checks) and set(hashes) == files - {"SHA256SUMS"}
    assert all(file_sha(STUDY / n) == h for n, h in hashes.items())
    assert json.loads(
        subprocess.check_output([sys.executable, "-S", str(STUDY / "summarize.py")])
    ) == read(STUDY / "summary.json")
    end, src, request = map(
        read, (RUN / "terminal.json", RUN / "source.json", RUN / "request.json")
    )
    assert (
        src["git_commit"] == "9913c3ffd2c827e04faaa32076435f10522cebda"
        and src["git_status"] == ""
    )
    assert request["plan_sha256"] == file_sha(STUDY / "PLAN.md") and request[
        "scorer_sha256"
    ] == file_sha(STUDY / "summarize.py")
    assert (
        end["status"] == "COMPLETED_TWO_FLAGS"
        and end["backbone_before"] == end["backbone_after"]
    )
    assert (
        end["G_trained"] is False
        and end["qualification_content_loaded"] is False
        and read(RUN / "native_parity.json")["exact"]
    )
    assert read(RUN / "fit_closed.json") == {
        "both_endpoints_saved": True,
        "fresh_scores": 0,
    }


def test_two_flags_training_and_endpoint_bindings():
    base = read(ROOT / "data/conditional_margin_20261103.json")
    design = read(ROOT / "data/two_flags_20261104.json")
    groups = {
        g["id"]: g
        for g in read(ROOT / "data/within_topic_wording_20261103.json")["groups"]
    }
    fit = make_cases(list(groups.values()), base, "fit")
    by = {
        (
            c["group"],
            c["stage"],
            c["flag_pair"],
            c["style"],
            c["source_order"],
            c["pole"],
        ): c
        for c in fit
    }
    schedule = read(RUN / "schedule.json")
    request = read(RUN / "request.json")
    assert digest(schedule) == request["schedule_sha256"]
    assert request["fixture_sha256"] == file_sha(
        ROOT / "data/two_flags_20261104.json"
    ) and request["base_sha256"] == file_sha(
        ROOT / "data/conditional_margin_20261103.json"
    )
    assert request["actions_sha256"] == file_sha(
        ROOT / "data/within_topic_wording_20261103.json"
    )
    parent = ROOT / "provenance/conditional_margin_20261103/run"
    starts = lines(RUN / "starts.jsonl")
    fits = lines(RUN / "fits.jsonl")
    assert [r["arm"] for r in starts] == [r["arm"] for r in fits] == list(ARM_NAMES)
    assert request["parent_sha256"] == file_sha(parent / "warm_task.npz")
    for start, end in zip(starts, fits, strict=True):
        assert (
            start["artifact_sha256"] == request["parent_sha256"]
            and start["parameters_sha256"]
            == read(parent / "warm_parameters.json")["sha256"]
        )
        file = RUN / (end["arm"] + ".npz")
        assert end["native_logits_exact"] and file_sha(file) == end["artifact_sha256"]
        with np.load(file, allow_pickle=False) as z:
            meta = json.loads(z["metadata_utf8"].tobytes())
            a = meta["architecture"]
            assert (
                a["rank"] == 16
                and a["scope"] == "explicit_task_context_only"
                and a["last_layers"] == 4
            )
            assert (
                meta["base_host_sha256"] == digest(read(RUN / "host.json"))
                and meta["dataset_sha256"] == request["fixture_sha256"]
            )
            assert (
                meta["source_sha256"] == read(RUN / "source.json")["source_sha256"]
                and meta["training"]["schedule_sha256"] == request["schedule_sha256"]
                and meta["training"]["parent_sha256"] == request["parent_sha256"]
            )
            assert all(
                z[k].dtype == np.float32 and np.isfinite(z[k]).all()
                for k in z.files
                if k != "metadata_utf8"
            )
    for row in lines(RUN / "training.jsonl"):
        step = row["step"] - 1
        bind = []
        assert (
            row["units"] == schedule[step]
            and math.isfinite(row["loss"])
            and math.isfinite(row["gradient_norm"])
        )
        assert 0 < row["unmasked_tokens"] <= row["padded_tokens"]
        for u, unit in enumerate(schedule[step]):
            for pole in (0, 1):
                c = by[(*unit, pole)]
                for w in range(4):
                    task, a, b = training_pairs(
                        c, groups[c["group"]], base, design, step, u, w
                    )
                    name, ts = ("action", a) if row["arm"] == "direct" else (task, b)
                    bind.append(
                        (
                            c["key"],
                            w,
                            name,
                            [(t["source"], t["question"], t["gold"]) for t in ts],
                        )
                    )
        assert digest(bind) == row["bindings_sha256"]
