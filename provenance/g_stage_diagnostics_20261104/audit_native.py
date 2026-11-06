"""Paired archival audit of native G; NumPy/stdlib only, no model or qualification data.

The six-field case key is valid within a source-wording panel.  It does not make
different physical readers, prompts, or wordings interchangeable.  Raw and
structured results below are explicitly a descriptive semantic overlap.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


BASE_COMMIT = "ba20b3125ac498ef1f2fa21730eef3557e3dd953"
STUDY = Path("provenance/supported_g_20261104")
LABELS = {"A": 235280, "B": 235305}
OPERATORS = ("plain", "decompose", "boundary", "reintegrate")
FREE_ROUTES = ("D_initial", "D_updated", "ED_initial", "ED_updated")
FULL_ROUTES = (
    "identity", "teacher", *OPERATORS, "opposed", "equivalent", "unrelated",
    "zero", "empty", "random", "permuted", "norm_only", "source_prefix",
    "reordered", "question_prefix", "literal",
)
INPUTS = (
    "data/supported_g_20261104.json",
    "data/within_topic_wording_20261103.json",
    "src/selffold/supported_boundary.py",
    "src/selffold/host.py",
    "src/selffold/transport.py",
    "scripts/raw_supported_boundary.py",
    "scripts/fit_continuity_recipient.py",
    "scripts/state_only_native.py",
    "scripts/structured_supported_boundary.py",
    str(STUDY / "raw-bound/sources.jsonl"),
    str(STUDY / "raw-bound/pairs.jsonl"),
    str(STUDY / "raw-bound/decisions.jsonl.gz"),
    str(STUDY / "raw-bound/free_text.jsonl"),
    str(STUDY / "raw-bound/transport.npz"),
    str(STUDY / "raw-bound/binding.json"),
    str(STUDY / "raw-bound/parity.json"),
    str(STUDY / "raw-bound/terminal.json"),
    str(STUDY / "raw-summary.json"),
    str(STUDY / "structured/states.npz"),
    str(STUDY / "structured/native.npz"),
    str(STUDY / "structured/native.json"),
    str(STUDY / "structured/dispatch.npz"),
    str(STUDY / "structured/literal.npz"),
    str(STUDY / "structured/binding.json"),
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def ledger(key, world):
    """Independent four-world bit-mask reference, not a model prediction."""
    masks = (0b0011, 0b1100, 0b0101, 0b1010)
    trigger = bool((masks[key[1]] >> world) & 1)
    veto = bool(((0 if key[2] < 0 else masks[key[2]]) >> world) & 1)
    return key[3] ^ int(trigger and not veto), trigger and veto


def load_native(repo: Path) -> dict:
    """Return audited native rows keyed by (*case_key, world, label_order).

    Only explicit supported-G development paths are opened.  A missing route
    cell is absent from the dictionary; it is never assigned a zero score.
    """
    repo = Path(repo)
    root = repo / STUDY / "raw-bound"
    source_list = [json.loads(line) for line in (root / "sources.jsonl").read_text().splitlines()]
    keys = [tuple(s["key"]) for s in source_list]
    if len(keys) != 128 or len(set(keys)) != len(keys):
        raise ValueError("Unexpected or duplicated native source keys")
    sources = dict(zip(keys, source_list, strict=True))
    # This module is pure Python and restricts its generator to train/fresh.
    sys.path.insert(0, str(repo / "src"))
    from selffold.supported_boundary import Case, cases, questions

    cfg = read(repo / "data/supported_g_20261104.json")
    menus = [g["options"] for g in read(repo / "data/within_topic_wording_20261103.json")["groups"]]
    expected_keys = [c.key for c in cases(cfg, "fresh", raw=True)]
    if keys != expected_keys:
        raise ValueError("Native archive differs from its registered development panel")
    question_hashes = {}
    for key, source in sources.items():
        c = Case(*key)
        if digest(c.source(menus, cfg["fresh"])) != source["source"]:
            raise ValueError("Source wording hash mismatch")
        if c.ledger() != source["gold"]:
            raise ValueError("Source ledger mismatch")
        for prefix_tag, prefix in (("ordinary", ""), ("prefixed", "For this decision, use the stated policy. ")):
            for q in questions(c, menus, prefix=prefix):
                question_hashes[(key, q["world"], q["label_order"], prefix_tag)] = digest(q["question"])

    rows = {tag: {} for tag in (*FULL_ROUTES, *FREE_ROUTES)}
    with gzip.open(root / "decisions.jsonl.gz", "rt") as stream:
        for r in map(json.loads, stream):
            if len(r) != 15:
                raise ValueError("Unexpected native decision schema")
            key = tuple(r[:6])
            tag, world, order, gold, predicted, token, a, b, qhash = r[6:]
            cell = (*key, world, order)
            if key not in sources or tag not in rows or cell in rows[tag]:
                raise ValueError("Unknown or duplicated native decision")
            if world not in range(4) or order not in (0, 1):
                raise ValueError("Invalid world/answer order")
            expected_gold = "AB"[ledger(key, world)[0] ^ order]
            if gold != expected_gold:
                raise ValueError("Decision disagrees with independent Boolean ledger")
            prefix_tag = "prefixed" if tag == "question_prefix" else "ordinary"
            if qhash != question_hashes[(key, world, order, prefix_tag)]:
                raise ValueError("Question hash mismatch")
            if predicted in LABELS and token != LABELS[predicted]:
                raise ValueError("Greedy label and native token disagree")
            valid = a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
            target = ("B" if gold == "A" else "A") if tag == "opposed" else gold
            margin = a - b if valid else None
            rows[tag][cell] = dict(
                case_key=list(key), route=tag, world=world, label_order=order,
                gold=gold, target_label=target, predicted_label=predicted, token=token,
                logprob_A=a, logprob_B=b, margin=margin,
                tied=bool(valid and margin == 0), valid_scores=valid,
                correct=bool(valid and margin != 0 and predicted == target),
                question_sha256=qhash, present=True,
            )
    all_cells = {(*k, w, o) for k in keys for w in range(4) for o in (0, 1)}
    for tag in FULL_ROUTES:
        if set(rows[tag]) != all_cells:
            raise ValueError(f"Missing full-panel cells for {tag}")
    free_list = [json.loads(line) for line in (root / "free_text.jsonl").read_text().splitlines()]
    free = {(*r["key"], r["stage"]): r for r in free_list}
    free_keys = {k for k in keys if k[0] in (0, 1) and k[1] == k[0] and k[5] == k[4]}
    expected_free = {(*k, stage) for k in free_keys for stage in ("initial", "updated")}
    if len(free) != len(free_list) or set(free) != expected_free:
        raise ValueError("Free-generation panel differs from registered subset")
    free_cells = {(*k, w, o) for k in free_keys for w in range(4) for o in (0, 1)}
    for tag in FREE_ROUTES:
        if set(rows[tag]) != free_cells:
            raise ValueError(f"Missing free-subset cells for {tag}")
    for key in free_keys:
        for stage, i in (("initial", 0), ("updated", 1)):
            if free[*key, stage]["state"] != sources[key]["hashes"][i]:
                raise ValueError("Generation state hash mismatch")
    return dict(keys=keys, sources=sources, rows=rows, free=free,
                free_keys=sorted(free_keys), free_cells=sorted(free_cells))


def count(rows, cells=None):
    selected = list(rows.values()) if cells is None else [rows[k] for k in cells if k in rows]
    labels = Counter(r["predicted_label"] if r["predicted_label"] is not None else "missing" for r in selected)
    cells_by_policy = {}
    by_world = {}
    for r in selected:
        key = tuple(r["case_key"])
        cells_by_policy.setdefault(key, []).append(r)
        by_world.setdefault((*key, r["world"]), []).append(r)
    return dict(
        n=len(selected), correct=sum(r["correct"] for r in selected),
        predicted_labels=dict(labels), ties=sum(r["tied"] for r in selected),
        invalid_scores=sum(not r["valid_scores"] for r in selected),
        categorical_correct_ignoring_ties=sum(r["predicted_label"] == r["target_label"] for r in selected),
        positive_margin=sum(r["valid_scores"] and r["margin"] > 0 for r in selected),
        negative_margin=sum(r["valid_scores"] and r["margin"] < 0 for r in selected),
        complete_policies=sum(len(v) == 8 and all(r["correct"] for r in v) for v in cells_by_policy.values()),
        whole_policy_denominator=sum(len(v) == 8 for v in cells_by_policy.values()),
        both_answer_orders_correct=sum(len(v) == 2 and all(r["correct"] for r in v) for v in by_world.values()),
        answer_order_pair_denominator=sum(len(v) == 2 for v in by_world.values()),
        answer_order_semantic_consistency=sum(
            len(v) == 2 and all(r["predicted_label"] in LABELS for r in v)
            and len({"AB".index(r["predicted_label"]) ^ r["label_order"] for r in v}) == 1
            for v in by_world.values()
        ),
        by_label_order={str(o): dict(
            n=sum(r["label_order"] == o for r in selected),
            correct=sum(r["label_order"] == o and r["correct"] for r in selected),
            categorical_correct_ignoring_ties=sum(r["label_order"] == o and r["predicted_label"] == r["target_label"] for r in selected),
            ties=sum(r["label_order"] == o and r["tied"] for r in selected),
        ) for o in (0, 1)},
    )


def pair(left, right, cells=None):
    keys = sorted(set(left) & set(right)) if cells is None else [k for k in cells if k in left and k in right]
    for k in keys:
        if left[k]["target_label"] != right[k]["target_label"]:
            raise ValueError("A paired transition needs identical target labels")
        if left[k]["question_sha256"] != right[k]["question_sha256"]:
            raise ValueError("A paired transition needs identical question wording")
    quadrants = Counter((left[k]["correct"], right[k]["correct"]) for k in keys)
    repaired = [k for k in keys if not left[k]["correct"] and right[k]["correct"]]
    spoiled = [k for k in keys if left[k]["correct"] and not right[k]["correct"]]
    return dict(
        n=len(keys), both_correct=quadrants[True, True], repaired=len(repaired),
        spoiled=len(spoiled), both_wrong=quadrants[False, False],
        net_correct=len(repaired) - len(spoiled),
        predicted_label_changes=sum(left[k]["predicted_label"] != right[k]["predicted_label"] for k in keys),
        greedy_token_changes=sum(left[k]["token"] != right[k]["token"] for k in keys),
        ab_scores_identical=sum((left[k]["logprob_A"], left[k]["logprob_B"]) == (right[k]["logprob_A"], right[k]["logprob_B"]) for k in keys),
        margins_identical=sum(left[k]["margin"] == right[k]["margin"] for k in keys),
        strict_margin_reversals=sum(left[k]["valid_scores"] and right[k]["valid_scores"] and left[k]["margin"] * right[k]["margin"] < 0 for k in keys),
        ties_resolved=sum(left[k]["tied"] and not right[k]["tied"] and right[k]["valid_scores"] for k in keys),
        ties_created=sum(not left[k]["tied"] and left[k]["valid_scores"] and right[k]["tied"] for k in keys),
        repairs_from_ties_without_label_change=sum(left[k]["tied"] and left[k]["predicted_label"] == right[k]["predicted_label"] for k in repaired),
        spoils_from_ties_without_label_change=sum(right[k]["tied"] and left[k]["predicted_label"] == right[k]["predicted_label"] for k in spoiled),
        repairs_on_affected_worlds=sum(ledger(k[:6], k[6])[1] for k in repaired),
        spoils_on_affected_worlds=sum(ledger(k[:6], k[6])[1] for k in spoiled),
        repaired_cells=[list(k) for k in repaired], spoiled_cells=[list(k) for k in spoiled],
    )


def structured_overlap(repo, data, subset):
    root = repo / STUDY / "structured"
    with np.load(root / "states.npz", allow_pickle=False) as states:
        lookup = {tuple(map(int, k)): i for i, k in enumerate(states["keys"])}
    indices = np.array([lookup[k] for k in subset])
    gold = np.array([[[LABELS["AB"[ledger(k, w)[0] ^ o]] for o in (0, 1)]
                      for w in range(4)] for k in subset])
    result = {}
    with (np.load(root / "native.npz", allow_pickle=False) as n,
          np.load(root / "dispatch.npz", allow_pickle=False) as d,
          np.load(root / "literal.npz", allow_pickle=False) as l):
        for tag in ("identity", "analytic", "updated", "ED_initial", "ED_updated", "D_initial", "D_updated"):
            archive = l if tag.startswith("D_") else n
            dispatch = l[tag][indices] if tag.startswith("D_") else d[tag][indices]
            top, ab = archive["top"][dispatch], archive["ab"][dispatch]
            correct = (top == gold) & np.isfinite(ab).all(-1) & (ab[..., 0] != ab[..., 1])
            result[tag] = dict(n=int(correct.size), correct=int(correct.sum()),
                               complete_policies=int(correct.all((1, 2)).sum()),
                               whole_policy_denominator=len(subset))
    cells = [(*k, w, o) for k in subset for w in range(4) for o in (0, 1)]
    return dict(
        n_policies=len(subset), n_decisions=len(cells),
        structured=result,
        native={tag: count(data["rows"][tag], cells) for tag in ("identity", *OPERATORS, "teacher", *FREE_ROUTES)},
    )


def analyze(repo: Path) -> dict:
    repo = Path(repo)
    data = load_native(repo)
    rows, keys, sources = data["rows"], data["keys"], data["sources"]
    root = repo / STUDY / "raw-bound"
    routes = {tag: count(values) for tag, values in rows.items()}
    archived = read(repo / STUDY / "raw-summary.json")["routes"]
    if any((routes[tag]["n"], routes[tag]["correct"]) != (archived[tag]["n"], archived[tag]["score"]) for tag in routes):
        raise ValueError("Recomputed native totals differ from frozen summary")
    changes = {k for k in keys if any(ledger(k, w)[1] for w in range(4))}
    stable = set(keys) - changes
    paired = {f"identity_to_{tag}": pair(rows["identity"], rows[tag]) for tag in (*OPERATORS, "teacher")}
    paired["boundary_to_teacher"] = pair(rows["boundary"], rows["teacher"])
    for tag in FREE_ROUTES:
        paired[f"boundary_to_{tag}_free_subset"] = pair(rows["boundary"], rows[tag])
    for a, b in (("D_initial", "D_updated"), ("ED_initial", "ED_updated"), ("D_updated", "ED_updated")):
        paired[f"{a}_to_{b}"] = pair(rows[a], rows[b])
    subset_scores = {}
    for name, subset in (("change_required_policies", changes), ("stable_policies", stable)):
        cells = [(*k, w, o) for k in sorted(subset) for w in range(4) for o in (0, 1)]
        subset_scores[name] = {tag: count(rows[tag], cells) for tag in ("identity", "boundary", "teacher")}
    affected = [cell for cell in rows["identity"] if ledger(cell[:6], cell[6])[1]]
    subset_scores["affected_worlds"] = {tag: count(rows[tag], affected) for tag in ("identity", "boundary", "teacher")}
    opposed_keys = {k: (k[0], k[1], k[2], 1 - k[3], k[4], k[5]) for k in keys}
    state_evidence = dict(
        n=128, identity_boundary_hash_different=sum(s["hashes"][0] != s["hashes"][1] for s in sources.values()),
        unique_identity_hashes=len({s["hashes"][0] for s in sources.values()}),
        unique_boundary_hashes=len({s["hashes"][1] for s in sources.values()}),
        unique_target_hashes=len({s["target"] for s in sources.values()}),
        boundary_target_hash_equal=sum(s["hashes"][1] == s["target"] for s in sources.values()),
        target_identity_hash_equal_stable=sum(sources[k]["target"] == sources[k]["hashes"][0] for k in stable),
        stable_denominator=len(stable),
        opposed_boundary_hash_different=sum(sources[k]["hashes"][1] != sources[opposed_keys[k]]["hashes"][1] for k in keys),
        raw_activation_arrays_in_inspected_archive=False,
        evidence_limit="Hashes establish changed numerical bytes, not semantic update, semantic absence, distance, or information loss.",
    )
    with np.load(root / "transport.npz", allow_pickle=False) as transport:
        if set(transport.files) != {"weight", "bias"}:
            raise ValueError("Unexpected ridge artifact")
        state_evidence["transport"] = {name: dict(shape=list(transport[name].shape), dtype=str(transport[name].dtype), finite=bool(np.isfinite(transport[name]).all())) for name in transport.files}
    free_outputs = list(data["free"].values())
    free_evidence = dict(
        n=len(free_outputs), policies=len(data["free_keys"]),
        distinct_texts=len({r["text"] for r in free_outputs}),
        texts=sorted({r["text"] for r in free_outputs}),
        stops=dict(Counter(r["stop"] for r in free_outputs)),
        emitted_token_counts=dict(Counter(str(r["token_count"]) for r in free_outputs)),
        equal_E_G_texts=sum(data["free"][*k, "initial"]["text"] == data["free"][*k, "updated"]["text"] for k in data["free_keys"]),
        limit="The identical emitted string contains no policy-specific distinction. This locates a collapse by D at the latest, not its first occurrence in latent processing.",
    )
    reversal = sum(rows["boundary"][k]["correct"] and rows["opposed"][k]["correct"]
                   and rows["boundary"][k]["valid_scores"] and rows["opposed"][k]["valid_scores"]
                   and rows["boundary"][k]["margin"] * rows["opposed"][k]["margin"] < 0
                   for k in rows["boundary"])
    raw_binding = read(root / "binding.json")
    structured_binding = read(repo / STUDY / "structured/binding.json")
    return dict(
        schema="native_g_archival_stage_audit_v1", base_commit=BASE_COMMIT,
        status="DESCRIPTIVE_REANALYSIS_ONLY", gate_status="NOT_QUALIFIED",
        independent_qualification_accessed=False, model_execution=False, new_training=False,
        input_sha256={name: sha(repo / name) for name in INPUTS},
        n_sources=len(keys), n_archived_rows=sum(len(x) for x in rows.values()),
        archived_route_totals_reproduced=True, routes=routes, paired=paired,
        subsets=subset_scores,
        opposed_reversal=dict(correct=reversal, n=len(rows["boundary"]),
                              unit="directions including both answer orders; not independent pairs"),
        state_evidence=state_evidence, free_readout=free_evidence,
        semantic_overlap=dict(
            matching_status="SAME_AUTHORED_POLICY_AND_WORLD_NOT_SAME_PHYSICAL_EVALUATION",
            primary_full_raw_panel=structured_overlap(repo, data, keys),
            primary_free_subset=structured_overlap(repo, data, data["free_keys"]),
            same_backbone=raw_binding["backbone"] == structured_binding["backbone"],
            same_source_config=raw_binding["config"] == structured_binding["config"],
            same_reader=raw_binding["reader"] == structured_binding["reader"],
            raw_reader_sha256=raw_binding["reader"], structured_reader_sha256=structured_binding["reader"],
            caveats=[
                "Structured sources are encoded as isolated fields; raw sources are encoded jointly into eight cut-13 slots.",
                "Structured native scoring receives an externally selected action row and a stored-action question; raw native scoring receives the full state plus a world-dependent policy question.",
                "Reader adapters and injection sites differ. Cross-interface differences cannot be attributed to G alone.",
                "Primary literal questions share wording, but their reader adapters and supplied descriptions differ.",
                "Fresh continuation reuses style indices for different source wording and changes the literal question. Bare case-key joins are not identical-input comparisons.",
            ],
        ),
        unavailable_measurements=[
            "No archived native inferred-premise formal interpreter: Boolean gold is an oracle reference, not a measured native comparator.",
            "No raw pre-writer/post-writer/transported activation arrays; source and boundary hashes and the ridge parameters are archived.",
            "Semantic correctness of raw G before a competent reader is not identified by these decisions or hashes.",
            "No free D/ED for 112 of the 128 native policies; absent cells are missing, not zero scores.",
            "No same-reader, same-physical-prompt structured-versus-native comparison.",
        ],
    )


def matched_rows(repo: Path):
    data = load_native(repo)
    routes = ("identity", *OPERATORS, "teacher", "zero", "unrelated", "literal", *FREE_ROUTES)
    for key in data["keys"]:
        for world in range(4):
            for order in (0, 1):
                cell = (*key, world, order)
                yield dict(
                    panel="supported_g_primary_raw", case_key=list(key), world=world,
                    label_order=order, source_sha256=data["sources"][key]["source"],
                    gold="AB"[ledger(key, world)[0] ^ order], affected_world=ledger(key, world)[1],
                    routes={tag: data["rows"][tag].get(cell, {"present": False, "reason": "outside_prespecified_free_readout_subset"}) for tag in routes},
                )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--out", type=Path, required=True,
                   help="Fresh, non-existing output directory; archived outputs are never replaced.")
    args = p.parse_args(argv)
    if args.out.exists():
        p.error("--out must name a non-existing directory")
    result = analyze(args.repo)
    output = args.out
    output.mkdir(parents=True, exist_ok=False)
    (output / "native_analysis.json").write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    with (output / "native_matched.jsonl.gz").open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as compressed:
            for row in matched_rows(args.repo):
                compressed.write((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
    print(json.dumps({"n_sources": result["n_sources"], "archived_route_totals_reproduced": True,
                      "boundary_pair": result["paired"]["identity_to_boundary"],
                      "output": str(output / "native_analysis.json")}, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
