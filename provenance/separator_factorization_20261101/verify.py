"""Compile/check a prospective design using frozen native IDs; never load a model.

This is a stimulus/identifiability checker, not a GPU runner. Standard library only.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import combinations, product
import json
import math
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = ROOT / "provenance/downstream_span_20261101/trials"
SITES = ("first_punct", "second_punct", "receiver_early", "receiver_late")


def canonical(value):
    # Matches selffold.records.canonical without importing its torch dependency.
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def design():
    return json.loads((HERE / "design.json").read_text())


@lru_cache(maxsize=None)
def archived(model, panel):
    folder = ARCHIVE / f"clauses-{panel}-{model}"
    prompts = [json.loads(s) for s in (folder / "prompts.jsonl").read_text().splitlines()]
    return prompts, json.loads((folder / "span_settings.json").read_text())


def character_token(prompt, position):
    """Require one unfused native token for exactly this suffix character."""
    hits = [i for i, (lo, hi) in enumerate(prompt["suffix_offsets"]) if lo <= position < hi]
    require(len(hits) == 1, "Ambiguous character/token boundary")
    i = hits[0]
    require(prompt["suffix_offsets"][i] == [position, position + 1], "Fused punctuation token")
    return prompt["recipient_ids"][prompt["slot_positions"][-1] + 1 + i]


def archive_identity(model, panel, config):
    rows, settings = archived(model, panel)
    expected = config["models"][model]
    folder = ARCHIVE / f"clauses-{panel}-{model}"
    request = json.loads((folder / "request.json").read_text())
    identity = json.loads((folder / "checkpoint_identity.json").read_text())
    host = json.loads((folder / "host_identity.json").read_text())
    require(request["snapshot_revision"] == expected["revision"], "Snapshot revision differs from archive")
    require(request["model"] == expected["profile"], "Model profile differs from archive")
    require(identity["checkpoint_sha256"] == expected["checkpoint_sha256"] == host["checkpoint_sha256"],
            "Checkpoint fingerprint differs from archive")
    require(host["cut"] == expected["cut"] and host["slots"] == config["geometry"]["source_slots"],
            "Cut/slot geometry differs from archive")
    require(host["dtype"] == "torch." + config["execution"]["dtype"], "Injection dtype differs from archive")
    runtime = json.loads((ROOT / config["execution"]["runtime_reference"]).read_text())
    environment = json.loads((folder / "environment.before.json").read_text())
    for key in ("python", "torch_cuda", "packages"):
        require(runtime[key] == environment[key], "Native runtime references disagree")
    groups = {row["group"] for row in rows}
    require(len(groups) == config["topics_per_panel"], "Wrong topic denominator")
    require(len(rows) == len(groups) * 2, "Incomplete/duplicate native order grid")
    require({(p["group"], p["plus_label"]) for p in rows} == set(product(groups, ("A", "B"))),
            "Incomplete native mapping grid")
    for p in rows:
        text, ids, w = p["rendered_suffix"], p["recipient_ids"], p["windows"]
        question_end = text.index(p["question"]) + p["question"].rindex("?")
        require(character_token(p, question_end) == expected["question_mark_id"],
                "Question-mark ID differs from frozen native offsets")
        chosen = p["options_plus_minus"] if p["plus_label"] == "A" else p["options_plus_minus"][::-1]
        for letter, phrase in zip(("A", "B"), chosen, strict=True):
            char = text.index(f"{letter} = {phrase}") + len(f"{letter} = {phrase}")
            require(character_token(p, char) == expected["period_id"], "Period ID differs from native offsets")
            a, b = w[f"{letter}_punct"]
            require(b == a + 1 and ids[a] == expected["period_id"], "Invalid period window")
            require(w[f"{letter}_words"][1] == a, "Period fused with phrase")
        require(w["score"] == [len(ids) - 2, len(ids) - 1], "Archived score moved")
        require(ids[-1] == settings["dummy_id"], "Archived dummy differs")
        require(len(p["slot_positions"]) == config["geometry"]["source_slots"], "Wrong source width")
        require(all(len(x) == config["geometry"]["source_slots"] for x in p["padded_lexical_ids"].values()),
                "Wrong lexical donor width")
    return sorted(groups)


def compile_layout(model, panel, group, punctuation, display, plus_label, u=0, v=0, *, config=None):
    config = config or design()
    rows, settings = archived(model, panel)
    matches = [p for p in rows if p["group"] == group and p["plus_label"] == plus_label]
    require(len(matches) == 1, "Missing/duplicate archive row")
    p = matches[0]
    require(display in config["factors"]["display_order"], "Undeclared display")
    require(punctuation in config["factors"]["punctuation"], "Undeclared punctuation")
    require(u in config["factors"]["front_padding"] and v in config["factors"]["post_padding"],
            "Undeclared distance grid")
    g, model_spec = config["geometry"], config["models"][model]
    filler = settings["dummy_id"]
    old_ids, old_windows = p["recipient_ids"], p["windows"]
    ids, segments, windows = [], [], {}

    def add(name, values, *, source_span=None):
        a = len(ids)
        ids.extend(values)
        span = [a, len(ids)]
        segments.append({"name": name, "span": span, "archive_span": source_span})
        return span

    add("native_prefix", old_ids[:old_windows["options"][0]], source_span=[0, old_windows["options"][0]])
    shadow = add("shadow", [filler] * g["shadow_bank_width"])
    add("front_padding", [filler] * u)
    word_cells, phrase_spans, labels = [], [], []
    for ordinal, letter in enumerate(display, start=1):
        label_span = [old_windows[f"{letter}_clause"][0], old_windows[f"{letter}_words"][0]]
        label_ids = old_ids[slice(*label_span)]
        require(len(label_ids) == 2, "Native labeled scaffold no longer has fixed width")
        labels.append(letter)
        add(f"{ordinal}_label", label_ids, source_span=label_span)
        phrase_span = old_windows[f"{letter}_words"]
        words = old_ids[slice(*phrase_span)]
        require(0 < len(words) <= g["word_cell_width"], "Option does not fit fixed word cell")
        a = len(ids)
        add(f"{ordinal}_word_padding", [filler] * (g["word_cell_width"] - len(words)))
        phrase_spans.append(add(f"{ordinal}_phrase", words, source_span=phrase_span))
        word_cells.append([a, len(ids)])
        key = "first_punct" if ordinal == 1 else "second_punct"
        token = model_spec["period_id"] if ordinal == 1 else model_spec[punctuation + "_id"]
        windows[key] = add(key, [token], source_span=old_windows[f"{letter}_punct"])
    add("pre_receiver_padding", [filler] * g["pre_receiver_padding"])
    windows["receiver_early"] = add("receiver_early", [model_spec["period_id"]])
    add("between_receivers_padding", [filler] * g["between_receivers_padding"])
    windows["receiver_late"] = add("receiver_late", [model_spec["period_id"]])
    add("post_padding", [filler] * v)
    closing = add("native_closing", old_ids[slice(*old_windows["closing"])], source_span=old_windows["closing"])
    score_index = len(ids) - 1
    windows["score"] = [score_index, score_index + 1]
    dummy = add("dummy", [filler])
    compensation = g["compensation_tokens"] - u - v
    require(compensation >= 0, "Negative length reservoir")
    add("compensation", [filler] * compensation)
    require(len(ids) <= g["max_forward_length"], "Oversized design; never truncate")
    require(ids[dummy[0] - 1] == old_ids[old_windows["score"][0]], "Scoring token changed")
    require(p["slot_positions"] == list(range(p["slot_positions"][0], p["slot_positions"][-1] + 1)),
            "Noncontiguous source slots")
    cell = {"model": model, "panel": panel, "group": group, "punctuation": punctuation,
            "display_order": display, "plus_label": plus_label, "u": u, "v": v,
            "block": "surface" if (u, v) == (0, 0) else "distance"}
    return {"cell": cell, "recipient_ids": ids, "score_index": score_index,
            "question": p["question"], "options_plus_minus": p["options_plus_minus"], "filler_id": filler,
            "slot_positions": p["slot_positions"], "padded_lexical_ids": p["padded_lexical_ids"],
            "windows": windows, "word_cells": word_cells, "phrase_spans": phrase_spans,
            "shadow_span": shadow, "displayed_labels": labels, "segments": segments,
            "dummy_index": dummy[0], "native_closing_span": closing,
            "recipient_ids_sha256": digest(ids), "archive_row_sha256": digest(p),
            "archive_path": str((ARCHIVE / f"clauses-{panel}-{model}/prompts.jsonl").relative_to(ROOT)),
            "archive_suffix_offsets": p["suffix_offsets"], "rendering": g["rendering"]}


def compile_all(config):
    layouts = []
    for model, panel in product(config["models"], config["panels"]):
        for group in archive_identity(model, panel, config):
            for punctuation, display, mapping in product(config["factors"]["punctuation"],
                                                        config["factors"]["display_order"],
                                                        config["factors"]["plus_label"]):
                layouts.append(compile_layout(model, panel, group, punctuation, display, mapping, config=config))
            for display, mapping, u, v in product(config["factors"]["display_order"],
                                                  config["factors"]["plus_label"],
                                                  config["factors"]["front_padding"],
                                                  config["factors"]["post_padding"]):
                if (u, v) != (0, 0):
                    layouts.append(compile_layout(model, panel, group, "period", display, mapping, u, v, config=config))
        rows, _ = archived(model, panel)
        for p in rows:
            cell = {"model": model, "panel": panel, "group": p["group"], "block": "bridge",
                    "punctuation": "period", "display_order": "AB", "plus_label": p["plus_label"], "u": 0, "v": 0}
            layouts.append({"cell": cell, "recipient_ids": p["recipient_ids"],
                            "score_index": len(p["recipient_ids"]) - 2,
                            "windows": {"first_punct": p["windows"]["A_punct"], "second_punct": p["windows"]["B_punct"]},
                            "archive_row_sha256": digest(p)})
    return layouts


def compatible(target, source, site, *, punctuation_cross=False):
    require(len(target["recipient_ids"]) == len(source["recipient_ids"]), "Different forward lengths")
    require(target["score_index"] == source["score_index"], "Different score indices")
    require(target["windows"][site] == source["windows"][site], "Different absolute anchor positions")
    lo, hi = target["windows"][site]
    equal_ids = target["recipient_ids"][lo:hi] == source["recipient_ids"][lo:hi]
    if punctuation_cross:
        require(site == "second_punct" and not equal_ids, "Not the declared punctuation-ID cross")
        return "intentional_punctuation_id_cross"
    require(equal_ids, "Different anchor IDs outside the punctuation cross")
    return "matched_positions_and_token_ids"


def validate_grids(layouts, config):
    indexed = {}
    for layout in layouts:
        c = layout["cell"]
        key = (c["model"], c["panel"], c["group"], c["block"], c["punctuation"],
               c["display_order"], c["plus_label"], c["u"], c["v"])
        require(key not in indexed, "Duplicated cell")
        indexed[key] = layout
    expected = len(config["models"]) * len(config["panels"]) * config["topics_per_panel"] * 22
    require(len(indexed) == expected, "Incomplete planned layout grid")
    comparisons = 0
    for layout in layouts:
        c = layout["cell"]
        if c["block"] != "surface":
            continue
        m, p, t = c["model"], c["panel"], c["group"]
        punctuation, display, mapping = c["punctuation"], c["display_order"], c["plus_label"]
        flip_display, flip_mapping = ("BA" if display == "AB" else "AB"), ("B" if mapping == "A" else "A")
        for sd, sm in ((display, flip_mapping), (flip_display, mapping), (flip_display, flip_mapping)):
            source = indexed[m, p, t, "surface", punctuation, sd, sm, 0, 0]
            for site in SITES:
                compatible(layout, source, site)
                comparisons += 1
        other = "question_mark" if punctuation == "period" else "period"
        source = indexed[m, p, t, "surface", other, display, mapping, 0, 0]
        compatible(layout, source, "second_punct", punctuation_cross=True)
        comparisons += 1
        first_end = layout["windows"]["first_punct"][1]
        require(layout["recipient_ids"][:first_end] == source["recipient_ids"][:first_end],
                "Later punctuation changed the first anchor's prefix")
        if punctuation != "period":
            continue
        for u in config["factors"]["front_padding"]:
            pair = [indexed[m, p, t, "surface" if (u, v) == (0, 0) else "distance",
                            "period", display, mapping, u, v] for v in config["factors"]["post_padding"]]
            left, right = pair
            end = left["windows"]["receiver_late"][1]
            require(left["recipient_ids"][:end] == right["recipient_ids"][:end],
                    "Distance pair changed the donor anchor prefix")
            require(len(left["recipient_ids"]) == len(right["recipient_ids"]), "Distance length mismatch")
            for v, point in zip(config["factors"]["post_padding"], pair, strict=True):
                require(point["score_index"] - layout["score_index"] == u + v, "Incorrect score displacement")
                for site in config["budget"]["distance_sites"]:
                    a0, a = layout["windows"][site][0], point["windows"][site][0]
                    require(a - a0 == u, "Incorrect anchor displacement")
                    require((point["score_index"] - a) - (layout["score_index"] - a0) == v,
                            "Incorrect transport distance")
    return {"layouts": len(layouts), "context_position_id_comparisons": comparisons,
            "max_forward_length": max(len(x["recipient_ids"]) for x in layouts),
            "topics_per_cell": config["topics_per_panel"],
            "donor_order_rows_per_display_panel_cell": config["topics_per_panel"] * 4,
            "layout_digest_sha256": digest(layouts)}


def matrix_rank(matrix):
    a = [[Fraction(x) for x in row] for row in matrix]
    rank = 0
    for col in range(len(a[0])):
        pivot = next((r for r in range(rank, len(a)) if a[r][col]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        scale = a[rank][col]
        a[rank] = [x / scale for x in a[rank]]
        for r in range(len(a)):
            if r != rank:
                factor = a[r][col]
                a[r] = [x - factor * y for x, y in zip(a[r], a[rank], strict=True)]
        rank += 1
        if rank == len(a):
            break
    return rank


def design_ranks():
    terms = [()] + [t for width in range(1, 5) for t in combinations(range(4), width)]
    surface = [[math.prod(row[i] for i in term) for term in terms] for row in product((-1, 1), repeat=4)]
    coordinates = [[1, u, v, u + v] for u, v in product((0, 16), repeat=2)]
    require(matrix_rank(surface) == 16, "Surface factorial is aliased")
    require(matrix_rank(coordinates) == 3, "Coordinate identity was not detected")
    return {"source_punctuation_recipient_punctuation_display_mapping": {"columns": 16, "rank": 16},
            "intercept_anchor_distance_score": {"columns": 4, "rank": 3,
                                                 "alias": "score - anchor - distance = 0"},
            "intercept_anchor_distance": {"columns": 3, "rank": matrix_rank([r[:3] for r in coordinates])}}


def mask_spec(layout, context):
    require(context in ("open", "release_early", "release_late", "never",
                        "sham_early", "sham_late", "sham_never"), "Unknown mask context")
    if context == "open":
        return set(), 0
    spans = [layout["shadow_span"]] if context.startswith("sham") else layout["word_cells"]
    protected = {i for lo, hi in spans for i in range(lo, hi)}
    suffix = context.rsplit("_", 1)[-1]
    release = len(layout["recipient_ids"]) if suffix == "never" else layout["windows"]["receiver_" + suffix][0]
    return protected, release


def edge_allowed(query, key, protected, release):
    return key <= query and not (key in protected and query not in protected and query < release)


def propagate_sources(length, word_cells, depth, allowed):
    """Reachability in a dense causal lower-stack supergraph, including residuals."""
    flags = [0] * length
    for bit, (lo, hi) in enumerate(word_cells):
        for i in range(lo, hi):
            flags[i] |= 1 << bit
    for _ in range(depth):
        result = flags.copy()
        for q in range(length):
            for k in range(q + 1):
                if allowed(q, k):
                    result[q] |= flags[k]
                    if result[q] == 3:
                        break
        flags = result
    return flags


def graph_checks(config):
    result = {}
    for model in config["models"]:
        groups = archive_identity(model, "original", config)
        layout = compile_layout(model, "original", groups[0], "period", "AB", "A", config=config)
        length, depth = len(layout["recipient_ids"]), config["models"][model]["cut"]
        expected = {"open": [1, 3, 3, 3], "release_early": [0, 0, 3, 3],
                    "release_late": [0, 0, 0, 3], "never": [0, 0, 0, 0],
                    "sham_early": [1, 3, 3, 3], "sham_late": [1, 3, 3, 3], "sham_never": [1, 3, 3, 3]}
        observed = {}
        for context, wanted in expected.items():
            protected, release = mask_spec(layout, context)
            require(not protected or len(protected) == 16, "Mask width differs from sham")
            flags = propagate_sources(length, layout["word_cells"], depth,
                                      lambda q, k: edge_allowed(q, k, protected, release))
            observed[context] = [flags[layout["windows"][site][0]] for site in SITES]
            require(observed[context] == wanted, f"Mask graph leak/access failure: {context}")
        hidden, _ = mask_spec(layout, "never")
        late = layout["windows"]["receiver_late"][0]
        naive = propagate_sources(length, layout["word_cells"], depth,
                                  lambda q, k: k <= q and not (q == late and k in hidden))
        require(naive[late] == 3, "Direct-only blocker did not expose the relay counterexample")
        result[model] = {"sites": list(SITES), "source_flags_by_context": observed,
                         "direct_only_blocker_leaks_both_cells": True,
                         "scope": "dense_causal_supergraph; numerical_checkpoint_mask_adapter_unvalidated"}
    return result


def compute_budget(config):
    f, b = config["factors"], config["budget"]
    surface = len(f["punctuation"]) * len(f["display_order"]) * len(f["plus_label"])
    base = len(f["display_order"]) * len(f["plus_label"])
    distance = base * (len(f["front_padding"]) * len(f["post_padding"]) - 1)
    require(b["surface_forwards_per_layout"] == 3 + 1 + 2 + 2 + len(SITES) * 2 * 3 + len(SITES) + 1,
            "Surface cost missing an operation")
    require(b["access_forwards_per_context_layout"] == 3 + 1 + 2 + 2 + 3 * 2 * 2,
            "Access cost missing a reference or route")
    require(b["distance_forwards_per_new_layout"] == 3 + 1 + 2 + 2 + 2 * 2 * 3 + 2 + 1,
            "Distance cost missing an operation")
    require(b["bridge_forwards_per_topic_model"] == 2 * (3 + 1 + 2 + 2 + 2 * 2 + 2 + 1),
            "Bridge cost missing an operation")
    counts = {"bridge": b["bridge_forwards_per_topic_model"],
              "surface": surface * b["surface_forwards_per_layout"],
              "mapping_display_swaps": surface * len(b["swap_contexts"]) * len(SITES) * 2 * 2,
              "punctuation_cross": surface * 1 * 2 * 2,
              "access": base * len(f["access_contexts"]) * b["access_forwards_per_context_layout"],
              "distance": distance * b["distance_forwards_per_new_layout"]}
    per_model = sum(counts.values()) * len(config["panels"]) * config["topics_per_panel"]
    maximum = (per_model + b["preflight_forward_allowance_per_model"]) * len(config["models"])
    require(per_model + b["preflight_forward_allowance_per_model"] == b["max_forwards_per_model"],
            "Per-model forward cap disagrees with design")
    require(maximum == b["max_forwards_total"], "Total forward cap disagrees with design")
    require(maximum * config["geometry"]["max_forward_length"] == b["max_processed_tokens_total"],
            "Processed-token cap is incorrect")
    require(b["max_processed_tokens_per_model"] * len(config["models"]) == b["max_processed_tokens_total"],
            "Per-model processed-token cap is incorrect")
    return {"forwards_per_topic_model": counts, "sum_per_topic_model": sum(counts.values()),
            "outcome_bridge_total": per_model * len(config["models"]), "maximum_with_canaries": maximum,
            "maximum_processed_tokens": b["max_processed_tokens_total"],
            "max_gpu_wall_seconds_total": config["execution"]["max_wall_seconds_per_model"] * len(config["models"])}


def finite_panel_decision(contrasts, references, *, delta=0.05, required_topics=6):
    """The finite-panel robustness rule, never a significance test."""
    require(len(contrasts) == len(references), "Unpaired contrast/reference")
    require(all(math.isfinite(x) for x in [*contrasts, *references]), "Nonfinite endpoint")
    if len(contrasts) != required_topics:
        return {"decision": "unresolved", "reason": "incomplete_topics"}
    numerator, denominator = sum(contrasts), sum(references)
    denoms = [denominator, *(denominator - x for x in references)]
    if any(x <= 0 for x in denoms):
        return {"decision": "unresolved", "reason": "nonpositive_reference"}
    values = [numerator / denominator, *((numerator - x) / d for x, d in zip(contrasts, denoms[1:], strict=True))]
    if all(x > delta for x in values):
        decision = "positive"
    elif all(x < -delta for x in values):
        decision = "negative"
    elif all(-delta <= x <= delta for x in values):
        decision = "equivalent"
    else:
        decision = "unresolved"
    return {"decision": decision, "common_reference_fraction": values[0], "leave_one_topic_out": values[1:], "delta": delta}


def frozen_files(config):
    raw = subprocess.check_output(["git", "-C", str(ROOT), "ls-tree", "-r", "-z", config["frozen_base_commit"]])
    changed, count = [], 0
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        _, kind, expected = metadata.decode().split()
        require(kind == "blob", "Unexpected non-blob baseline entry")
        path = ROOT / name.decode()
        if not path.is_file():
            changed.append(name.decode())
            continue
        content = path.read_bytes()
        if hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest() != expected:
            changed.append(name.decode())
        count += 1
    require(not changed, "Frozen baseline files changed: " + ", ".join(changed))
    return {"base_commit": config["frozen_base_commit"], "checked_baseline_files": count, "changed_baseline_files": changed}


def verify():
    config = design()
    protected = frozen_files(config)
    layouts = compile_all(config)
    receipt = {"status": "PLAN_VALIDATED_NO_GPU_RUN", "gpu_ready": False, "new_model_forwards": 0,
               "runtime_tokenizer_or_checkpoint_loaded": False, "frozen_baseline": protected,
               "geometry": validate_grids(layouts, config), "identifiability": design_ranks(),
               "attention_graph": graph_checks(config), "compute_budget": compute_budget(config),
               "remaining_execution_gates": ["explicit_score_index_adapter_and_tests",
                                               "per_checkpoint_numeric_mask_adapter_and_tests",
                                               "clean_implementation_and_plan_execution_seal",
                                               "offline_native_canaries_and_exact_archived_bridge"],
               "plan_files_sha256": {name: file_sha(HERE / name)
                                      for name in ("PLAN.md", "design.json", "verify.py", "cell_reducer.py")},
               "tests_sha256": file_sha(ROOT / "tests/test_separator_plan.py")}
    return receipt, layouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Require committed verification.json to match")
    parser.add_argument("--out", type=Path, help="Write layouts exclusively; never model scores")
    args = parser.parse_args()
    receipt, layouts = verify()
    if args.check:
        require(receipt == json.loads((HERE / "verification.json").read_text()), "Preflight receipt is stale")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as f:
            for layout in layouts:
                f.write(canonical(layout) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
