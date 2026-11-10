"""Saved-input audits and reductions for the carried-reference experiment."""
import hashlib
import json
from collections import defaultdict

import numpy as np
import torch

from .lm_writer import decode_state_slot
from .lm_writer_reports import source_pairs
from .report_replay import read, reduce_compact
from .role_reference import source_reference
from .role_trials import donor_indices
from .roundtrip import text_features


def restore_slots(root, archive, styles):
    """Decode absolute slots without access to current policy target bits."""
    keys = archive["fit_keys"] + archive["keys"]
    pairs, index = source_pairs(keys)
    reference = source_reference(archive["members"])
    groups = read(root / "data/within_topic_wording_20261103.json")["groups"]
    slots, records = {}, {}
    for style in styles:
        bank = torch.full((len(keys), 16, 16, 16), -2, dtype=torch.long)
        rows = [json.loads(line) for line in
                (archive["archive"] / f"reports_{style}.jsonl").read_text().splitlines()]
        for row in rows:
            gi, _ = pairs[row["source"]]
            slot, missing = decode_state_slot(row["text"], groups[gi]["options"])
            cases = np.flatnonzero(index == row["source"])
            value = -1 if missing else slot
            bank[cases, row["prior"], row["mask"], row["out"]] = value
            bits = torch.zeros_like(reference[cases]) if missing else reference[cases] ^ slot
            old = archive["bits"][style][cases, row["prior"], row["mask"], row["out"]]
            if not torch.equal(bits, old):
                raise ValueError("Derived reference/slot does not reproduce the archived decoder")
        slots[style], records[style] = bank, rows
    return slots, records, reference


def report_audit(archive, records, reference, primary_topics):
    keys = archive["fit_keys"] + archive["keys"]
    pairs, index = source_pairs(keys)
    pair_ref = [int(reference[np.flatnonzero(index == i)[0]]) for i in range(len(pairs))]
    output = {}
    for style, rows in records.items():
        errors = {name: dict(records=0, missing=0, actual_correct=0, instruction_correct=0)
                  for name in ("out_equals_instruction", "out_differs_from_instruction")}
        cells = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for row in rows:
            source = row["source"]
            ref = pair_ref[source]
            panel = "primary_fit_menus" if pairs[source][0] in primary_topics else "unfitted_menus"
            item = errors["out_equals_instruction" if row["out"] == row["mask"]
                          else "out_differs_from_instruction"]
            item["records"] += 1
            item["missing"] += int(row["missing"])
            # These are checks of the saved lexical decoder, not semantic annotations.
            item["actual_correct"] += int(not row["missing"] and row["bit"] == (row["out"] & 1))
            item["instruction_correct"] += int(not row["missing"] and row["bit"] == (row["mask"] & 1))
            if row["out"] != row["mask"] or row["prior"] in (0, 15):
                continue
            role = row["out"] & 1
            relative = tuple(((row["out"] >> i) & 1) ^ role for i in range(4))
            features = hashlib.sha256(text_features(row["text"]).numpy().tobytes()).hexdigest()
            for view, value in (("exact_text", row["text"]), ("hashbag", features)):
                for condition in ("absent", "present"):
                    observation = (relative, value, ref if condition == "present" else None)
                    for scope in ("all_menus", panel):
                        cells[(scope, view, condition)][observation][role] += 1
        aliases = {}
        for (scope, view, condition), table in cells.items():
            values = list(table.values())
            aliases.setdefault(scope, {}).setdefault(view, {})[condition] = dict(
                records=sum(sum(v) for v in values), distinct_inputs=len(values),
                best_role_correct=sum(max(v) for v in values),
                conflicting_records=sum(sum(v) for v in values if min(v) > 0))
        output[style] = dict(decoder_faithfulness=errors, reference_conditioned_aliases=aliases)
    return dict(unit="unique_generated_report_record", equal_record_weights=True,
                role_bound_not_trajectory_bound=True, independent_semantic_annotation=False,
                styles=output)


def logical_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def save_compact(path, data):
    """Lossless column-major storage avoids large repeated pattern payloads."""
    with path.open("xb") as stream:
        np.savez_compressed(stream, **{k: np.asfortranarray(v) if np.ndim(v) > 1 else v
                                      for k, v in data.items()})
    return {k: dict(shape=list(np.shape(v)), dtype=str(np.asarray(v).dtype),
                    c_order_sha256=logical_hash(v)) for k, v in data.items()}


def summarize_panels(data, keys, cfg, top, labels):
    masks = cfg["fit_masks"] + cfg["test_masks"]
    selection = {
        "all_menus": list(range(len(keys))),
        "primary_fit_menus": [i for i, k in enumerate(keys) if k[0] in cfg["fit_topics"]],
        "unfitted_menus": [i for i, k in enumerate(keys) if k[0] not in cfg["fit_topics"]]}
    result = {}
    for panel, ids in selection.items():
        sliced = {k: data[k][ids] if k in ("depths", "second", "fourth", "twins", "prior_mismatch")
                  else data[k] for k in data}
        ks = [keys[i] for i in ids]
        result[panel] = reduce_compact(sliced, ks, masks, labels, top[ids], donor_indices(ks),
                                       cfg["test_masks"], cfg["fit_topics"])
    return result


def paired_reference(intact, opposed, keys, cfg, top, labels):
    """One weight per distinct source/first-mask/second-mask pair."""
    paths = intact["paths"]
    if not np.array_equal(paths, opposed["paths"]):
        raise ValueError("Reference conditions use different paths")
    indices = np.unique(paths[:, :2], axis=0, return_index=True)[1]
    masks = np.asarray(cfg["fit_masks"] + cfg["test_masks"])
    second = masks[paths[indices, 1]]
    target = labels[np.asarray([k[3] for k in keys])[:, None, None, None]
                    ^ ((second[:, None] >> np.arange(4)) & 1)[None, ..., None]
                    ^ np.arange(2)]
    def decisions(data):
        members = (data["depths"][:, indices, 1, None] >> np.arange(4)) & 1
        return top[np.arange(len(keys))[:, None, None], members]
    original, flipped = decisions(intact), decisions(opposed)
    good = (original == target).all((-1, -2))
    reversed_action = (flipped == target[..., ::-1]).all((-1, -2))
    result = {}
    for name, ids in (("all_menus", list(range(len(keys)))),
                      ("primary_fit_menus", [i for i, k in enumerate(keys) if k[0] in cfg["fit_topics"]]),
                      ("unfitted_menus", [i for i, k in enumerate(keys) if k[0] not in cfg["fit_topics"]])):
        result[name] = dict(units=int(good[ids].size),
            intact_complete=int(good[ids].sum()),
            flipped_when_intact_complete=[int((good[ids] & reversed_action[ids]).sum()), int(good[ids].sum())])
    return result


def gate_results(design, cfg, summaries, pairs):
    def fraction(pair):
        return pair[0] / pair[1] if pair[1] else 0.
    result = {}
    for style in design["styles"]:
        seeds = {}
        for seed in cfg["seeds"]:
            label = f"{style}_{seed}"
            original = summaries[f"unreferenced_{label}_intact"]["primary_fit_menus"]
            carried = summaries[f"referenced_{label}_intact"]["primary_fit_menus"]
            erased = summaries[f"referenced_{label}_fixed_zero"]["primary_fit_menus"]
            score = fraction(carried["all_four_correct"])
            gain = score-fraction(original["all_four_correct"])
            drop = score-fraction(erased["all_four_correct"])
            flip = pairs[f"referenced_{label}"]["primary_fit_menus"]["flipped_when_intact_complete"]
            conditions = dict(
                primary_all_four=score >= design["primary_min_all_four"],
                novel_masks=fraction(carried["all_four_novel_mask_strings"]) >= design["primary_min_novel_masks"],
                gain=gain >= design["minimum_gain_over_unreferenced"],
                reference_erasure_drop=drop >= design["minimum_drop_when_reference_erased"],
                initial_complete=carried["first_complete_source_mask"] == [448, 448],
                intervention_consistency=fraction(flip) >= design["minimum_conditional_flip"],
                equivalent_donors=all(v["equivalent_correct"] == v["matched"]
                                      for v in carried["opposed"].values()))
            seeds[str(seed)] = dict(passed=all(conditions.values()), checks=conditions,
                primary_all_four=carried["all_four_correct"],
                novel_masks=carried["all_four_novel_mask_strings"], gain=gain,
                erasure_drop=drop, conditional_flip=flip)
        result[style] = dict(passed=all(s["passed"] for s in seeds.values()), seeds=seeds)
    return dict(scope="scaffolded_readout_known_fit_menus", learned_reference_retention=False,
                qualification_loaded=False, styles=result)
