"""Post-result, model-free joins of the frozen supported-G development archives.

No model, trainer, qualification loader, or original reducer is imported.  Gold
actions are recomputed from the authored Boolean ledger.  A source's global
action IDs, literal propositions, and menu-constrained native answers are kept
separate.  The helpers are importable by the combined stage audit.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

BASE_COMMIT = "ba20b3125ac498ef1f2fa21730eef3557e3dd953"
STUDY = "provenance/supported_g_20261104"
ACTOR = "provenance/role_memory_20261104/run/produce/fits.npz"
MENUS = "data/within_topic_wording_20261103.json"
CONFIG = "data/supported_g_20261104.json"
READOUT_CONFIG = "data/supported_g_readout_20261104.json"
MASKS = np.array([0b0011, 0b1100, 0b0101, 0b1010])
WORLD = np.arange(4)
VALUES = ("off", "on")
KINDS = ("structured", "readout")


def read_json(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def canonical_digest(value):
    """Match records.digest without importing its model dependencies."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return text_sha(text)


def predicates(base, guard):
    trigger = ((MASKS[base, None] >> WORLD) & 1).astype(bool)
    mask = np.where(guard == 0, 0, MASKS[np.maximum(guard - 1, 0)])
    veto = ((mask[:, None] >> WORLD) & 1).astype(bool)
    return trigger, veto


def render(actions):
    return "\n".join(
        f"When the left signal is {VALUES[w // 2]} and the right signal is "
        f"{VALUES[w % 2]}, I choose {action}."
        for w, action in enumerate(actions)
    )


def source_text(key, menus, styles):
    topic, base, veto, pole, style, order = map(int, key)
    words = styles[style]
    actions = [words[r].format(a=menus[topic][pole ^ r]) for r in (0, 1)]
    trigger = words[2].format(s=("left", "right")[base // 2], v=VALUES[base % 2])
    guard = words[4] if veto < 0 else words[3].format(
        s=("left", "right")[veto // 2], v=VALUES[veto % 2]
    )
    return "\n".join([*actions[:: 1 if order == 0 else -1], trigger, guard])


def literal_question(kind, menus, key, world, order, spec):
    left, right = VALUES[world // 2], VALUES[world % 2]
    if kind == "structured":
        stem = (
            f"Today the left signal is {left} and the right signal is {right}. "
            "Which action follows my policy?"
        )
    else:
        stem = spec["fresh_question"].format(left=left, right=right)
    a, b = menus[int(key[0])][:: 1 if order == 0 else -1]
    return stem + f"\nA = {a}. B = {b}. Respond with only A or B."


def decode(top, ab, labels, topics, gold_action, gold_label):
    pred = np.where(top == labels[0], 0, np.where(top == labels[1], 1, -1))
    margin = ab[..., 0] - ab[..., 1]
    valid = (pred >= 0) & np.isfinite(margin) & (margin != 0)
    action = np.where(
        pred >= 0, 2 * topics[:, None, None] + (pred ^ np.arange(2)), -1
    )
    return dict(
        prediction_label=pred, prediction_action=action, top=top, ab=ab,
        margin=margin, valid=valid, correct=(pred == gold_label) & valid,
    )


def load_panel(repo: Path, kind: str) -> dict:
    """Return joined arrays; kind is allowlisted and cannot address qualification data."""
    if kind not in KINDS:
        raise ValueError("Only the two already-opened development panels are allowed")
    repo = Path(repo)
    folder = repo / STUDY / kind
    states = arrays(folder / "states.npz")
    native = arrays(folder / "native.npz")
    dispatch = arrays(folder / "dispatch.npz")
    binding = read_json(folder / "binding.json")
    native_meta = read_json(folder / "native.json")
    if binding["qualification_content_loaded"] is not False:
        raise ValueError("This reducer accepts only unopened-qualification development bindings")
    for field, path in (
        ("config", CONFIG if kind == "structured" else READOUT_CONFIG),
        ("writer", ACTOR), ("source_and_update", f"{STUDY}/structured/heads.npz"),
    ):
        if binding[field] != sha256(repo / path):
            raise ValueError(f"Archive binding differs: {field}")
    labels = np.asarray(native_meta["labels"])
    menus = [g["options"] for g in read_json(repo / MENUS)["groups"]]
    actions = [a for menu in menus for a in menu]
    keys = states["keys"]
    if keys.shape != (1280, 6) or len({tuple(k) for k in keys}) != len(keys):
        raise ValueError("Unexpected or duplicated development case keys")
    topics, base, veto, pole, style, order = keys.T
    trigger, guard = predicates(base, veto + 1)
    gold_action = 2 * topics[:, None] + (pole[:, None] ^ (trigger & ~guard))
    gold_label = (gold_action % 2)[:, :, None] ^ np.arange(2)
    routes = {}
    for name, ix in dispatch.items():
        if ix.shape != (len(keys), 4):
            raise ValueError("Unexpected dispatch shape")
        if not np.all(native["topics"][ix] == topics[:, None]):
            raise ValueError("A native dispatch crossed menus")
        target = 1 - gold_label if name == "opposed" else gold_label
        routes[name] = decode(
            native["top"][ix], native["ab"][ix], labels, topics, gold_action, target
        )
    literal_banks, literal_texts = {}, {}
    for suffix in ("", "_parent") if kind == "readout" else ("",):
        bank = arrays(folder / f"literal{suffix}.npz")
        with gzip.open(folder / f"rendered{suffix}.jsonl.gz", "rt") as stream:
            text_rows = [json.loads(line) for line in stream]
        if len(text_rows) != len(bank["top"]):
            raise ValueError("Literal text/score banks differ")
        literal_banks[suffix], literal_texts[suffix] = bank, text_rows
        for name in ("D_initial", "D_updated", "oracle") if kind == "readout" else (
            "D_initial", "D_updated"
        ):
            ix = bank[name]
            if not all(text_rows[j]["topic"] == t for j, t in zip(ix, topics, strict=True)):
                raise ValueError("A literal dispatch crossed menus")
            routes[name + suffix] = decode(
                bank["top"][ix], bank["ab"][ix], labels, topics, gold_action, gold_label
            )
    expected_actions = np.stack(
        [2 * topics + (pole ^ order), 2 * topics + (pole ^ order ^ 1)], axis=1
    )
    field_ok = dict(
        trigger=states["field_base"] == base,
        veto=states["guard"] == veto + 1,
        role=states["field_regular"] == order,
        first_action=states["field_actions"][:, 0] == expected_actions[:, 0],
        second_action=states["field_actions"][:, 1] == expected_actions[:, 1],
    )
    field_ok["all"] = np.stack(list(field_ok.values()), axis=1).all(axis=1)
    return dict(
        kind=kind, keys=keys, states=states, native=native, dispatch=dispatch,
        binding=binding, native_meta=native_meta, labels=labels, menus=menus,
        actions=actions, literal_banks=literal_banks, literal_texts=literal_texts,
        routes=routes, gold_action=gold_action, gold_label=gold_label,
        trigger=trigger, veto=guard, affected=trigger & guard,
        field_ok=field_ok, expected_actions=expected_actions,
    )


def count(values, include=None):
    values = np.asarray(values)
    if include is not None:
        values = values[np.broadcast_to(include, values.shape)]
    return dict(n=int(values.size), correct=int(values.sum()))


def paired(first, second, include=None):
    return {
        name: count(values, include)["correct"]
        for name, values in (
            ("both_correct", first & second), ("lost", first & ~second),
            ("gained", ~first & second), ("both_wrong", ~first & ~second),
        )
    } | {"n": count(first, include)["n"]}


def route_summary(route, include=None):
    ok = route["correct"]
    out = count(ok, include)
    out["ties"] = count(route["margin"] == 0, include)["correct"]
    out["nonfinite"] = count(~np.isfinite(route["margin"]), include)["correct"]
    out["outside_AB"] = count(route["prediction_label"] < 0, include)["correct"]
    if include is None:
        out["complete_policies"] = count(ok.all((1, 2)))
        out["correct_both_orders"] = count(ok.all(2))
        out["by_label_order"] = [count(ok[:, :, o]) for o in (0, 1)]
    return out


def field_confusion(truth, prediction):
    wrong = truth != prediction
    if not wrong.any():
        return []
    pairs, counts = np.unique(
        np.stack([truth[wrong], prediction[wrong]], axis=1), axis=0, return_counts=True
    )
    return [dict(truth=int(a), inferred=int(b), n=int(n)) for (a, b), n in zip(pairs, counts)]


def substitutions(repo, panel):
    """Formal field replacements reuse archived exact float rows and their physical queries."""
    s, k, n = panel["states"], panel["keys"], panel["native"]
    with np.load(repo / ACTOR, allow_pickle=False) as z:
        pointer, writer = z["reward_17_address"], z["reward_17_message"]
    with np.load(repo / STUDY / "structured/heads.npz", allow_pickle=False) as z:
        gates = z["gates"] > 0
    inferred_trigger, _ = predicates(s["field_base"], s["guard"])
    ix = np.arange(len(k))
    branches = np.stack(
        [s["table"][ix, np.argmax(inferred_trigger == b, axis=1)] for b in (0, 1)], axis=1
    )
    bank = {
        (int(t), r.tobytes()): i
        for i, (t, r, missing) in enumerate(zip(n["topics"], n["rows"], n["missing"], strict=True))
        if not missing
    }
    results, hits = {}, {}
    for name, base, guard in (
        ("observed", s["field_base"], s["guard"]),
        ("true_trigger", k[:, 1], s["guard"]),
        ("true_veto", s["field_base"], k[:, 2] + 1),
        ("both", k[:, 1], k[:, 2] + 1),
    ):
        trigger, _ = predicates(base, guard)
        table = branches[ix[:, None], trigger.astype(int)]
        bit = (((2.0 * trigger - 1) * writer[:4]).sum(1) + writer[4] > 0).astype(int)
        distance = np.abs(table - table[:, :1]).sum(-1)
        relative = distance / np.maximum(distance.max(-1, keepdims=True), 1e-12)
        selected = (pointer[bit, 0, None] * relative).argmax(1)
        updated = np.where(gates[guard, :, None], table[ix, selected, None, :], table)
        if name == "observed" and not (
            np.array_equal(bit, s["bit"]) and np.array_equal(updated, s["updated"])
        ):
            raise ValueError("Observed G not reconstructed from frozen actor")
        addresses = np.array([
            [bank[int(topic), row.tobytes()] for row in updated[i]]
            for i, topic in enumerate(k[:, 0])
        ])
        route = decode(
            n["top"][addresses], n["ab"][addresses], panel["labels"], k[:, 0],
            panel["gold_action"], panel["gold_label"],
        )
        hits[name] = route["correct"]
        results[name] = route_summary(route)
        results[name]["strict_argmax_rows"] = count(updated.argmax(-1) == panel["gold_action"])
        results[name]["paired_vs_observed"] = paired(hits["observed"], hits[name])
    results["method"] = "post_result_formal_field_substitution_exact_row_bank_reuse_no_new_forward"
    return results


def analyze_panel(repo, panel):
    s, k, routes = panel["states"], panel["keys"], panel["routes"]
    n, d = panel["native"], panel["dispatch"]
    index = np.arange(len(k))
    inferred_trigger, inferred_veto = predicates(s["field_base"], s["guard"])
    regular_rows = s["table"][index, (~inferred_trigger).argmax(1)]
    formal = np.where(inferred_veto[:, :, None], regular_rows[:, None, :], s["table"])
    assembled = np.take_along_axis(
        s["field_actions"], s["field_regular"][:, None] ^ inferred_trigger.astype(int), axis=1
    )
    hard = s["updated"].argmax(-1)
    hard_good = hard == panel["gold_action"]
    in_menu = hard // 2 == k[:, 0, None]
    good = routes["updated"]["correct"]
    literal = routes["D_updated"]
    whole_hard = hard_good.all(1)
    fields = panel["field_ok"]
    action_words = panel["actions"]
    renderer_exact = {}
    for suffix, bank in panel["literal_banks"].items():
        text_rows = panel["literal_texts"][suffix]
        for route, values in (("D_initial", s["table"].argmax(-1)), ("D_updated", hard)):
            actual = [text_rows[j]["text"] for j in bank[route]]
            expected = [render([action_words[int(a)] for a in row]) for row in values]
            renderer_exact[route + suffix] = sum(a == b for a, b in zip(actual, expected))
    ed = n["rows"][d["ED_updated"]]
    relations = dict(
        incoming_argmax_assembled_from_inferred_fields=bool(np.array_equal(assembled, s["table"].argmax(-1))),
        incoming_bit_equals_inferred_trigger_at_reference_world=bool(np.array_equal(s["bit"], inferred_trigger[:, 0])),
        formal_inferred_interpreter_rows_equal_G=bool(np.array_equal(formal, s["updated"])),
        formal_archived_dispatch_equal_G=bool(np.array_equal(d["analytic"], d["updated"])),
        formal_archived_top_equal_G=bool(np.array_equal(routes["analytic"]["top"], routes["updated"]["top"])),
        formal_archived_AB_equal_G=bool(np.array_equal(routes["analytic"]["ab"], routes["updated"]["ab"])),
        updated_guard_consumed=bool(np.all(s["next_guard"] == 0)),
        renderer_matches_global_argmax_tables=renderer_exact,
        ED_argmax_rows_preserved=count(ed.argmax(-1) == hard),
        ED_bit_identical_float_rows=count(np.all(ed == s["updated"], axis=-1)),
        ED_top_tokens_identical=count(routes["ED_updated"]["top"] == routes["updated"]["top"]),
        ED_AB_scores_exact=bool(np.array_equal(routes["ED_updated"]["ab"], routes["updated"]["ab"])),
        ED_AB_max_abs_difference=float(np.max(np.abs(routes["ED_updated"]["ab"] - routes["updated"]["ab"]))),
    )
    for required in (
        "incoming_argmax_assembled_from_inferred_fields", "incoming_bit_equals_inferred_trigger_at_reference_world",
        "formal_inferred_interpreter_rows_equal_G", "formal_archived_dispatch_equal_G",
        "formal_archived_top_equal_G", "formal_archived_AB_equal_G", "updated_guard_consumed",
    ):
        if not relations[required]:
            raise ValueError(f"Broken archived invariant: {required}")
    if any(v != len(k) for v in renderer_exact.values()):
        raise ValueError("Renderer text does not match archived state")
    error_groups = []
    bits = np.stack([~fields[name] for name in fields if name != "all"], axis=1)
    names = [name for name in fields if name != "all"]
    for pattern in np.unique(bits, axis=0):
        include = np.all(bits == pattern, axis=1)
        error_groups.append(dict(
            wrong_fields=[name for name, wrong in zip(names, pattern) if wrong],
            sources=int(include.sum()),
            strict_complete_tables=count(whole_hard, include),
            native_complete_policies=count(good.all((1, 2)), include),
            native=count(good, include[:, None, None]),
            literal=count(literal["correct"], include[:, None, None]),
        ))
    partitions = {}
    for name, rows in (
        ("correct_rendered_action", hard_good),
        ("wrong_rendered_action_in_menu", ~hard_good & in_menu),
        ("rendered_action_outside_menu", ~in_menu),
    ):
        include = rows[:, :, None]
        partitions[name] = dict(
            rows=int(rows.sum()), native=count(good, include),
            literal=count(literal["correct"], include),
            literal_faithful_to_rendered_action=(
                count((literal["prediction_action"] == hard[:, :, None]) & literal["valid"], include)
                if name != "rendered_action_outside_menu" else None
            ),
            ED=count(routes["ED_updated"]["correct"], include),
        )
    pairs = {}
    for first, second in (
        ("identity", "updated"), ("analytic", "updated"),
        ("updated", "D_updated"), ("updated", "ED_updated"),
        ("D_updated", "ED_updated"), ("D_updated_parent", "D_updated"),
        ("oracle_parent", "oracle"),
    ):
        if first in routes and second in routes:
            pairs[f"{first}__{second}"] = paired(routes[first]["correct"], routes[second]["correct"])
    change = panel["affected"].any(1)
    stability = {}
    for name, include in (("change_required", change), ("stable", ~change)):
        stability[name] = dict(
            sources=int(include.sum()),
            initial_complete=count(routes["identity"]["correct"].all((1, 2)), include),
            G_complete=count(good.all((1, 2)), include),
            strict_G_complete=count(whole_hard, include),
            G_newly_complete=count(good.all((1, 2)) & ~routes["identity"]["correct"].all((1, 2)), include),
            paired=paired(routes["identity"]["correct"], good, include[:, None, None]),
            unchanged_float_tables=count(np.all(s["table"] == s["updated"], axis=(1, 2)), include),
        )
    examples = []
    masks = (
        ("successful_update_literal_failure", whole_hard[:, None, None] & good
         & ~routes["identity"]["correct"] & panel["affected"][:, :, None]
         & ~literal["correct"]),
        ("correct_table_literal_failure", whole_hard[:, None, None] & good & ~literal["correct"]),
        ("native_correct_rendered_action_outside_menu", good & ~in_menu[:, :, None]),
        ("literal_correct_despite_wrong_rendered_action", ~hard_good[:, :, None] & literal["correct"]),
    )
    cfg = read_json(repo / CONFIG)
    spec = read_json(repo / READOUT_CONFIG)
    styles = cfg["fresh"] if panel["kind"] == "structured" else spec["fresh"]
    for name, eligible in masks:
        locations = np.argwhere(eligible)
        if not len(locations):
            continue
        i, w, o = map(int, locations[0])
        bank = panel["literal_banks"][""]
        text = panel["literal_texts"][""][bank["D_updated"][i]]["text"]
        examples.append(dict(
            category=name, key=k[i].tolist(), world=w, label_order=o,
            source=source_text(k[i], panel["menus"], styles),
            source_fields={name: bool(values[i]) for name, values in fields.items()},
            intended_action=action_words[int(panel["gold_action"][i, w])],
            hard_rendered_action=action_words[int(hard[i, w])], rendered_text=text,
            literal_question=literal_question(panel["kind"], panel["menus"], k[i], w, o, spec),
            soft_mass_at_gold=float(s["updated"][i, w, panel["gold_action"][i, w]]),
            soft_mass_at_argmax=float(s["updated"][i, w, hard[i, w]]),
            outcomes={name: dict(
                label="AB"[r["prediction_label"][i, w, o]] if r["prediction_label"][i, w, o] >= 0 else None,
                margin=float(r["margin"][i, w, o]), correct=bool(r["correct"][i, w, o]),
            ) for name, r in routes.items() if name in ("identity", "analytic", "updated", "D_updated", "ED_updated")},
        ))
    chosen = ("identity", "analytic", "updated", "D_initial", "D_updated", "ED_initial", "ED_updated", "D_updated_parent", "oracle", "oracle_parent", "source_prefix")
    return dict(
        panel=panel["kind"], sources=len(k), logical_decisions=int(good.size), known_menu_clusters=8,
        physical_native_queries=int(n["top"].size),
        physical_literal_queries={suffix or "current": int(bank["top"].size) for suffix, bank in panel["literal_banks"].items()},
        fields={name: count(values) for name, values in fields.items()},
        field_confusions=dict(
            trigger=field_confusion(k[:, 1], s["field_base"]),
            veto=field_confusion(k[:, 2] + 1, s["guard"]),
            actions=field_confusion(panel["expected_actions"].flatten(), s["field_actions"].flatten()),
        ),
        relations=relations,
        strict_argmax_rows=count(hard_good), strict_complete_tables=count(whole_hard),
        native_complete_despite_incorrect_global_table=count(good.all((1, 2)) & ~whole_hard),
        complete_table_despite_wrong_source_fields=count(whole_hard, ~fields["all"]),
        routes={name: route_summary(routes[name]) for name in chosen if name in routes},
        paired_transitions=pairs, stability=stability, field_error_partition=error_groups,
        rendered_action_partition=partitions,
        literal_on_wholly_correct_tables=count(literal["correct"], whole_hard[:, None, None]),
        literal_on_correct_rendered_rows=count(literal["correct"], hard_good[:, :, None]),
        native_correct_hard_wrong=count(good, ~hard_good[:, :, None]),
        native_wrong_hard_correct=count(~good, hard_good[:, :, None]),
        literal_on_native_correct_hard_wrong=count(literal["correct"], good & ~hard_good[:, :, None]),
        literal_valid_fidelity_within_menu=count(
            (literal["prediction_action"] == hard[:, :, None]) & literal["valid"], in_menu[:, :, None]
        ),
        field_substitutions=substitutions(repo, panel), examples=examples,
    )


def input_paths():
    fixed = [ACTOR, MENUS, CONFIG, READOUT_CONFIG, f"{STUDY}/structured/heads.npz"]
    for kind in KINDS:
        fixed += [f"{STUDY}/{kind}/{name}" for name in (
            "states.npz", "native.npz", "native.json", "dispatch.npz", "binding.json",
            "literal.npz", "rendered.jsonl.gz",
        )]
    fixed += [f"{STUDY}/readout/{name}" for name in ("literal_parent.npz", "rendered_parent.jsonl.gz")]
    fixed += [f"scripts/{name}" for name in (
        "structured_supported_boundary.py", "calibrate_boundary_readout.py", "state_only_native.py",
    )]
    fixed += ["src/selffold/supported_boundary.py", "src/selffold/role_memory.py"]
    return sorted(fixed)


def analyze(repo: Path) -> dict:
    repo = Path(repo)
    inputs = {name: sha256(repo / name) for name in input_paths()}
    panels = {kind: analyze_panel(repo, load_panel(repo, kind)) for kind in KINDS}
    return dict(
        schema="supported_g_stage_structured_audit_v1", base_commit=BASE_COMMIT,
        status="POST_RESULT_DEVELOPMENT_DIAGNOSTIC_NOT_QUALIFIED", model_forwards=0,
        training_updates=0, qualification_opened=False, input_sha256=inputs,
        matching=dict(
            key_fields=["panel", "topic", "trigger", "veto", "pole", "style", "source_order", "world", "label_order"],
            panel_namespace_required=True,
            common_truth="complete_original_policy_in_all_four_worlds",
            native_physical_interface="external_world_address_then_stored_action_question_with_frozen_row_reader",
            literal_physical_interface="whole_rendered_table_plus_world_question_with_literal_reader",
            exact_prompt_pair="fresh_parent_vs_calibrated_literal_reader",
            native_vs_literal_is_identical_prompt=False,
            uncertainty="descriptive_archival_counts; repeated world/order/style rows are not independent samples",
            hash_schemes=dict(
                source_sha256="SHA256(canonical JSON string), matching archived records.digest",
                literal_question_sha256="SHA256(canonical JSON string), matching archived records.digest",
                rendered_sha256="SHA256(canonical JSON string)",
                native_prompt_sha256="archived SHA256(canonical JSON token-ID list), not a text hash",
                utf8_sha256="SHA256(raw UTF-8 text bytes); explicitly suffixed companion fields",
            ),
        ),
        interpretation_limits=[
            "Literal answers and native row answers share logical questions, not prompts, adapters or world-addressing workload.",
            "Strict action-table correctness is global argmax equality, not a claim about all information in the soft row.",
            "For a rendered action outside the offered menu, A/B correctness cannot demonstrate fidelity to the text.",
            "D and ED are parallel reads of the emitted text; ED does not re-encode the literal reader's A/B answer.",
            "ED preserves choices and hard action IDs here, not floating states or A/B scores.",
            "Formal substitutions reuse only exact archived physical rows and do not establish unseen-form transfer.",
        ],
        panels=panels,
    )


def matched_rows(repo: Path):
    """Yield per-decision joins, preserving distinct physical question IDs."""
    repo = Path(repo)
    spec = read_json(repo / READOUT_CONFIG)
    cfg = read_json(repo / CONFIG)
    for kind in KINDS:
        p = load_panel(repo, kind)
        styles = cfg["fresh"] if kind == "structured" else spec["fresh"]
        bank = p["literal_banks"][""]
        for i, key in enumerate(p["keys"]):
            source = source_text(key, p["menus"], styles)
            text = p["literal_texts"][""][bank["D_updated"][i]]["text"]
            for w in range(4):
                for o in range(2):
                    question = literal_question(kind, p["menus"], key, w, o, spec)
                    yield dict(
                        panel=kind, key=key.tolist(), world=w, label_order=o,
                        source_sha256=canonical_digest(source), source_utf8_sha256=text_sha(source),
                        gold_action_id=int(p["gold_action"][i, w]),
                        gold_label="AB"[p["gold_label"][i, w, o]],
                        hard_rendered_action_id=int(p["states"]["updated"][i, w].argmax()),
                        rendered_sha256=canonical_digest(text), rendered_utf8_sha256=text_sha(text),
                        native_prompt_sha256=p["native_meta"]["prompts"][f"{key[0]}/{o}"],
                        literal_question_sha256=canonical_digest(question), literal_question_utf8_sha256=text_sha(question),
                        literal_input_digest=canonical_digest([text, question]),
                        same_physical_question_across_native_and_literal=False,
                        routes={name: dict(
                            top=int(route["top"][i, w, o]), label=int(route["prediction_label"][i, w, o]),
                            action_id=int(route["prediction_action"][i, w, o]),
                            margin=float(route["margin"][i, w, o]),
                            valid=bool(route["valid"][i, w, o]), correct=bool(route["correct"][i, w, o]),
                        ) for name, route in p["routes"].items() if name in (
                            "identity", "analytic", "updated", "D_initial", "D_updated", "ED_initial",
                            "ED_updated", "D_updated_parent", "oracle", "oracle_parent",
                        )},
                    )


def write_matched(repo: Path, output: Path):
    """Write new deterministic gzip output; refuse to replace an existing file."""
    with output.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as stream:
        for row in matched_rows(repo):
            stream.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True, help="New output directory; must not already exist")
    parser.add_argument("--matched", action="store_true", help="Also write the per-decision joined rows")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("--out must be a new directory; archived outputs are never replaced")
    result = analyze(args.repo)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "structured_analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.matched:
        write_matched(args.repo, args.out / "structured_matched.jsonl.gz")
    print(json.dumps({"output": str(args.out), "model_forwards": 0, "qualification_opened": False}))
    return result


if __name__ == "__main__":
    main()
