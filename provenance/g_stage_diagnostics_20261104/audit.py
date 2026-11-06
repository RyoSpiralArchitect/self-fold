"""Reproduce the post-result G stage audit without models or qualification inputs.

Run with --out pointing to a new directory. All reads address explicitly opened
development artifacts. No existing result or paper is rewritten.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import audit_native as raw
import audit_structured as structured


BASE_COMMIT = raw.BASE_COMMIT
HERE = Path(__file__).resolve().parent
STUDY = Path("provenance/supported_g_20261104")
STRUCTURED_ROUTES = ("identity", "analytic", "updated", "D_updated", "ED_updated")
NATIVE_ROUTES = ("identity", "plain", "decompose", "boundary", "reintegrate",
                 "teacher", "D_updated", "ED_updated")
FROZEN_MANIFEST_SHA256 = "2ed2398e6333ba024564c28b1870445556dbe53014982c5727d269b3e78a2d2e"


def exact_evaluation_match(left, right):
    """A semantic case-key match alone cannot authorize a physical comparison."""
    fields = ("source_digest", "question_digest", "reader", "geometry", "scoring")
    return all(left.get(k) is not None and left.get(k) == right.get(k) for k in fields)


def comparison_contract(repo):
    study = repo / STUDY
    sb = raw.read(study / "structured/binding.json")
    rb = raw.read(study / "raw-bound/binding.json")
    return dict(
        schema="g_stage_comparison_contract_v1", base_commit=BASE_COMMIT,
        status="POST_RESULT_DEVELOPMENT_REANALYSIS", qualification_status="NOT_QUALIFIED",
        semantic_key=["source_wording_panel", "topic", "trigger", "veto", "pole",
                      "style", "source_order", "world", "answer_order"],
        hash_schemes={
            "digest_json_sha256": "SHA256(canonical UTF-8 JSON value), matching selffold.records.digest",
            "text_utf8_sha256": "SHA256(UTF-8 text bytes); not interchangeable with digest_json_sha256",
            "native_prompt_token_digest": "archived digest of token IDs, not a text hash",
        },
        scoring="full-vocabulary winner must equal gold A/B; finite nonzero A-B margin required",
        targets={
            "task_correctness": "the original policy's independent four-world Boolean ledger",
            "rendered_text_fidelity": "the action actually named in the emitted world row",
            "outside_menu": "TARGET_OUTSIDE_OPTIONS: binary task accuracy cannot measure literal fidelity",
        },
        interfaces={
            "structured_native": dict(reader=sb["reader"], host=sb["host"],
                encoder="four separately encoded source fields",
                input="world selected externally; continuous 16-action row through embedding bank",
                question="Which option names my stored action?"),
            "structured_literal": dict(reader=sb["literal_reader"], host=sb["host"],
                input="entire fixed-vocabulary four-row rendered text",
                question="world-dependent policy question"),
            "raw_native": dict(reader=rb["reader"], host=rb["task_host"],
                encoder="joint description into eight question-free cut-13 workspace rows",
                input="entire eight-slot state at cut 13",
                question="world-dependent policy question"),
            "raw_literal": dict(reader=rb["reader"], host=rb["task_host"],
                input="free generated D(E) or D(G) text",
                question="world-dependent policy question"),
        },
        exact_within_interface_comparisons=[
            "structured identity / formal interpreter / learned G with fixed native row reader",
            "native identity / four native G operations / independently encoded target with fixed native recipient",
            "fresh parent vs calibrated literal reader: identical emitted text and question; reader is the intervention",
        ],
        comparisons_requiring_care=[
            "Structured and raw share 128 authored primary sources and 1024 semantic query cells, not one physical evaluation.",
            "Structured native vs literal differs in prompt, adapter and external world addressing.",
            "The fresh panel reuses integer style IDs but changes both source wording and literal question.",
            "D(G) is the emitted description. Literal answers and E(D(G)) are parallel readout branches, not a serial A/B chain.",
        ],
        unavailable=[
            "formal interpreter from native inferred premises (no such premise decoder was archived)",
            "native free D and ED for 112 policies outside the prespecified 16-policy subset",
            "native E/post-writer/pre-transport/post-transport/injected activation arrays",
            "identical physical question/reader six-route comparison",
            "structured same-reader soft-row vs one-hot global-argmax intervention",
        ],
        missing_policy="Use explicit unavailable status; never assign absent measurements zero correctness.",
        uncertainty="Descriptive finite-archive counts; styles, worlds and orders are repeated observations within eight known menus (four in raw).",
    )


def overlap_rows(repo):
    """Join the primary semantic overlap and keep physical mismatch explicit."""
    native = raw.load_native(repo)
    panel = structured.load_panel(repo, "structured")
    cfg = raw.read(repo / "data/supported_g_20261104.json")
    spec = raw.read(repo / "data/supported_g_readout_20261104.json")
    index = {tuple(map(int, k)): i for i, k in enumerate(panel["keys"])}
    for key in native["keys"]:
        i = index[key]
        source = structured.source_text(key, panel["menus"], cfg["fresh"])
        if raw.digest(source) != native["sources"][key]["source"]:
            raise ValueError("Authored sources differ despite equal integer keys")
        for world in range(4):
            for order in (0, 1):
                cell = (*key, world, order)
                q = structured.literal_question("structured", panel["menus"], key, world, order, spec)
                nr = native["rows"]["identity"][cell]
                if raw.digest(q) != nr["question_sha256"]:
                    raise ValueError("Primary literal and raw question texts differ")
                if "AB"[int(panel["gold_label"][i, world, order])] != nr["gold"]:
                    raise ValueError("Joined source/world/answer order has different gold")
                row = dict(
                    panel="primary_semantic_overlap", topic=key[0], trigger=key[1],
                    veto=key[2], pole=key[3], style=key[4], source_order=key[5],
                    world=world, answer_order=order, gold=nr["gold"],
                    source_digest_json_sha256=raw.digest(source),
                    literal_and_raw_question_digest_json_sha256=raw.digest(q),
                    structured_native_prompt_token_digest=panel["native_meta"]["prompts"][f"{key[0]}/{order}"],
                    same_authored_source=True, same_semantic_query=True,
                    same_physical_evaluation=False,
                    matching_status="DESCRIPTIVE_SEMANTIC_OVERLAP_ONLY",
                    affected_world=bool(panel["affected"][i, world]),
                    structured_global_argmax_action=int(panel["states"]["updated"][i, world].argmax()),
                    true_global_action=int(panel["gold_action"][i, world]),
                    native_formal_interpreter_status="UNAVAILABLE_NO_INFERRED_PREMISE_DECODER",
                )
                for tag in STRUCTURED_ROUTES:
                    r = panel["routes"][tag]
                    pred = int(r["prediction_label"][i, world, order])
                    row.update({
                        f"structured_{tag}_present": True,
                        f"structured_{tag}_prediction": "AB"[pred] if pred >= 0 else "OTHER",
                        f"structured_{tag}_margin": float(r["margin"][i, world, order]),
                        f"structured_{tag}_correct": bool(r["correct"][i, world, order]),
                    })
                for tag in NATIVE_ROUTES:
                    r = native["rows"][tag].get(cell)
                    row.update({
                        f"native_{tag}_present": r is not None,
                        f"native_{tag}_prediction": r["predicted_label"] if r else "",
                        f"native_{tag}_margin": r["margin"] if r else "",
                        f"native_{tag}_correct": r["correct"] if r else "",
                        f"native_{tag}_status": "MEASURED" if r else "OUTSIDE_FREE_READOUT_SUBSET",
                    })
                yield row


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_archive(repo):
    study = repo / STUDY
    if raw.sha(study / "SHA256SUMS") != FROZEN_MANIFEST_SHA256:
        raise ValueError("The frozen PR #27 checksum manifest changed")
    # Check only explicit development inputs. Listing these Git objects does not
    # open any other data files, including the independent qualification panel.
    paths = sorted(set(raw.INPUTS) | set(structured.input_paths()) | {"paper/main.md"})
    objects = subprocess.check_output(
        ["git", "ls-tree", "-rz", "--full-tree", BASE_COMMIT, "--", *paths], cwd=repo
    ).split(b"\0")
    expected_objects = {}
    for entry in objects:
        if entry:
            meta, name = entry.split(b"\t", 1)
            expected_objects[name.decode()] = meta.split()[2].decode()
    if set(expected_objects) != set(paths):
        raise ValueError("The pinned commit does not contain every declared audit input")
    for name, expected in expected_objects.items():
        content = (repo / name).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if actual != expected:
            raise ValueError(f"Input differs from pinned PR #27 commit: {name}")
    hashes = {}
    for line in (study / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = (study / relative).resolve()
        if not path.is_relative_to(study.resolve()):
            raise ValueError("Frozen manifest escaped its development directory")
        actual = raw.sha(path)
        if actual != expected:
            raise ValueError(f"Frozen development artifact changed: {relative}")
        hashes[relative] = actual
    summarize = load_module("summarize", study / "summarize.py")
    old = sys.modules.get("summarize")
    sys.modules["summarize"] = summarize
    try:
        fields = load_module("original_g_fields", study / "diagnose_fields.py")
        reductions = {}
        for kind, name in (("structured", "structured"), ("readout", "readout"), ("raw-bound", "raw")):
            ok = summarize.summarize(study, kind) == raw.read(study / f"{name}-summary.json")
            if not ok:
                raise ValueError(f"Original reduction changed: {kind}")
            reductions[name] = ok
        substitutions = {}
        for kind in ("structured", "readout"):
            ok = fields.diagnose(study, kind) == raw.read(study / f"{kind}-fields.json")
            if not ok:
                raise ValueError(f"Original field diagnostic changed: {kind}")
            substitutions[kind] = ok
    finally:
        if old is None:
            sys.modules.pop("summarize", None)
        else:
            sys.modules["summarize"] = old
    gate = raw.read(study / "gate.json")
    if gate["status"] != "NOT_QUALIFIED" or gate["qualification_opened"]:
        raise ValueError("Unexpected qualification status")
    return dict(
        schema="g_stage_audit_verification_v1", base_commit=BASE_COMMIT,
        pinned_input_git_objects_verified=len(expected_objects),
        frozen_archive_files_verified=len(hashes), frozen_manifest_sha256=raw.sha(study / "SHA256SUMS"),
        original_summary_reductions_exact=reductions, original_field_reductions_exact=substitutions,
        qualification_status=gate["status"], qualification_opened=False,
        qualification_commitment_sha256=gate["qualification_commitment_sha256"],
        human_transition_review=gate["human_transition_review"],
        new_model_forwards=0, new_training_updates=0,
        paper_v1_sha256=raw.sha(repo / "paper/main.md"),
        analysis_code_sha256={p.name: raw.sha(p) for p in (HERE / "audit.py", HERE / "audit_native.py", HERE / "audit_structured.py")},
    )


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def write_jsonl_gz(path, rows):
    with path.open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as zipped:
            for row in rows:
                zipped.write((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())


def run(repo, out):
    repo, out = Path(repo).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Choose a new output directory; existing evidence is never overwritten")
    for protected in (STUDY, Path("data"), Path("paper"), Path("src"), Path("scripts")):
        if out.is_relative_to((repo / protected).resolve()):
            raise ValueError("Output cannot be inside an existing source/data/paper directory")
    receipt = verify_archive(repo)
    a, b = structured.analyze(repo), raw.analyze(repo)
    rows = list(overlap_rows(repo))
    if len(rows) != 1024:
        raise ValueError("Wrong semantic overlap denominator")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "structured_analysis.json", a)
    write_json(out / "native_analysis.json", b)
    write_json(out / "comparison_contract.json", comparison_contract(repo))
    write_jsonl_gz(out / "structured_matched.jsonl.gz", structured.matched_rows(repo))
    write_jsonl_gz(out / "native_matched.jsonl.gz", raw.matched_rows(repo))
    with (out / "matched_primary_overlap.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    receipt["matched_semantic_overlap_rows"] = len(rows)
    receipt["native_free_readout_rows_present"] = sum(r["native_D_updated_present"] for r in rows)
    write_json(out / "verification.json", receipt)
    inputs = {**a["input_sha256"], **b["input_sha256"]}
    inputs[str(STUDY / "SHA256SUMS")] = raw.sha(repo / STUDY / "SHA256SUMS")
    inputs[str(STUDY / "gate.json")] = raw.sha(repo / STUDY / "gate.json")
    inputs["paper/main.md"] = receipt["paper_v1_sha256"]
    write_json(out / "input_manifest.json", dict(base_commit=BASE_COMMIT, files=inputs))
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=HERE.parents[1])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    receipt = run(args.repo, args.out)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
