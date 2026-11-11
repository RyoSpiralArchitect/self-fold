"""Post-result input-alias diagnostic; no fitting, model calls or gate change.

On the saved correct-transition cells (out == instruction), count opposite
role targets sharing the current pointer reader's observation. These finite-bank
majority bounds concern classification of the off/off role, not all-four
trajectory accuracy. They are not bounds on natural language in general.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from selffold.report_replay import read, verify_files
from selffold.roundtrip import text_features

ROOT = Path(__file__).resolve().parents[3]


def diagnose():
    manifest = read(Path(__file__).with_name("INPUTS.json"))
    verify_files(ROOT, manifest["files"])
    old = ROOT / manifest["archive"]
    result = {}
    for style in ("rt_directed", "rt_natural"):
        records = [json.loads(line) for line in (old / f"reports_{style}.jsonl").read_text().splitlines()]
        requests = read(old / f"reports_{style}.binding.json")["requests"]
        sources = {r["source"]: (r["group"], r["pole"]) for r in requests}
        by_style = {}
        for panel in ("initial_prior_8", "all_nonconstant_priors"):
            tables = {name: defaultdict(lambda: [0, 0]) for name in ("text", "hashbag")}
            examples = {name: {} for name in tables}
            count = 0
            for row in records:
                if row["out"] != row["mask"] or row["prior"] in (0, 15):
                    continue
                if panel == "initial_prior_8" and row["prior"] != 8:
                    continue
                role = row["out"] & 1
                # For two distinct hard members, this is exactly the row
                # contrast normalized by BoundaryMemory.read. Absolute
                # action identities and source pole are absent from that read.
                relative = tuple(((row["out"] >> i) & 1) ^ role for i in range(4))
                features = text_features(row["text"]).numpy().astype(np.float32)
                keys = dict(text=(relative, row["text"]),
                            hashbag=(relative, hashlib.sha256(features.tobytes()).hexdigest()))
                for name, key in keys.items():
                    tables[name][key][role] += 1
                    examples[name].setdefault(key, {}).setdefault(role, dict(
                        source=row["source"], group=sources[row["source"]][0],
                        pole=sources[row["source"]][1],
                        prior=row["prior"], mask=row["mask"],
                        out=row["out"], text=row["text"], required_role=role,
                        relative=list(relative), request_sha256=row["request_sha256"]))
                count += 1
            by_style[panel] = {}
            for name, table in tables.items():
                conflicts = [v for v in table.values() if min(v) > 0]
                by_style[panel][name] = dict(
                    records=count, distinct_inputs=len(table),
                    contradictory_inputs=len(conflicts),
                    records_in_contradictory_inputs=sum(sum(v) for v in conflicts),
                    best_deterministic_role_correct=sum(max(v) for v in table.values()),
                    unavoidable_role_errors=sum(min(v) for v in table.values()),
                    contradictory_witness=next((list(v.values()) for v in examples[name].values()
                                                if len(v) == 2), None))
        result[style] = by_style
    return dict(scope="post_result_saved_correct_transition_input_aliases",
                unit="unique_pair_prior_instruction_output_report_record",
                assumption="equal record weights; deterministic role classifier using normalized row contrast and the stated report representation",
                input_contract=dict(
                    implemented_reader="BoundaryMemory.read: normalized row contrast and hashbag_v1",
                    text_view="counterfactual exact report text with the same normalized row contrast",
                    unavailable_to_pointer=["source pole", "absolute action identity", "prior instruction"],
                    decoded_bit_comparator="uses action options and source pole outside this pointer reader"),
                not_an_all_four_bound=True, results=result)


if __name__ == "__main__":
    print(json.dumps(diagnose(), indent=2, sort_keys=True))
