"""Produce six read-only evidence views and a non-admitting development record."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from summarize import CHECKS


def encoded(obj):
    return (json.dumps(obj, sort_keys=True, indent=2) + "\n").encode()


def build(root):
    routes = {
        "structured_primary": ("structured", "structured-summary.json"),
        "structured_reworded": ("readout", "readout-summary.json"),
        "raw_native": ("raw-bound", "raw-summary.json"),
    }
    records = {}
    entries = {}
    for check in CHECKS:
        path = "checks/" + check + ".json"
        payload = {
            "scope": "mechanical_development_only",
            "check": check,
            "interfaces": {},
        }
        for key, (folder, name) in routes.items():
            result = json.loads((root / name).read_bytes())
            payload["interfaces"][key] = dict(
                measurement=result["checks"][check],
                execution_binding_sha256=sha256(
                    (root / folder / "binding.json").read_bytes()
                ).hexdigest(),
                reduction_sha256=sha256((root / name).read_bytes()).hexdigest(),
            )
        records[path] = encoded(payload)
        entries[check] = dict(
            verdict="not_run",
            scope="original_independent_qualification",
            evidence_path=path,
            evidence_sha256=sha256(records[path]).hexdigest(),
        )
    records["gate.json"] = encoded(
        dict(
            status="NOT_QUALIFIED",
            split="development_only",
            reviewer=None,
            qualification_opened=False,
            human_transition_review="deferred",
            checks=entries,
            qualification_commitment_sha256="b408768d82296c367cc686fdd608b749ce4da911d1dfcdc099f91bf261c19997",
        )
    )
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("study", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    for name, content in build(a.study).items():
        out = a.output / name
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("xb") as f:
            f.write(content)
