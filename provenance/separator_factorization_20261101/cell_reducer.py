"""Reuse frozen raw-score reducers on one explicitly grouped factor/display cell.

No model imports, no raw-data writes, and no changes to the published reducer.
The whole-study planned-grid/contrast wrapper is separate execution work.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path


FROZEN = Path(__file__).resolve().parent.parent / "downstream_span_20261101/summarize.py"
SPEC = importlib.util.spec_from_file_location("frozen_downstream_span_metrics", FROZEN)
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)


def validate_cell(records, topics):
    """Allow explicit missing topics, but never duplicates or mixed display cells."""
    topics = set(topics)
    seen, cells = set(), set()
    for row in records:
        key = row["group"], row["pole"], row["plus_label"]
        if key in seen or key[0] not in topics or key[1] not in ("plus", "minus") or key[2] not in ("A", "B"):
            raise ValueError("Duplicated/unplanned topic-pole-order row")
        seen.add(key)
        if "cell" in row:
            cell = {k: v for k, v in row["cell"].items()
                    if k not in ("group", "plus_label", "layout_sha256", "score_index")}
            cells.add(tuple(sorted(cell.items())))
    if len(cells) > 1:
        raise ValueError("Mixed factor/display cells must be reduced separately")
    complete = {t for t in topics if {(t, p, o) for p in ("plus", "minus") for o in ("A", "B")} <= seen}
    return complete


def reduce_cell(records, select, *, topics):
    topics = set(topics)
    complete_rows = validate_cell(records, topics)

    def check_scores(scores):
        if scores is not None and (set(scores) != {"A", "B"} or not all(math.isfinite(x) for x in scores.values())):
            raise ValueError("Invalid/nonfinite raw label log probabilities")
        return scores

    def checked(row):
        return check_scores(select(row))

    for row in records:
        for reference in ("native", "suffix"):
            check_scores(row["reference"][reference]["scores"])

    # The old reducer requires complete four-row topics; keep their missingness
    # visible outside the legacy result rather than padding them with invented rows.
    measured = {t for t in complete_rows if all(checked(r) is not None for r in records if r["group"] == t)}
    result = legacy.panel_metrics([r for r in records if r["group"] in complete_rows], checked)
    return {"planned_topics": len(topics), "complete_topics": sorted(measured),
            "missing_topics": sorted(topics - measured), "primary_complete": measured == topics,
            "legacy_metrics": result}


def context_select(name, site, method):
    if method not in ("raw", "delta"):
        raise ValueError("Undeclared transfer route")
    return lambda row: legacy.context_scores(row, name, site, method)


def mapping_conflicts(records, *, context, site, method):
    # The frozen function expects 'reversed'. This view never mutates archived
    # records and works for mapping_flip, display_flip, or a named context.
    view = [{**row, "contexts": {**row["contexts"], "reversed": row["contexts"][context]}} for row in records]
    return legacy.mapping_conflicts(view, site, method)
