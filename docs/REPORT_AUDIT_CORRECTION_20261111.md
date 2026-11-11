# Report audit and generation-boundary corrections

The pre-merge fork review of PR #35 found two implementation defects. These
corrections do not replace the saved write6/write7 reports, write8 fits or
historical stopping receipts. No training or language generation was rerun.

## Report counts versus decoder counts

The audit's main consensus categories already counted each report once.
Its unique-form display mixed those report categories with individual decoder
outcomes, however. A resolved report could appear in the displayed `missing`
count when the ordinal decoder did not recognize its lexical answer.

The corrected display counts each report once under its full consensus
category. Recomputing all 40,320 write6 reports leaves the main category totals
unchanged. Directed reports have **0 unresolved reports and 0 unresolved
forms**, rather than the misleading display of 11,560 `missing` reports.
Natural reports have **6,936 unresolved reports and 333 unresolved forms**;
the old mixed-unit display counted 11,475 `missing` reports. None of these
decoder measurements establishes writer omission.

[Original and corrected display comparison](../provenance/message_boundary_20261106/review_corrections_20261111/audit_comparison.json)
binds both script versions and output files. The original corpus and its
[artifact conditions](../provenance/message_boundary_20261106/write8/ARTIFACT_LICENSE.md)
are retained.

## The last allowed token must remain a valid report

The cached greedy loop performed another forward after the final allowed
non-EOS token. Near a forward, token-compute or wall-time boundary, that unused
forward could raise an error instead of returning the completed censored
report. The loop now forwards only when another token decision is needed and
does no prefill when zero output tokens are allowed. It also adopts each
returned cache, supporting caches that are replaced rather than mutated.

Regression fixtures check length and context caps, zero budgets, EOS,
immutable caches and a genuinely incomplete generation. Existing historical
reports and compute receipts describe their original execution; they have not
been recalculated as if this repair had been present.

## Refit wall-time contract

The write8 runner records the design's one-hour wall budget but does not
enforce it internally. Its recorded execution completed in 1,298 seconds.
Future reproductions on Furnace/Linux **must use an external process-group
timeout**, covering the runner and its evaluation workers:

```sh
timeout --kill-after=5s 3600s env PYTHONPATH=src OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 python3 scripts/replay_lm_writer.py \
  --out <new-output-directory> --workers 2
```

This command requires GNU `timeout`. The five-second termination grace is
not an extension of the scientific budget. A timeout leaves an incomplete
run; preserve it and do not resume into the same directory. The original
sealed plan remains available as the historical record.
