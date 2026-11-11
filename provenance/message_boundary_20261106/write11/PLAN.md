# Selecting the intended record versus the observed record (write11)

Prospective bounded development protocol, 2026-11-11. Parent:
`1205bc4675d9d5ec9edd7dc515d12c2642ff8ee7` (write10). Seal this plan, exact
prompts, input hashes, code and tests before generating any scientific answer.
The original independent qualification is neither opened nor used.

## Question and single intervention

Write10's actual-output question scored 224/224 by the frozen action decoder,
but only 21/112 conflicting question pairs answered both the instruction and
observation questions correctly. Exact canonical naming was 198/224. Keep
the successful report readers, both learned memory cells and their parameters
fixed. Test whether an explicitly supplied **record-selection cue** makes
reporting sensitive to the intended source of the question.

Both conditions show the same complete previous table, instruction record,
observed current table and the same two canonical action names. Both receive
the same request to copy exactly one complete action name. The two active
records have the same role labels in both conditions. The `copy_only` control
asks the natural-language question alone. The `source_cue` intervention adds:

> Requested record: INSTRUCTION RECORD. Read the off/off entry of that record.
> The other records are not the answer to this question.

The actual-output question substitutes `OBSERVATION RECORD` for the record name.
The program supplies the requested record identity, not the correct action,
selected row text, source pole or target bit. It leaves both conflicting tables
visible. No answer extraction, response repair, constrained decoding or extra
model call is permitted. All output text goes unchanged to the frozen scorer
and consumers. This is a typed question interface, not a claim that the model
learns to infer reference from arbitrary natural questions.

The common canonical-name list and labeled records are also scaffolds. The
matched contrast isolates the extra record-selection cue **within** that common
format. It does not isolate those common changes from the old write10 prompt.
Both conditions are freshly generated; write10's CPU float32 counts are
historical development evidence, not a matched CUDA/BF16 control.

## Fixed panel and controls

Use the same four fitted menus (0, 2, 4, 6), both poles, priors 8/3/12/0/15 and
instructions 3/12. The instructions are excluded from the frozen reader's
fitting masks and differ from write10's 6/9 generation panel. Enumerate all
feasible outputs: **224 distinct transitions**. None of their full transition
keys occurs in write10. The older exhaustive report bank contains these action
and mask combinations, so this is not globally unseen task content, new
vocabulary, or independent qualification.

Present each transition in both instruction-then-observation and
observation-then-instruction order, always retaining the previous table first.
Ask both questions in both conditions: **224 x 2 orders x 2 questions x 2
conditions = 1,792 fresh generations**. Rotate the four matched question/arm
conditions across presentations. Do not select or drop transitions after
generation. There are 112 distinct off/off conflicts, each tested with both
questions and both orders in each arm. Always copying the most recent record,
the observation, or the first listed action cannot pass those joint blocks.

The priors cover the full reachable closure of four updates on 3/12, including
collapsed states, so every later report lookup must be present. Retain the
same original action-bank meanings, pole balance and frozen lexical decoder.
There is no hidden original policy sentence supplied at the report boundary.

## Decisions fixed before execution

Score both arms with exactly the same decoder and canonical-name normalization
as write10. Retain raw text, fingerprints and stopping receipts. Report intended
and actual correctness, correct/erroneous outputs, conflict/nonconflict,
collapsed/noncollapsed, per-menu cells and both record orders.

For **absolute reporting admission**, each arm must pass all of the following:

- All write10 report criteria, pooled and separately in each order: >=95%
  actual correctness overall and on conflicting off/off assignments, >=90%
  exact actual-action naming, >=90% actual correctness in every menu, and
  >=90% paired question selectivity on conflicts.
- >=95% instruction-question correctness and >=90% exact instruction-action
  naming, pooled and separately in each order.
- >=90% of the 112 distinct conflicting transitions must answer **both
  questions correctly in both orders**, four answers per joint unit.

For an **attributable cue benefit**, the cue's joint-unit rate must exceed the
copy-only rate by at least 20 percentage points. Save both-correct, cue-only,
control-only and neither-correct paired counts. This benefit is a separate
claim from absolute admission. If both arms pass but the gain is small, the
shared format may suffice; do not credit the extra cue with a demonstrated
benefit. If the cue improves but misses admission, report an insufficient gain.

## Fixed downstream consumers

For each arm and table order, reuse write10's one-step continuation probe on
all three frozen write9 directed readers. Start from the **realized output**,
then request masks 10/5. Compare generated reports with canonical actual-action
reports. Require >=90% complete world/order units for actual-question and
oracle reports on noncollapsed tables in every seed and order. Publish the
collapsed stratum separately: this row-copy interface cannot invent a missing
action. Instruction-question reports intentionally refer to a different state
on conflicts; their downstream results are descriptive, not the actual-report
admission criterion. The probe makes no new native scoring calls.

Then compose each arm/order/question bank with engineered copy and both frozen
write10 memory cells, three reader seeds and intact/fixed-zero/opposed reference:
**216 evaluations**, each containing 32 source realizations x 2^4 strings =
512 four-update histories. Every learned cell uses only the incoming reference
and, if applicable, current report features; there is no source-reference
refresh after initialization. The old extraction and routing rules stay fixed.

For each memory kind, require >=90% complete actual-report histories, >=20-point
loss with fixed-zero reference and >=90% conditional action inversion in every
seed and record order. Learned cells must also retain both possible incoming
bits on >=99% of the fresh actual reports. Audit the instruction reports too,
without refitting. A full **scoped source-cued interface** claim requires the
cue's absolute report admission, its frozen continuation gate and that memory
kind's composition gate. Report cue benefit separately. Do not require the
copy-only arm to fail in order to admit a working interface.

These histories repeat a small menu bank. Report counts as exact coverage of
that bank, not as independent tasks or evidence of open-domain generalization.

## Compute, integrity and closure

Use the pinned Gemma-2-2B-IT snapshot, CUDA bfloat16, eager attention, a single
request at a time and the existing greedy KV generation policy. Use context
1,024, at most 48 new tokens and early EOS. Capped output is content-scored
normally and never discarded just for reaching the cap. Both arms share the
same hardware and decoding policy. Two neutral GPU timing prompts are unscored
preflight; no policy-bank prompt is used in preflight.

One GPU run, no fitting or prompt/seed search: at most 3,600 seconds of model-host
time, 100,000 forwards and 3,000,000 processed tokens. Release the GPU after
generation and compute frozen probes/composition on one CPU thread. Preserve
the first technical stop; budget exhaustion is incomplete, never a pass.
Inspect resource availability immediately before launch; do not displace
another workload. Use an isolated clean source worktree and external run output.

Close after all 1,792 generations, 21,504 frozen continuation judgments, 216
composition evaluations and one fresh-process reduction that regenerates every
summary and every frozen trajectory exactly. Verify all 311 pinned inputs
before and after the run. Preserve write6–write10, publish failed criteria and
scope the conclusion to the supplied record cue. If admission fails, close this
single intervention; no further tuning on the same panel. The released paper
and DOI remain unchanged.
