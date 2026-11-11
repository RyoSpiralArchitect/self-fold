# A record-selection cue is insufficient (write11)

**The extra record cue fails reporting admission and the registered matched
benefit.** Complete conflicting units rise from **50/112 (44.64%)** to
**57/112 (50.89%)**, a 6.25-point gain against a 20-point requirement. Each
unit requires correct answers to both questions in both record orders. The
frozen actual-report consumer still completes all four updates, but that does
not establish reliable reporting of intended versus observed states. Both
combined interface gates remain false.

The [plan](PLAN.md), exact prompts, tests and [311 input hashes](INPUTS.json)
were sealed at `2d835bf933db6b9e32c015cada23a9f20d9fb75d` before any scientific
generation. The study completed **1,792 fresh Gemma reports**, 21,504 frozen
continuation judgments and 216 four-update evaluations. No model, reader or
memory cell was fitted, and no new native judgment scores were computed.
This single intervention is closed without another prompt or seed search.

## What changed and what was held fixed

Both arms show the same previous table, instruction record and observed table,
with record labels and the two canonical action names. Both request exactly
one complete action name. The control, `copy_only`, asks the natural-language
question. The intervention, `source_cue`, additionally names the requested
record and says to read its off/off entry. It supplies a record identity, never
the target action, selected row text or target bit. Both tables remain visible,
and no response is repaired before scoring or consumption.

The 224 transition keys use four known menus, both poles and instructions 3/12,
which were excluded from the reader's fitting masks and differ from write10's
6/9 panel. Each transition is presented in two record orders, with two questions
and two arms. The priors include the complete reachable four-update closure.
These are freshly generated prompts, not globally unseen tasks: the old
exhaustive report bank contains the same menu/mask family. The previous policy
sentence is absent, but the report writer sees the rendered records.

The common labeled-record format and canonical-name list are themselves
scaffolds. This comparison isolates the additional record cue within that
format. Write10's raw-prompt, CPU-float32 results are historical motivation;
they are not a matched control for this new CUDA-bfloat16 panel.

## Reporting and the paired intervention

Pooled counts include both record orders. The same frozen scorer and naming
normalization are used for both arms.

| Check | Copy-only control | Explicit record cue |
| --- | ---: | ---: |
| Correct instruction-question answer | 362/448 (80.80%) | 373/448 (83.26%) |
| Correct actual-output-question answer | 419/448 (93.53%) | 419/448 (93.53%) |
| Correct actual answer on instruction/output conflicts | 199/224 | 199/224 |
| Both questions correct in a conflicting presentation | 128/224 | 145/224 |
| Both questions correct in **both** record orders | 50/112 | 57/112 |
| A valid canonical action name, irrespective of correctness | 896/896 | 896/896 |

Every output has the canonical form of one of the two actions. Format loss
has disappeared on this panel, but wrong actions remain wrong: the registered
**correct canonical naming** counts equal the requested-answer counts above.
In particular, instruction naming still misses its 90% threshold. Actual
accuracy misses 95%, conflict accuracy misses 95%, and joint selectivity misses
90%. The [report receipt](produce/report_summary.json) keeps all failed checks.

For the 112 paired joint units, both arms succeed on 47, the cue alone succeeds
on 10, the control alone succeeds on 3, and neither succeeds on 52. The net
gain is therefore (10 - 3)/112 = **6.25 percentage points**. This exact coverage
comparison does not treat the repeated menu examples as independent tasks or
claim a population effect.

Record order still matters:

| Order | Control: instruction; actual | Cue: instruction; actual | Control; cue paired selectivity |
| --- | --- | --- | --- |
| Instruction, then observation | 174/224; 208/224 | 177/224; 206/224 | 52/112; 57/112 |
| Observation, then instruction | 188/224; 211/224 | 196/224; 213/224 | 76/112; 88/112 |

The per-menu counts also prevent the pooled improvement from suggesting a
uniform gain. Each cell below contains 112 answers across both orders.

| Menu | Control: instruction; actual | Cue: instruction; actual |
| --- | --- | --- |
| Report recency | 95/112; 112/112 | 95/112; 112/112 |
| Exact quotation versus summary | 80/112; 91/112 | 76/112; 92/112 |
| Centimeters versus inches | 91/112; 112/112 | 103/112; 112/112 |
| Mail by sender versus date | 96/112; 104/112 | 99/112; 103/112 |

## Frozen consumers retain their narrower success

The one-step probe starts from the realized table and requests masks 10/5.
All three seeds give the same counts below. Each unit covers all four worlds
and both native answer orders.

| Report arm / table order | Actual-question report, noncollapsed | Canonical actual-action oracle |
| --- | ---: | ---: |
| Control / instruction then observation | 178/192 | 192/192 |
| Control / observation then instruction | 184/192 | 192/192 |
| Cue / instruction then observation | 180/192 | 192/192 |
| Cue / observation then instruction | 184/192 | 192/192 |

All meet the separate 90% noncollapsed continuation threshold. Collapsed
tables give **0/256** in every arm, order and seed, including the oracle: the
row-copy interface still cannot recreate the missing action. The instruction
reports refer to intended assignments and are not required to describe the
actual table on conflicts; their lower continuation counts are saved in the
[complete probe summary](produce/continuation_summary.json), not interpreted
as a standalone loss of instruction-report quality. Unfitted-menu panels are
0/0 because fresh generation did not include them.

For actual-report four-update composition, **every arm, record order, memory
choice and seed** gives the following result:

| Reference condition / check | Count |
| --- | ---: |
| Intact, complete four-update histories | 512/512 |
| Fixed zero after the first memory write | 256/512 |
| Complemented after the first memory write | 0/512 |
| Conditional next-step action inversion | 128/128 |

Each history count is 32 source realizations x 16 instruction strings, not 512
independent tasks. The engineered copy and the two frozen learned memory cells
all preserve the reference, and both learned cells retain both possible input
bits on **448/448 fresh bit trials per question/arm/order bank** in every seed.
Initial reference extraction and role routing remain supplied.

The intended and actual assignments agree along these intact actual-report
trajectories. Their perfect score therefore does not override failures on the
conflicting report panel. Instruction-report composition is also retained:
the control gives 480/512 and 448/512 intact histories in the two orders; the
cue gives 448/512 and 408/512. These counts hold for each memory choice and seed.
No arm was refitted after viewing this behavior. The [combined gate](produce/gate.json)
keeps reporting false, continuation true, and both learned-cell combined
criteria false, even though the actual-report composition gates pass.

## Post-result diagnosis: record selection is not the entire error

This diagnostic was added **after** the run and does not change any admission
criterion. Select records where the previous, instruction and observed
off/off entries all name the same action. A wrong answer on this subset cannot
be explained solely by choosing the wrong one of those three off/off entries.
This all-three agreement split differs from the primary instruction/output
conflict split.

| Agreement subset | Control | Record cue |
| --- | ---: | ---: |
| Instruction-question correctness | 116/128 | 112/128 |
| Actual-question correctness | 124/128 | 124/128 |

For example, in transition `2:0:3:3:3`, all three off/off entries say
`summarize the main ideas`, but the cue's instruction answer in the
instruction-then-observation order is `preserve the exact words`. Both are valid
action names. The finding leaves coordinate lookup, action selection and
context interference as possibilities; it does not identify which internal
mechanism caused the error. [Every agreement-subset error and input hashes](posthoc_agreement.json)
are retained. The [small audit script](../../../scripts/audit_source_selection.py)
reproduces this descriptive split without a model call.

## Closure and next decision

All 1,792 generations end at EOS after 3–7 new tokens; none is censored.
Generation takes 177 seconds and the complete run 185 seconds. The host records
11,272 forwards, 11,271 generated tokens and 375,061 processed tokens. Two
neutral preflight generations are separate. The availability wrapper admits
the run only after two empty GPU-process observations; its final wait is
recorded in [resource-wait.jsonl](resource-wait.jsonl). No other workload is
stopped or modified.

The [fresh-process reduction](reduction.json) reproduces all 1,792 scores,
21,504 continuation judgments, 216 frozen trajectories, 648 composition panels
and 72 paired reference comparisons exactly. It makes no new generation or
native-scoring calls. The [validation receipt](validation.json) separates that
computational check from engineering tests and the failed scientific gates.
All 311 prior bindings and write6–write10 artifacts remain unchanged.

The present cue is an insufficient intervention, so this branch of development
closes without more wording trials on the same panel. Before further reporter
learning, the next bounded experiment should separate **reading the requested
row** from **selecting a record among distractors**. A program-supplied correct
single-record condition can serve as an explicit oracle context control; it
must not be advertised as learned record selection. If it reads reliably while
the full context fails, a learned selection interface becomes a justified next
target. If even the single record fails, row/action readout must be repaired
first. Freeze new cases and those decision rules before the next live run.

This result does not establish autonomous source resolution, general semantic
recurrence or an independent six-check pass. The released v1 paper and DOI are
unchanged. [Run commands](RUNS.md), [SHA256SUMS](SHA256SUMS) and
[artifact conditions](ARTIFACT_LICENSE.md) accompany the complete record.
