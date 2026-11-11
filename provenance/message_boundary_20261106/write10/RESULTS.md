# Actual-output reports and learned reference retention (write10)

**A small reference memory can be learned from downstream judgment reward,
and the actual-output reports support the next update. The full registered
conjunction nevertheless fails because the reporter does not reliably
distinguish instruction questions from observation questions, and misses the
exact-naming threshold.** These are separate outcomes, not a revised gate.

Gemma generated 448 new reports. The actual-output question identifies the
realized off/off action in 224/224 records, including 176 erroneous transitions.
With both actions still present, the frozen reader correctly continues in
192/192 units per seed. The learned two- and 34-parameter reference cells both
meet their own gate in all three seeds, and both compose successfully with the
new reports across 512/512 four-update histories. Reference extraction and
pointer-role routing are still supplied; no Gemma weights are trained here.

The [plan](PLAN.md), code and [176 input bindings](INPUTS.json) were sealed at
`0d9f4c103c933615aa9af409dd46f961a3f83660` before scientific execution. There are
no additional seeds, prompt revisions or extended fits after seeing results.
Fresh generation is followed by frozen calculations using the bound native
row winners, not millions of new Gemma decisions. [Run commands](RUNS.md)
distinguish generation, fitting, composition and independent reduction.

## Actual reporting works; question selectivity does not

The panel contains 224 reachable transitions over four previously fitted
menus, both poles, two unfitted instructions and all feasible output tables.
Each transition receives two questions with identical context: which action
the **instruction assigned**, or which action the **table now contains** in
off/off. The original policy sentence is not supplied. The report writer sees
the rendered previous table, instruction assignments and actual current table.
The primary menus concern report recency, quotation versus summary, length
units, and mail sorting. This panel does not retest unseen action vocabulary.

| Report check | Instruction question | Actual-output question |
| --- | ---: | ---: |
| Correct actual off/off action, all transitions | 203/224 | 224/224 |
| Correct actual action, erroneous transitions | 155/176 | 176/176 |
| Correct action requested by the question, all transitions | 133/224 | 224/224 |
| Correct requested action when instruction and output conflict at off/off | 21/112 | 112/112 |
| Exact canonical requested action name | 116/224 | 198/224 |
| Missing action mentions / capped generations | 0/224 / 0/224 | 0/224 / 0/224 |

Every actual-output menu scores 56/56 for decoded action correctness. But only
**21/112 conflicting pairs (18.75%)** answer both questions correctly, against
the registered 90% requirement. On the instruction question, 91/112 conflicting
records name the actual output instead. Thus the matched question change
improves actual reporting, but does not establish reliable selection between
intended and observed states. A report that always describes the current table
would also fail this paired check.

Exact canonical naming is **198/224 (88.39%)**, below 90%. All 26 failures occur
in the recency menu: `earliest reports` (22), `earliest` (2), `most recent` (1)
and `most recent reports` (1), versus canonical names beginning with `the`.
These outputs are decoded correctly by the frozen lexical scorer, but the
registered exact-name condition remains failed. The scorer and normalization
are not changed after inspection. The [report gate](reports/report_summary.json)
therefore passes actual accuracy, conflict accuracy and every-menu accuracy,
but fails naming and paired question selectivity. These are machine-decoded
results, not independent human semantic annotations.

All 448 generations end at EOS after 3–7 new tokens; none reaches the
48-token cap. The fresh CPU float32 run takes 972 seconds and records 2,835
forwards, 2,834 generated tokens and 75,872 processed tokens. Both arms share
this execution policy; the old CUDA/BF16 reports are not a matched numerical
control. Two neutral preflight generations are separate from the 448 cases.

## A faithful report helps while both actions remain available

The next-step probe starts from each actual output table, then requests one of
two further assignments. It uses all three frozen write9 readers and their
supplied initial reference, with no fitting. The canonical-action oracle
replaces only the report text. Each strict unit covers four worlds and both
answer orders; the counts below hold separately in every seed.

| Starting table / incoming report | Complete next-step units |
| --- | ---: |
| Both actions remain / actual-output report | 192/192 |
| Both actions remain / instruction-question report | 176/192 |
| Both actions remain / canonical actual-action oracle | 192/192 |
| One action has been lost / actual-output report | 0/256 |
| One action has been lost / canonical actual-action oracle | 0/256 |

The noncollapsed continuation gate passes in all three seeds. This includes
erroneous, reversed tables; the improvement over the instruction-question bank
is 16/192 units (8.33 percentage points). Collapsed tables are still reported
correctly (128/128 actual-action reports), yet even the oracle cannot restore
the absent action through this row-copy interface. That is a distinct state
loss, not evidence that those reports omit the current action. No new native
scoring calls occur in this probe. The [saved judgments](reports/continuation_judgments.json)
and [summary](reports/continuation_summary.json) retain the failed stratum.
Empty unfitted-menu panels are marked 0/0 because that panel was not run.

## Learned retention: the small memory gate passes

Both final cells pass every registered retention criterion in all three seeds.
The two-parameter cell receives only the incoming reference bit; the
34-parameter cell additionally receives the report's 32 frozen hash features.
Only the second update's saved native-choice reward trains the first memory
write. Neither cell is trained against reference-bit labels. The language
model, report readers and role routing remain frozen.

The following counts hold separately for **each cell kind and each seed**
(17, 41 and 73), and match the engineered-copy comparator:

| Check, four fitted menus | Result |
| --- | ---: |
| Complete four-update histories | 1,229,312/1,229,312 |
| Histories using only unfitted masks | 41,472/41,472 |
| Complete first-update units | 448/448 |
| Histories after fixing the reference to zero | 614,656/1,229,312 |
| Histories after complementing the reference | 0/1,229,312 |
| Opposite action after reference inversion, conditional on a correct second update | 6,272/6,272 |
| Retaining both possible input bits over saved directed reports | 12,992/12,992 |

A history requires correct native choices at all four updates, four worlds and
both answer orders. Its large denominator repeats 32 source realizations and
14^4 instruction strings over four menus; it is not a count of independent
tasks. The reference intervention occurs after the first memory write. Later
writes consume only the incoming payload, without refreshing the source bit.
The pointer-role reversal is a supplied routing convention; the intervention
therefore checks causal use of that convention, not its discovery.

The untrained cells are retained in the audit. On the primary directed-report
bank, initial retention for the minimal cell was 6,496/12,992, 6,496/12,992 and
12,992/12,992. The third seed already implemented a correct hard-threshold copy
by chance; this is not counted as a new behavioral acquisition. Initial
report-conditioned retention was 4,784/12,992, 6,496/12,992 and 7,110/12,992.
All six final cells retain both incoming bits on all 12,992 directed and all
12,992 natural reports across eight menus: 25,984/25,984 bit trials per style.
The natural-report audit concerns copying a bit, not understanding the report.
It does not reverse write9's negative natural-language result.

All six fits use exactly 512 updates. Mean sampled native reward over the first
32 updates ranges from 0.677 to 0.697; over the last 32 it ranges from 0.933 to
0.946. Training retains a 0.12 exploration floor, whereas the evaluations use
deterministic thresholding. The weights, full curves and paired sampling hashes
are saved in [retention/](retention/). The three directed readers' parameter
hashes remain unchanged. All nine engineered-copy trajectories reproduce write9.

Better retention does not improve the unfitted-menu readout. Both learned
cells and the copy comparator give the same counts:

| Seed | Four unfitted menus | All eight menus |
| --- | ---: | ---: |
| 17 | 384,160/1,229,312 (31.25%) | 1,613,472/2,458,624 (65.625%) |
| 41 | 345,744/1,229,312 (28.125%) | 1,575,056/2,458,624 (64.0625%) |
| 73 | 384,160/1,229,312 (31.25%) | 1,613,472/2,458,624 (65.625%) |

The [retention gate](retention/gate.json) is restricted to the known menus.
The [complete summaries](retention/summary.json) retain the other menus and
all intervention conditions. Six fits and 27 full evaluations closed in
293 seconds, with zero new language generations or native-scoring calls.
The [fresh-process reduction](retention_reduction.json) reconstructs all
81 panels and nine paired comparisons exactly.

## Frozen composition succeeds within its smaller domain

After both component runs close, both fresh question banks are joined to the
engineered copy and the two final learned cells. There is no fitting on these
reports. The 54 evaluations cover both question banks, three memory choices,
three seeds and three reference conditions. Each covers 32 source realizations
crossed with all 16 four-update strings of the two selected instructions.

All memory choices and seeds give the following counts for **each** question
bank:

| Four-update check | Result |
| --- | ---: |
| Intact reference, complete histories | 512/512 |
| Reference fixed to zero, complete histories | 256/512 |
| Reference complemented, complete histories | 0/512 |
| Conditional action inversion, distinct two-update units | 128/128 |

Every final cell also copies both possible incoming bits on every fresh report:
448/448 bit trials per question bank. The two-parameter cell has no report input;
the 34-parameter cell retains the bit while receiving features of the fresh
reports. Fresh generation does not imply that every text or feature is unseen.
The scoped composition gate passes. The successful trajectories keep instruction
and output aligned, so both question banks tie here: this result does **not**
erase the failure to distinguish the questions on conflicting states. The
separate erroneous-state probe above provides the evidence about recovery.

The [combined gate](composition/gate.json) remains **false for both learned
cells**, because the reporting conjunction failed. Its other conjuncts—frozen
continuation, learned retention and scoped composition—pass. All trajectories,
intervention comparisons and fresh-report bit audits remain in
[composition/](composition/). The [fresh-process composition replay](composition_reduction.json)
reruns all 54 evaluations and reproduces every saved array, all 162 panels and
18 paired comparisons exactly, without a model call.

## Closure and next anchor

The bounded study is complete: 448 fresh scientific generations, six fits,
27 full-bank retention evaluations and 54 smaller composition evaluations.
The [report reduction](report_reduction.json) independently reproduces the 448
record scores and 5,376 frozen continuation judgments. The [validation receipt](validation.json)
binds the three reductions and the source-seal engineering checks. All 176
pinned files and the write6–write9 records remain unchanged.

This closes the question of whether a small binary copy path can be learned
from judgment reward in the existing scaffold. It also establishes accurate
machine-decoded reporting of the realized action on this panel. It does not
establish general instruction/observation discrimination, vocabulary transfer,
natural-report competence, learned source-reference extraction, learned routing,
Gemma hidden-state retention, large-scale RL improvement or raw-slot semantic
recurrence. The independent six-check qualification is not opened.

The next bounded anchor should hold the successful reader and memory fixed
and test **which state the report refers to**: intended assignment versus
observed output. Freeze new conflict examples and a question-selection criterion
before trying one matched reporting intervention; retain exact naming as a
separate requirement. A pass would admit question-sensitive reporting within
this interface. A failure would close that intervention and retain the narrower
actual-observation result, rather than extending fitting until the same panel
turns green. Recovery after losing an action needs a separately specified
reconstruction path; more faithful reporting alone did not provide one here.

[SHA256SUMS](SHA256SUMS) covers the study's local evidence, excluding itself.
The [artifact terms](ARTIFACT_LICENSE.md) accompany the model-derived arrays
and generated reports. The released v1 paper and DOI are unchanged.
