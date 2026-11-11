# Actual-output reporting and learned reference retention (write10)

Prospective bounded development protocol, 2026-11-11, before scientific
generation or fitting. Parent: write9, `44b106a`. The initial source reference
and the reader's role-exchange convention remain supplied. Human transition
review and the original independent qualification are not run or opened.

## Two separate questions and one frozen composition

Write9 demonstrated directed-report use with an engineered reference copy.
It did not establish faithful reporting of erroneous realized outputs or a
learned retention mechanism. Test these separately, then compose the final
fixed components without further fitting or selecting seeds.

### A. Does changing the question make the report track the realized output?

Use the same four fitted action menus (0, 2, 4, 6), both source poles, priors
8/6/9/0/15 and instructions 6/9. Enumerate every `feasible_outs(prior, mask)`:
224 distinct transitions. Generate **both** the original instruction question
and an actual-current-table question for each: **448 fresh reports**. Alternate
the order of paired questions. The table/instruction context is identical.
The priors form the complete reachable closure for four updates on these two
instructions, including collapsed output states. This permits a later exact
four-update comparison, rather than an incomplete report lookup. Vocabulary
transfer is not retested by this generation panel; write9's limitation stands.

Use the pinned Gemma-2-2B-IT checkpoint, CPU float32, six threads, eager attention,
greedy per-request KV decoding, context 1,024 and maximum 48 new tokens. Stop
early on EOS; retain censored, empty and imperfect reports and score their
content normally. Do not retry prompts or tune wording. Both question arms are
generated afresh under this identical execution policy: the earlier CUDA/BF16
reports are not a numerically matched comparator. A separate unscored CPU
preflight used two neutral timing prompts, no policy-bank cases. No GPU work is
started while the other Furnace training job is active.

Freeze the lexical decoder. Report actual-action correctness, instruction-action
correctness, exact canonical-action naming, missing mentions and censoring.
Separate correct outputs, erroneous outputs, off/off instruction/output
conflicts, collapsed/noncollapsed tables and each menu. No human semantic
annotation is implied. The reporting gate requires actual correctness >=95%
overall and on conflicting off/off assignments, >=90% strict canonical naming,
>=90% actual correctness in each menu, and >=90% **paired question selectivity**:
the instruction question selects the instructed action and the actual question
selects the different realized action. Report each check if the conjunction
fails; an actual-only success is not full question selectivity.

Probe each final write9 directed reader without fitting. Starting from the
realized table, use the report plus the same initial source reference, then ask
for masks 10/5. Score against bound saved full-vocabulary native row winners.
Also substitute the correct canonical action name. The separate continuation
gate requires >=90% complete world/order units for generated and oracle reports
in every seed on **noncollapsed** tables. A pointer cannot reconstruct an action
absent from a collapsed table; report that stratum without treating it as a
report-only defect. This probe makes no new native scoring calls.

### B. Can judgment reward teach a small cell to retain the reference?

Freeze all three write9 directed readers, their feature map, and role routing.
The initial reference is extracted once per episode from its compiled action
rows. Replace direct copying with either a two-parameter logistic cell of the
incoming bit, or a 34-parameter cell that also receives the 32 report features.
The cell emits exactly one binary bit. Initialization is random, not an identity
map. The two cells share initial bit/bias coordinates and sampling schedules.
Neither sees source keys, action targets, future instructions or reference
labels as training targets. Only the second update's archived native-choice
reward trains the first memory write by a Bernoulli score-function estimator.
No gradient changes the frozen reader. This is a small binary-memory test,
not training Gemma to retain a latent state.

Six fits: two cell kinds x seeds 17/41/73; 512 Adam updates, batch 128, learning
rate .06, exploration .12, norm clipping 1. Fit on the original two-update
training-source/mask bank; use final checkpoints only. All reports and native
scores for fitting come unchanged from write7; no fresh report enters fitting.
Audit both possible incoming bits on every saved directed and natural report,
before and after fitting, separating fitted/unfitted menus. This audits the
copy operation, not task semantics. Natural reports are a transfer diagnostic.

Evaluate the two learned cells and engineered copy across all 14^4 four-update
strings, all 64 test sources, three seeds and intact/fixed-zero/opposed reference:
**27 evaluations**, three panels each. Intervene after the first memory write;
subsequent references must come only from the incoming payload through the cell.
Check all nine engineered-copy trajectory arrays against write9. Save full
packed output patterns, reference traces, pre-intervention copy decisions,
weights, curves, parameter/schedule hashes and per-array logical hashes.

For each cell kind, every seed must reach >=90% primary four-update and
unfitted-mask histories, >=99% directed-report reference retention on fitted
menus, >=20 percentage-point loss under fixed-zero reference, initial 448/448
units, and >=90% conditional action inversion among correct second-update units.
The last check exercises the pre-existing routing convention, not emergent
reference semantics. The four unfitted menus and all eight are always reported.

### C. Compose the frozen components on the complete new report bank

After A and B close, evaluate engineered copy and both final learned cells on
both freshly generated question banks, every 6/9 four-update string, the 32
test-source realizations of the four menus, all seeds and all reference modes:
54 compact evaluations, each with 512 histories. No fitting on these reports.
Lookups use actual prior/instruction/output; missing coverage is an error.
Record four-update correctness, per-depth correctness, reference traces and
paired inversion. The scoped composition gate requires every seed for a cell
kind to reach >=90% actual-report complete histories, >=20-point fixed-zero
loss and >=90% conditional inversion. A full combined claim additionally
requires A's reporting/continuation gates and that kind's B gate. Keep partial
successes and failures explicit; do not redefine the conjunction after results.
Also audit both possible incoming reference bits on every newly generated report
as a retention-transfer diagnostic, without changing any learned weights.

## Execution, closure and evidence

Use isolated worktrees. A uses at most six CPU threads and one hour of measured
model-host wall time; B uses one Torch thread per worker, at most two workers
and one hour. Preserve the first technical stop; no seed, prompt, feature,
threshold or budget search. Budget exhaustion is incomplete, never a scientific
pass. Old artifacts and all pinned inputs remain unchanged. Commit the plan,
code and tests before scientific execution; keep outputs outside the worktree.

Close after A's 448 generations, B's six fits/27 evaluations, C's 54 evaluations
and a fresh-process replay from saved reports, trajectories and native scores.
Bind the A/B run directories into C by file hashes. Raw report text, generation
receipts and all failed strata remain available. Combinatorial histories and
repeated report records are not independent tasks. No result establishes
general semantic recurrence, a raw eight-slot G update, large-scale RL gains,
learned source-reference extraction, or independent six-check qualification.
