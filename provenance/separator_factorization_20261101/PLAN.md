# Separator identity, displayed order, causal access and readout distance

**Prospective manipulation plan, 2026-11-01 JST. No new model forward or GPU
experiment has been run for this plan.** Freeze this plan, `design.json`, the
compiler/checker, and tests in a clean commit before collecting any new scores.
The machine-readable [preflight receipt](verification.json) describes what has
actually been checked. It is not an execution-admission receipt.

## 1. Starting observation and protected scope

The [frozen localization results](../downstream_span_20261101/RESULTS.md) found a
mapping-sensitive contribution at the second displayed option's period. This
plan asks which controlled changes affect that contribution; it does not assume
that punctuation forms a semantic summary or that the period is necessary.
The base is `6b054baf2571c3fa0e4f19d5d841ea760d3eaba2` (merged PR #4).
Every file already tracked at that commit, including the paper, plans, raw
trials, summaries and checksums, remains byte-identical. New files are additive.
No result here can retroactively change the frozen study's denominator or claims.

Use the same pinned Mistral and Gemma snapshots, BF16, eager attention, batch one,
no quantization, no KV cache, eight source slots, and inputs to decoder layers 16
and 13 respectively (zero-based). Reuse all six original and all six formerly
prospective topics. Report those panels separately. **Both panels now have
exposed outcomes**: only these new manipulations are prospective. This is not a
new-topic confirmation study. No topic or checkpoint is selected by new scores.

## 2. What can be identified

| Contrast | Held fixed | Changed | Identified scope / remaining limitation |
| --- | --- | --- | --- |
| Donor punctuation | Recipient IDs, anchor, length, mapping, display, score | Only the donor's second separator ID | Sensitivity of the extracted lexical difference to `.` versus `?`; token identity and its learned function remain coupled |
| Recipient punctuation | Donor residual/difference, anchor, length, mapping, display, score | Only recipient separator ID/background | Readout sensitivity to the recipient context; separate from donor extraction |
| Display order | Mapping of words to A/B, cell widths, anchors, score | Whole labeled cells printed AB versus BA | Ordinal versus label dependence within this scaffold; no architecture-level claim |
| Mapping reversal | Display order, cell widths, anchor IDs, score | Words assigned to A/B | Source-letter-sensitive versus recipient-remapped contributions |
| Access / receipt | All token IDs and all positions, recipient forward | Donor attention edges below the cut | Contribution of specified causal paths, not unrestricted evidence about meaning |
| Early versus late release | Same late anchor and its direct access to both cells | Whether intervening non-cell positions could receive the cells earlier | Earlier indirect paths; does not uniquely establish summation, a semantic accumulator, or one formation layer |
| Distance grid | Full forward length, option lengths, within-pair anchor prefix | Front and post reservoirs | Fixed-anchor, fixed-distance and equal-score comparisons; not three independent absolute/relative coordinates |

For anchor index `a`, score index `s`, distance is exactly `d = s - a`.
The columns `a`, `d`, `s` therefore cannot all be independent. The checker must
report rank deficiency for that three-coordinate model. No regression is allowed
to relabel this identity as a separately estimated causal effect of distance.
Masking is an intervention on computation and introduces its own distribution
shift. Matched mask shams and rebasing reduce particular alternatives; they do
not prove that all such shifts have been eliminated.

The blocks are deliberately bounded, not a saturated four-factor experiment:
punctuation is crossed with display/mapping in S, while A and D hold punctuation
at period; A uses `(u,v)=(0,0)` and D uses open access. Punctuation-by-release,
punctuation-by-distance and release-by-distance interactions are **not estimated**.
The rank-16 check refers only to source punctuation x target punctuation x display
x mapping at the second separator, not to every proposed factor jointly.

## 3. A fixed token scaffold and a bridge to the natural template

Use the archived native token IDs and offsets in
`../downstream_span_20261101/trials/clauses-{panel}-{model}/prompts.jsonl` and
the corresponding `span_settings.json`. `verify.py` compiles every layout without
model weights, tokenizer downloads, or GPU access. It verifies both archived
orders, single-character separator offsets, unfused periods, label-scaffold
widths, all bounds and all source-slot lengths. Punctuation has exactly two
predeclared levels, the native single tokens `.` and `?`. No punctuation is chosen
by its measured effect. The question-mark IDs come from the archived question
stems; they are not guessed vocabulary IDs.

For each topic retain the native prefix through the question, ending immediately
before the first labeled cell. Append the following **token-ID segments**:

1. A 16-token neutral shadow bank, followed by `u` neutral front tokens.
2. Two labeled option cells in display order AB or BA. Each uses its archived
   two-token label/equals scaffold, then an eight-token word cell, then a single
   separator. The phrase is right-aligned in the word cell, with the archived
   neutral placeholder on its left. The first separator is always `.`. The
   second is `.` or `?`.
3. Four neutral tokens, a standalone period (`receiver_early`), 15 neutral
   tokens, and another standalone period (`receiver_late`). These periods have
   the **same ID as the option periods**, but neither is an option word.
4. `v` neutral post tokens, the unchanged native closing instruction/chat suffix,
   one neutral dummy, and `32-u-v` neutral compensation tokens.

Use `u,v` in `{0,16}` only. All layouts of one topic/checkpoint have exactly the
same full forward length. Word cells, source slots and every scored window have
explicit absolute, zero-based, half-open indices. The score index is the last
native closing token, immediately before its dummy. It is **not necessarily
`len(ids)-2`**, because compensation follows the dummy. The existing `_capture`
and `_replace_span` helpers hard-code `-2`: a future runner must introduce and
test an explicit score-index adapter, never call these unchanged for this grid.
Trailing tokens are causally after the score and must not be read as task input.

This is an explicitly synthetic fixed-width scaffold. Concatenated native token
segments are the stimuli; they need not equal a fresh tokenization of their
decoded text. Store both IDs and native-tokenizer decoded text at execution,
segment provenance/offsets and padding spans. Never report synthetic offsets as
native whole-prompt offsets. Padding, moved scaffolds and `?` can change the task's
interpretation. Successful effects here do not by themselves generalize to normal
prose, other punctuation, unseen topics or other cuts.

The **bridge** separately repeats both original natural-template orders. At the
two original period anchors collect native/neutral/full-suffix references,
single-site only measurements and exact reconstruction/null checks. Compare all
24 donor/orders per model/panel with their archived label log probabilities.
Any nonzero discrepancy fails the bridge; report it without changing the old
results or calibrating a new tolerance. A weak or absent effect in the padded
scaffold is a failed transfer to that scaffold, not a refutation of the old result.

## 4. Measurement blocks fixed before outcomes

### S: surface, labels, and ordinal position

At `u=v=0`, cross second punctuation (two levels), display order (AB/BA), and
word-to-label mapping (plus=A/B): eight layouts per topic/checkpoint. Measure
four one-token sites: first separator, second separator, early receiver and late
receiver. For both lexical donor poles perform **only**, **omit**, and **native
knockout**, using the existing definitions. Keep every window even if its effect
has the unexpected sign. These operations are not assumed additive.

For each layout transplant from the other mapping, other display order, and both
flipped, at all four sites. Record raw and rebased reads. A mapping flip keeps
the printed labels in place and swaps their words. A display flip moves labeled
cells while preserving each label's words. Thus, `plus_label` always names the
plus **label**, not the first displayed cell. Do not average AB with BA until the
display-specific effect and interaction have been reported.

Fully cross donor and recipient second punctuation at the second anchor. The
matched cells are already in S; add the opposite-punctuation donor for each
recipient, with both raw and rebased reads. For cross-context comparisons use
the lexical difference as the primary route:

`recipient_neutral + (source_lexical - source_neutral)`

Compute in FP32 and cast once to BF16 at injection, as before. Raw reads remain
mandatory secondary evidence. Source/recipient anchor-ID inequality is allowed
**only in the explicitly tagged punctuation cross**; all other context swaps
require equal anchor indices, equal anchor IDs, equal full forward lengths and
equal score indices. Do not move a residual to rescue an incompatible condition.
The first anchor is a causal negative control for the later punctuation change:
its donor residuals must be exactly equal across the two punctuation sources.
For a matched source/recipient background require the rebased BF16 tensor to be
bit-identical to the original lexical tensor before reusing its raw measurement
as the matched delta cell. A failure invalidates that reuse/job; it does not
authorize omitting the matched delta or adding an unbudgeted forward.

### A: move the first allowed receipt without moving tokens

Use the four period layouts at `u=v=0`. Let `K` be both eight-token word cells,
including their padding. The same fixed spans are protected under both mappings;
mask shape therefore cannot disclose phrase length or the mapping. Labels,
separators, source slots and question stay in place. Donor-only masks operate in
**every layer 0 through cut-1**. For an otherwise permitted causal edge from key
`k` to query `q`, remove it exactly when:

`k in K and q not in K and q < release_index`.

For `release_early`, the index is `receiver_early`; for `release_late`, it is
`receiver_late`; for `never`, it is the full forward length. At the release index
itself access is allowed. All other edges retain the checkpoint's normal causal
and local-attention restrictions. The mask is an additional restriction, never
an override that opens a forbidden edge. All masks are removed from layer `cut`
onward. The ordinary recipient is unmasked throughout.

Blocking only a probe's direct edges in its last layer is **not sufficient**:
an intervening token could relay the hidden content. The specified restriction
closes every outgoing path to an earlier non-cell query. The checker propagates
two independent content-source flags through the lower-layer causal graph and
verifies this closure. It also demonstrates the leak in a direct-only blocker.
An adapter must verify the same property numerically before a GPU outcome run.

For each release use a mask sham with the same rule and 16-key width on the
neutral shadow bank instead of K. This preserves edge counts per external query
once both banks are in the past; the key positions and some early-query counts
differ and must be reported. These are matched-size shams, not identical causal
positions. `open` is the existing unmasked S donor. In all six new contexts
capture a lexical donor for each pole and its **own mask-matched neutral donor**.
At the second separator and both receiver sites transfer raw and rebased states
into the same open recipient. Never subtract an open neutral from a masked source.

The late receiver has direct access to both cells under both early and late
release. Only early release allows earlier non-cell positions to have received
them. An early-minus-late effect can therefore implicate those indirect paths
within this computational intervention. At a pre-release site the option-cell
**values** have no path to its captured residual. This does **not** imply a zero
endpoint pole gap: the visible lexical source slots can carry donor information,
and upper recipient layers can reinterpret it using the recipient's options.

### D: transport distance with a balanced length reservoir

Use period punctuation, both displays and both mappings at all four `(u,v)`
settings. The four `(0,0)` layouts reuse S; twelve layouts are additional.
Measure only/omit/native-knockout at the second separator and late receiver.
Repeat the matched references and nulls for each added layout.

For either site, write its `(0,0)` coordinates as `(a0,d0,s0)`. Then:

| Front u | Post v | Anchor a | Distance d | Score s |
| --- | --- | --- | --- | --- |
| 0 | 0 | a0 | d0 | s0 |
| 0 | 16 | a0 | d0+16 | s0+16 |
| 16 | 0 | a0+16 | d0 | s0+16 |
| 16 | 16 | a0+16 | d0+16 | s0+32 |

Compare v within each u (fixed anchor and unchanged anchor prefix), u within
each v (equal distance), and the middle two cells (equal score index). Require
bit-exact equality of the donor pre-cut residual through the late receiver
across v at each fixed u. Forward length and corresponding anchor indices are
equal in that check. Use that identical captured donor at each v. A difference
therefore concerns later transport/readout, not a change to that donor residual.
It still changes absolute score position and intervening filler context. Do not
claim a distance-only mechanism or interpret the front shift as a pure position
intervention; it also changes the preceding padding.

## 5. References, nulls and execution gates

Fix Python, NumPy and Torch seeds to 20261101, disable dropout with eval mode,
enable deterministic algorithms, set `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and
`TOKENIZERS_PARALLELISM=false`. There is no sampling. Execute models in the
declared JSON order, panels in the declared order, topics by sorted ID, and
within-topic factor cells in `compile_all` order. Within a layout use the
reference/operation order in the budget description and the declared context,
site, pole (plus then minus) and route (raw then delta) orders. No ordering is
selected from timings or scores. Record actual versions/environment with each job.
Capture all eight S layouts for a topic before computing its cross-context swaps;
reuse those owned, immutable captures instead of adding new source forwards.
Free the topic's transient captures before advancing to the next topic.

For native execution match Python, Torch CUDA and the recorded package versions
in `design.json`'s frozen runtime reference: Python 3.14.4, Torch
2.13.0+cu132, Transformers 4.57.1, NumPy 2.5.4, Accelerate 1.14.0, PyYAML
6.0.3 and pytest 9.1.1, with Torch CUDA 13.2. The checker verifies this reference
and all four archive identities; it does not claim that its own CPU interpreter
or installed packages match that GPU environment. Record GPU/driver identities,
retain the profile admission floors, and require the exact bridge on the actual
hardware. A runtime change requires a prospective amendment before new scores.

Every S layout costs at most 37 forwards: neutral and two lexical captures (3),
whole-suffix neutral replay (1), two combined slot-plus-suffix reconstructions
(2), two full-suffix references (2), four sites x two poles x three interventions
(24), four neutral single-site self-replays (4), and one empty-hook sham (1).
All 37 are specified even where values could be reused. The empty hook installs
and removes the injection machinery but changes no position. Zeroing a residual
is not a null intervention. Compare all null and reconstruction checks across
the entire score-position vocabulary with tolerance **exactly zero**, not only A/B.

An A context costs at most 20 forwards: 3 captures, 1 mask-matched neutral replay,
2 combined reconstructions and 2 full-suffix references on the mask-matched
source background, plus 3 sites x 2 poles x 2 routes into the open target. The
masked-source reference is diagnostic; the open target's full-suffix reference
remains the cross-context denominator. A new D layout costs 23: the same 8
matched captures/references, 12 interventions, 2 site nulls and one empty hook.
A natural bridge layout costs 15: those 8 matched references, four period-only
reads, two site nulls and one empty hook. Do not substitute suffix reconstruction
for combined reconstruction: a suffix-only transplant need not equal native logits.

Before outcome execution a separate clean implementation commit must pass:

- CPU tests of an **explicit score-index** adapter, with trailing-token changes
  unable to influence that score and equal length at every transplant.
- Checkpoint-specific mask-adapter tests in eager mode. Use all-open custom masks
  as exact no-op controls, verify causal/local masks are retained, and change
  protected cell values while checking that pre-release captures are exactly
  unchanged. Verify early/late direct access, post-cut mask removal, self-replay,
  reconstruction and hook cleanup. Include a random-weight CPU test for each
  architecture before any native GPU canary. No assumed 4-D-mask API is approved
  by this plan.
- A clean source seal covering this plan directory explicitly, every implementation
  file, fixed fixtures, tokenizer/chat-template identity, checkpoint fingerprints,
  dependency versions and the full generated layout digest. The existing
  `code_manifest()` omits provenance: retain it but add a plan manifest/hash.
- Offline native parity, the exact archived bridge and the above numerical mask
  canaries within the fixed allowance. An all-open mask discrepancy is a failed
  measurement condition, not permission to relax tolerances after inspection.

The present contribution implements the plan/layout/graph verification only.
It deliberately exposes no GPU-run command. **GPU-ready is false** until those
implementation checks and a separate execution seal exist. This is a technical
readiness gate, not a request for an extra approval before doing the current task.

## 6. Endpoints, denominators and decisions

Reuse `panel_metrics`, `context_scores`, `mapping_conflicts`, and their sign/tie
conventions from `../downstream_span_20261101/summarize.py`. The additive
`cell_reducer.py` delegates to these exact frozen functions, rejects duplicated
orders and mixed cells, and supplies a nonmutating `reversed` view for the frozen
conflict counter's hard-coded context name. Its compatibility tests use actual
archived likelihoods, without creating any new measured data. Subset one complete
factor cell and physical display before calling `panel_metrics`: each topic must
have exactly two poles x two `plus_label` orders. Do not pass duplicated display
rows to its four-row assertion. For each intervention, first average the A/B
mapping orders toward the plus option, then subtract the minus donor's margin
from the plus donor's margin to obtain `g_i` in nats. Report display-specific
results, followed by the equal-weight average over AB and BA for each topic.

Report the old cell-specific retained fraction `sum(g_i)/sum(full_i)`, numerator,
denominator, per-topic gaps, correct donor signs, native-sign agreement, exact
ties, source/recipient native-letter conflicts with their eligible counts, and
leave-one-topic-out ranges. Omit and knockout keep their distinct backgrounds.
Keep raw and delta routes separate. Preserve negative and greater-than-one values.

**Primary comparisons use raw gap differences and one common reference** per
model/panel. Let `b_i` be the mean over displays of the period S full-suffix gap
at `(u,v)=(0,0)`; `B=sum(b_i)`. For any planned paired contrast `c_i`, report
`C=sum(c_i)/B`, and `C_-i=sum(j!=i,c_j)/sum(j!=i,b_j)` for all six omissions.
This prevents punctuation- or distance-dependent denominators from manufacturing
an interaction. A nonpositive B or any nonpositive leave-one-out denominator
makes all reference-scaled decisions in that stratum **unresolved**. Raw nats
are still reported. Never substitute the old paper's numerical denominator.

Fix the smallest effect of interest at **delta=0.05** of this common reference.
This is a declared operational convention, not a power-calibrated universal
semantic threshold. For a directional contrast, support within this finite panel
requires C and all six C_-i to exceed delta; a stable opposite direction requires
all to be below -delta. Practical equivalence requires C and all six C_-i to lie
in [-delta,delta]. Everything else is unresolved. This is a robustness rule,
**not a confidence interval, p-value, or population null test**. With six exposed
topics per panel, no confirmatory population-significance claim is planned. Each
model/panel receives its own decision; a cross-checkpoint replicated statement
requires the same decision in both models of the same panel. Report every
predeclared decision, including disagreements; do not select the favorable stratum.

## 7. Operational hypotheses and falsifying patterns

The table defines the contrasts `c_i` used above. Unless specified otherwise,
hold a period recipient at `(0,0)`, average displays equally, use the second
separator and the primary delta route for context changes. Site-only readings
in the matched surface cells use raw as in the previous study. Each additional
diagnostic is reported but cannot replace the primary contrast.

| Hypothesis / primary comparison | Pattern supporting the operational hypothesis | Pattern against it / unresolved interpretation |
| --- | --- | --- |
| **Period donor advantage.** Difference between period-source and question-mark-source g at a fixed period recipient | Stable positive c by the declared rule | Practical equivalence rejects an advantage >=delta in these cells; stable negative favors the other token. Neither determines punctuation's intrinsic nature |
| **Recipient sensitivity.** Period-target minus question-mark-target g with the same period source/difference | Stable nonzero direction identifies a readout-context contribution (report either sign, with no post-hoc directional test) | Practical equivalence rejects a target sensitivity >=delta here; unresolved values do not show invariance |
| **Source-letter component.** `c_i=(g_same_mapping-g_mapping_flip)/2` | Stable positive odd component **and** stable negative flipped gap relative to the common reference support a sign-reversing mapping-sensitive signal. Report source-letter following on actual native conflicts | Equivalent odd component rejects a component >=delta; a positive odd component without a negative flipped gap is attenuation/mixed coding, not letter reversal. Equivalent even component alone does not establish semantic content |
| **Ordinal versus B-label dependence.** For period raw-only, define `L_AB=g_second-g_first` and `L_BA=g_second-g_first`; primary ordinal `c_i=(L_AB+L_BA)/2` | Stable positive ordinal component **and positive L in both displays** support a later-position advantage beyond the B label. Diagnostic B-label component is `(L_AB-L_BA)/2` | Opposite signs in the two displays with a stable B component favor label dependence. Equivalent ordinal component rejects a consistent advantage >=delta. Mixed components or label-dependent signs remain mixed |
| **Option access contributes.** At the late receiver, `c_i=(g_release_late-g_never)-(g_sham_late-g_sham_never)` | Stable positive access contrast with clean graph/numerical controls supports a contribution of the newly permitted option paths | Equivalent contrast rejects a contribution >=delta under this manipulation; a stable opposite effect indicates suppression. Nonzero never g is not a mask failure. Failed closure checks invalidate the comparison |
| **Earlier indirect paths contribute.** At the late receiver, `c_i=(g_release_early-g_release_late)-(g_sham_early-g_sham_late)` | Stable positive history contrast, with identical direct late-access permissions, supports a contribution from earlier non-cell paths | Equivalent contrast rejects an advantage >=delta from these paths. Negative favors delayed receipt. It does not adjudicate all notions of causal accumulation |
| **Readout attenuation with extra separation.** At each of the second separator and late receiver, `c_i=g_v0_only-g_v16_only`, separately at each u | Stable positive c at **both u values** supports attenuation in this reservoir family for that site, with exactly unchanged donor captures | Equivalent effects at both u reject attenuation >=delta here; a stable negative favors facilitation. Dependence on u is position/context interaction. Absolute scoring position remains an alternative |

For the two sign-dependent subcriteria in the mapping/ordinal rows apply the
same reference and leave-one-out rule (not unnormalized per-topic sign counts).
For recipient sensitivity predeclare positive, negative, equivalent, or unresolved
as four exhaustive outcomes. No significance decision or mechanistic uniqueness
is obtained by inspecting many sites. No threshold, best site, cut, topic,
punctuation, distance, or template is tuned after an outcome is seen.

## 8. Missingness, invalidation and stopping

- Token/position/ID mismatches are preflight failures for the affected complete
  paired comparison. Emit the missing reason and planned denominator; no manual
  shifting, replacement topic, alternative punctuation, width expansion or
  score-dependent fallback is allowed in v1. Primary finite-panel decisions need
  **all six complete topics**; incomplete data are diagnostic only.
- Nonfinite values, wrong checkpoint/template, dirty source, oversized contexts,
  nonzero null/reconstruction errors, bridge drift, mask leakage, retained hooks,
  shifted score indices or unequal forward lengths invalidate that job. Preserve
  all partial raw rows and the failed terminal record. Unrelated complete strata
  may be reported with their own explicit status; do not combine them into a
  successful whole-study claim.
- A positive but very small common reference is not discarded; report its scale
  and the resulting instability. A nonpositive reference makes ratios/decisions
  unresolved. An absent S effect, negative g, sign disagreement or failed
  hypothesis is an outcome, not a technical excuse to rerun.
- No free generation, fit, fine-tuning, hyperparameter search, pilot-based budget
  expansion or optional stopping. A budget, wall, or memory failure stops at the
  next safe forward boundary. No automatic retries or reuse of an old output
  directory. A later correction requires a versioned amendment recorded before
  any affected further scores; failures and already exposed outcomes stay visible.

## 9. Fixed maximum compute

Counts below are per topic/checkpoint; they do not count topics, labels, or routes
as independent statistical samples. Count every actual donor, neutral, sham,
masked, reference and recipient forward. Reused values need provenance; reuse may
save budget, never create a discretionary search allowance.

| Block | Arithmetic | Forwards / topic / model |
| --- | --- | ---: |
| Natural-template bridge | 2 layouts x 15 | 30 |
| S captures, references, region interventions and nulls | 8 layouts x 37 | 296 |
| S mapping/display/both swaps | 8 x 3 contexts x 4 sites x 2 poles x 2 routes | 384 |
| S opposite-punctuation sources | 8 x 1 site x 2 poles x 2 routes | 32 |
| A new source mask contexts | 4 period layouts x 6 contexts x 20 | 480 |
| D new layouts (base four already in S) | 12 layouts x 23 | 276 |
| **Outcome/bridge maximum** | | **1,498** |

Twelve topics x two checkpoints gives **35,952** planned outcome/bridge forwards.
Reserve at most 256 engineering/native-admission canary forwards per checkpoint:
the hard maximum is **18,232 per checkpoint, 36,464 total**. Each full sequence
is at most 384 tokens, giving a hard processed-token maximum of **14,002,176**.
The per-checkpoint processed-token cap is **7,001,088**.
Set a two-hour wall limit per checkpoint (four GPU-hours total maximum), count
failed/canary attempts in the same limits, and retain the existing memory floors.
Checkpoint loading/admission are inside the wall envelope. Unused allowance
cannot be spent on reruns or extra conditions. If this bound is insufficient,
record an incomplete study; a new budget requires a separate prospective version.
No wall-time prediction or statistical power guarantee follows from these caps.

## 10. Files, analysis compatibility and reproduction before a GPU run

Retain `request.json`, `source.json`, `checkpoint_identity.json`,
`host_identity.json`, `environment.before/after.json`, `native_parity.json`,
`span_settings.json`, `prompts.jsonl`, `interventions.jsonl`, `terminal.json` and
SHA-256 receipts. Reuse the nested `reference`, `regions`, and `contexts` score
objects and raw `scores: {A: logprob, B: logprob}`. Each row still has `group`,
`pole`, `plus_label`, and `recipient_ids_sha256`. Add a versioned `cell` object
with model, panel, block, display, target punctuation, u, v, `layout_sha256`, and
explicit score index; contexts add source punctuation, source mapping/display,
mask/release/blocked-key spans, route, source hashes and matched-neutral hashes.
An intentionally mismatched punctuation ID receives its own status tag.
Keep the old `status: "measured"` for a collected window and store the separate
`compatibility: "intentional_punctuation_id_cross"` field; otherwise the frozen
`context_scores` function would silently exclude it. Failed windows retain a
non-measured status and an explicit reason.

`prompts.jsonl` adds full compiled IDs, source-slot IDs, segment maps, real
phrase spans, protected eight-token cells, shadow span, all site windows, native
archive path/row hash, declared dummy and trailing compensation. Store native
archive offsets separately from synthetic segment coordinates. Duplicate the
open target references in each source-context row if needed for the existing
summarizer; never substitute a masked-source full gap. Emit all expected cells,
missing cells, eligibility counts, per-topic paired contrasts, common-reference
metrics, old cell fractions, conflict counts, failure status and compute use.

The old summarizer is reused as a cell-level reducer, not edited to silently
accept a different experiment. `cell_reducer.py` checks one already grouped cell;
a future whole-study analysis wrapper groups by the full
factor key and display, validates the complete planned grid first, then calls
its existing functions and evaluates the frozen contrasts above. Raw scores
must permit independent recomputation; synthetic preflight files contain no
fabricated likelihoods and must never be ingested as measured interventions.

```bash
# Standard library only; reads local frozen archives, never imports a model.
python provenance/separator_factorization_20261101/verify.py --check

# Optional complete layout manifest; --out refuses an existing file.
python provenance/separator_factorization_20261101/verify.py --out runs/separator-layouts.jsonl

# CPU engineering tests, including deliberate leakage/rank/geometry failures.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
```

`verify.py` checks the frozen baseline, native-token geometry, all paired grids,
the estimable design ranks, the unavoidable coordinate alias, mask graph closure,
planned denominators and the compute arithmetic. Its deterministic JSON receipt
and layout digest can be checked before implementing or loading any GPU model.
The runtime mask and score adapters remain separately required, explicitly
unvalidated implementation work. No new scientific result is asserted here.
