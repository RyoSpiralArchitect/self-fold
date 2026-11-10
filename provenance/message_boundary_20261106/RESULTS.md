# What the symbolic residue of an intermediate state can and cannot carry

**Development result, executed on SpiralReality-furnace (RTX 5090, torch 2.13.0+cu132, transformers 4.57.1). Frozen source compiler `ab3dd6…` and frozen native row reader `427e13…` exactly as bound in `role_memory_20261104`; checkpoint `f3503907…`. Not a qualification run.**

**Conclusion: within this structured interface, verbalizing the intermediate table costs nothing and a learned one-bit message restores everything the words alone could not carry. Soft and hard (named-action) boundary forms are indistinguishable on every endpoint; the message channel alone decides success, and reward-trained writers again converge to the convention coordinate — off/off selects the switched row — in all seeds.**

> PR #32 review correction: the `mixture/`, `write/` and `write2/` runs used
> `mixture_boundary.twins_w`, whose world/history/erasure axes were interleaved.
> Their archived twin counts must not be used as interpreted scientific results.
> The repaired reducer reverses that layout in a new summary without changing
> the raw arrays. Their old `written` fields also recorded the unused controller
> writer for LM/supplied/none arms, not the carried message. These defects do
> not occur in the separate all-four scoring path. The initial `run/` uses a
> different twin producer. See [review, scope and replay requirements](PR32_REVIEW.md).

## The factorial result

Nine arms `(form × channel)` × three seeds {17, 41, 73}. Evaluation is exhaustive by construction: all 14 nonconstant masks, all 38,416 four-edit strings, 64 test sources. Denominators are whole populations, not selected strings.

| Arm | Native correct per depth | All four correct | …even prefix | …odd prefix | Twins intact | Twins erased | Opposed reversed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| soft + learned bit | 1.0, 1.0, 1.0, 1.0 | 100% | 100% | 100% | 100% | 0% | 100% |
| hard + learned bit | 1.0, 1.0, 1.0, 1.0 | 100% | 100% | 100% | 100% | 0% | 100% |
| hard + learned 2 bits | 1.0, 1.0, 1.0, 1.0 | 100% | 100% | 100% | 100% | 0% | 100% |
| soft + supervised bit | 1.0, 1.0, 1.0, 1.0 | 100% | 100% | 100% | 100% | 0% | 100% |
| hard + supervised bit | 1.0, 1.0, 1.0, 1.0 | 100% | 100% | 100% | 100% | 0% | 100% |
| soft + supplied bit | 1.0, 1.0, 1.0, 1.0 | 100% | 100% | 100% | 100% | 0% | 100% |
| hard + supplied bit | 1.0, 1.0, 1.0, 1.0 | 100% | 100% | 100% | 100% | 0% | 100% |
| soft + no message | 1.0, 0.5, 0.5, 0.5 | 12.5% | 25% | 0% | 0% | 0% | 50% |
| hard + no message | 1.0, 0.5, 0.5, 0.5 | 12.5% | 25% | 0% | 0% | 0% | 50% |

All four endpoints — per-depth native correctness, all-four trajectories, identical-table twin histories with reversed A/B margins, and opposed-donor reversals — are identical between `soft` and `hard` in every channel and every seed. `soft_none` and `hard_none` reproduce the role_memory no-bit control exactly: 12.5% all-four (see below for the exact stratification).

## What the writers learned

`written_matches_offoff_bit`: every learned writer emitted the convention coordinate — whether the off/off world selects the switched row — on 100% of the 3 × 38,416 × 64 recorded boundary writes, per seed. The write is a pure, deterministic function of the current instruction's off/off bit: correlation 1.000, identical rates conditioned on the previous role (1.000 under both values). It does not encode history; it forwards one coordinate of the just-consumed mask — "what I just did to row 0" — which is exactly the minimal self-description the next edit needs. For the two-bit arm, bit 1 was redundant (agreement 0.5 = unused noise in seed 17; 1.0 in seeds 41, 73). The reward channel again discovers the supervised coordinate without being shown it.

## Mechanism of the no-message failure, resolved exactly

Stratifying all 38,416 strings by the number of off/off role exchanges in the first three edits:

| Role exchanges | Strings | All-four correct |
| --- | ---: | ---: |
| 0 | 4,802 | 100% |
| 1 | 14,406 | 0% |
| 2 | 14,406 | 0% |
| 3 | 4,802 | 0% |

The 12.5% figure is exactly the never-flipped subset; the earlier "even parity" description is a selected-string artifact — two exchanges do not cancel. Depth-wise, the controller acts as "row 0 is always the default": depth-2 hit rate is 1.0 when the first mask's off/off bit is 0 and 0.0 when it is 1; at depth 3 the rate is 1.0 / 0.5 / 0.0 for zero / one / two accumulated exchanges. Errors are monotone in accumulated role exchanges, identical for `soft` and `hard`.

## An important honesty check: the "soft" rows were already almost symbolic

The source compiler's rows are near-categorical: median max weight 0.9996, minimum 0.9869 across the 128 test pairs. Every soft row's argmax equals its hard counterpart's action (128/128), every hard row names an in-menu action, and the frozen reader's |A−B| margins are statistically indistinguishable between forms (median 21.4 soft vs 21.6 hard). **So "verbalization loses nothing" is established here only in the regime where the state was already ~99% discrete.** The genuinely informative regime — verbalizing a state whose meaning lives in the mixture weights — remains untested by this panel. What this run does establish cleanly is the other half: the residual that a bare action table *cannot* carry is symbolic, one bit wide, learnable from outcome reward, and identical in content to the supplied convention.

## Reading

1. **The mixture carried nothing the task needed.** Soft rows' continuous weights vanish entirely under `hard` with zero cost at every depth. The only necessary residue beyond named actions is itself discrete — which is exactly what words transmit. This is consistent with the G-stage diagnostic (re-encoding preserved hard actions and dropped floats): here, the dropped floats were not load-bearing.
2. **The bit is the minimal self-description that survives any channel.** A table of named actions is blind to its own role history; one symbol restores it. Where that symbol comes from — learned reward, supervision, or supplied code — does not matter once present, and erasure collapses every arm to zero on twin histories, so the message is causally necessary rather than decorative.
3. **For the original T/L/B question:** this experiment predicts B ≈ L on this task family *provided* the role bit is allowed to cross as a symbol. The strict word-only route (`hard_none`) retains all-four correctness only on the never-flipped subset specified above; terminal parity alone is not the all-prefix criterion. Whether the model's own state-conditioned verbalization writes the needed bit remains a separate writer question.
4. **Limits.** Sixteen policies and familiar action menus; the `hard` boundary models verbalization as "named argmax action", an idealization justified by the diagnostic's finding that E(D) preserved hard actions — real generation could write less faithful text. Message bits are symbols at the interface, not natural language. The near-categorical source rows (above) mean the soft-versus-hard contrast tested a weak version of the question. No new wording panel, no qualification inputs, no claim beyond this structured interface.

## Next measurement suggested by this analysis — executed

The mixture sweep ran on furnace (`mixture/`): member rows blended β·own + (1−β)·partner at β ∈ {1.0, 0.8, 0.6}, plus a `traced` channel that writes the off/off role violation into row-0 mixture weights (ε = 0.25, invisible to argmax). All realized row vectors were scored exactly through the frozen reader. Verdict on the preregistered predictions:

- (a) confirmed: `hard_none` and `soft_none` stay at the never-flipped ceiling at every β.
- (b) **rejected in the negative direction on all-four correctness**: `soft_traced` and `hard_traced` stayed near the no-message control. The previously quoted twin separation (~0.29 at β=1.0) came from the axis defect noted above and is withdrawn pending corrected reduction. The mark is present in the weights, but the fitted pointer did not recover the required all-four behavior.
- (c) **rejected**: `soft_reward` never exceeded `hard_reward`; at β=0.8 both are 100%, at β=0.6 soft *falls below* hard (0.656 per depth vs 1.0) — because the reader itself fails on heavily mixed rows (the learned bit still wrote the convention perfectly, agreement 1.0). The only soft-vs-hard gap found is readout fidelity, not stored content.

Within these tested representations and fitted readers, no soft-state advantage was observed. This does not locate every possible verbalization loss upstream or rule out a different reader of the weight trace.

## W-RITE: reports of supplied ideal assignments (`write/`, `write2/`)

**Scope.** `prompt_for` renders four ideal world/action assignments from the source options, source polarity and requested mask. It does not read the controller's realized post-update tensor. The decoder also receives the source option-to-role mapping. Reports are looked up again by source and mask on later steps. Each channel trains its own controller. This is a supplied-table/report/decoder/controller experiment, not a fixed-G L/B contrast or autonomous `D(Z)`.

The natural and guided prompts ask what changed, but supply no previous table. That request is under-specified. A coordinate mention count is a lexical statistic, not a measure of spontaneous self-awareness or proof of which information the text encodes.

| channel | archived `write/` all-four | archived `write2/` all-four |
| --- | ---: | ---: |
| learned bit | 100% | 100% |
| supplied convention | 100% | 100% |
| directed report | 92.2% | 100% |
| guided report | not run | 81.0% |
| natural report | 53.4% | 53.8% |
| no message | 12.5% | 12.5% |

These are the archived summaries, not a new post-review replay. Each percentage is complete correctness over all four edits, four worlds and both answer orders for 64 sources × 38,416 strings. The 14 masks pool eight fitting masks and six other masks. The three seeds vary controller fitting, not independent frozen-writer report samples. Per-report bit accuracy and all-four trajectory correctness have different denominators and must not be combined into a "53–77% capacity" range.

The second round changed the lexical decoder and added a guided prompt. Its directed arm's reported 100% shows that this directed report channel can support the tested controller on this bank. The earlier bare-verb misses were not evidence of writer silence. Conversely, the lower natural/guided scores do not establish an intrinsic language-capacity ceiling, a plateau, or that no other decoder could recover the bit. The current heuristic ignores row binding and can prefer a different member even in text that explicitly specifies off/off.

Unmatched reports were assigned bit zero. That helps true-zero cases and hurts true-one cases, so it is not a generally conservative lower bound. The decoder's substring and character-strip behavior is corrected for future runs as `distinctive_tokens_v3`; the archived runs are not relabeled as v3.

The archived `offoff_rate` averaged all 16 allocated mask columns although only 14 were generated. On the recorded Boolean mention flags, `write2` therefore has 164/2,912 natural reports (5.63%) and 420/2,912 guided reports (14.42%), not 4.9% and 12.6%. This arithmetic correction does not reannotate raw text. Future runs record counts over sampled masks and actual EOS/censoring receipts, save the guided bit matrix, and require a complete prompt/checkpoint/source-bound cache.

**Evidence boundary.** Historical raw reports and summaries remain intact. `trajectories.npz` is not published with these runs, and the old cache had no prompt/checkpoint binding. Full numerical replay of the historical reports has not been completed in this review. In particular, their twin counts and nonlearned `written_matches_offoff_bit` values must be regenerated or reinterpreted with the repaired reducer before citation. [Detailed review and commands](PR32_REVIEW.md).

## S-WRITE (`write3/`): supplied-table prompt study (registered realized-state test NOT executed)

Executed on furnace after PR #33's design, but **deviating from the registered
protocol** (`docs/STATE_WRITER_PLAN.md`) in ways the PR #34 review correctly
flagged — the P1–P3 registrations do not apply:

- Reports are still generated from `(source, pole, mask)` ideal assignments;
  `The table now reads:` repeats those same assignments — no realized W ever
  reaches the writer.
- The prior arm used `assignment_words(..., 0)` = `[m0,m0,m0,m0]`, but the
  actual initial member layout is `[m0,m0,m0,m1]`, and later priors depend on
  earlier edits — so it is not a referent-restoration test.
- `decode_state` still receives the source pole; controllers are refit per
  channel.
- The shipped decoder was a ±120-char window, not a row-binding parser: it
  misread correct "changed from X to Y" reports (review-verified synthetic
  failures). Fixed post-review as `rowbound_v1` with regression tests.

Recorded all-four outcomes **under the shipped window decoder** (descriptive
only, not plan verdicts):

| channel | all-four (3 seeds) | twins intact (repaired axes) |
| --- | ---: | ---: |
| supplied convention / learned bit | 100% | 1.000 |
| state_directed | 100% | 1.000 |
| state_guided | 87.3% | 0.866 |
| state_natural | 33.8% | 0.438 |
| state_prior_natural | 12.5% | 0.000 |
| no message | 12.5% | 0.000 |

### Additive parser audit (`rowbound_v1` re-decode of all 11,648 saved reports)

Per-report bit correctness vs the true mask bit0 — decoder cross on the same
corpus, no regeneration:

| style | window decoder correct/wrong/missing | rowbound_v1 correct/wrong/missing |
| --- | --- | --- |
| state_directed | 2912 / 0 / 0 | 2912 / 0 / 0 |
| state_guided | 2508 / 308 / 96 | **2816 / 0 / 96** |
| state_natural | 1796 / 960 / 156 | 1796 / 960 / 156 |
| state_prior_natural | 1024 / 906 / 982 | 1054 / 876 / 982 |
| write2 directed / guided / natural | (unchanged — unanchored styles fall back to the global decoder) | 2912·0·0 / 2148·132·632 / 2120·572·220 |

Readings the audit supports:

- The window decoder invented every state_guided wrong bit; under
  `rowbound_v1` its per-report bit accuracy is 96.7% (the recorded 87.3%
  all-four was fitted/evaluated on the *old* bits — recomputing it needs a
  rerun, and is left unclaimed here).
- state_natural's 61.7% per-report accuracy is decoder-invariant — genuine
  unanchored-report ambiguity.
- state_prior_natural stays bad under any decoder (~36% correct, 30% wrong,
  34% missing) — real writer failure, consistent with its floor-level
  all-four.
- The W-RITE-vs-S-WRITE all-four contrast remains confounded (prompt +
  decoder + per-channel refit all changed at once); these tables separate
  only the decoder factor.

What remains open: a genuine realized-state run (actual post-update W in the
prompt, actual preceding table, pole-free decoder, shared downstream pointer)
needs new sealed evidence; the planned P-tests are hereby marked **not
executed under their registered conditions**.

## S-WRITE sealed run (`write4/`): true priors, slot decode, shared pointer

Sealed execution addressing `write3` deviations, **with remaining limits the
re-review correctly kept open**:

- `state_prior_true` reports key on `(source, prior, current)` with the
  actual prior table — initial layout `[m0,m0,m0,m1]` (mask 8) at step 0, the
  *requested previous mask* afterwards. **Not the realized table**: after a
  pointer error the requested mask does not reconstruct what was actually
  carried. The registered realized-state test of the controller's own tensor
  remains **not executed**; this is a rendered-history prompt study.
- `decode_state_slot` binds the off/off value to an *option slot*; the
  slot→role-bit mapping via source pole is **declared carried side
  information** (same role metadata the supplied convention encodes), not
  information recovered from the report alone. `rowbound_v3` adds multi-row
  clauses, temporal precedence (`now` > plain > `previously`),
  negation-complement votes and pronoun-anchor reset — all re-review
  counterexamples covered by regression tests.
- The **shared-pointer** block is a *convention-trained fixed-reader*
  comparison: the supplied-fitted controller evaluates every LM matrix.
  Directed, prior_true and supplied actors converge to identical hashes, so
  it is not an independent refitting result.
- `state_prior_true` generation is eval-time-only: 14,336 reports over 64
  test sources × 16 priors × 14 masks; the 144 fit rows are convention-filled
  and are now excluded from report statistics.

| channel | all-four (3 seeds) | twins intact/erased | shared-pointer all-four |
| --- | ---: | --- | ---: |
| supplied / reward | 100% | 1.000 / 0.000 | — |
| state_directed | 100% | 1.000 / 0.000 | 100% |
| **state_prior_true** | **14.0%** | 0.179 / 0.000 | 14.0% |
| no message | 12.5% | 0.000 / 0.000 | — |

**Corrected report statistics** (generated reports only — the earlier 13.8%
missing / 2.8% mention rates wrongly averaged in 144×16×14 synthetic fit
cells): state_prior_true **44.9% missing, 9.0% off/off mention** over 14,336
reports; per-report bit correctness 28.4% / wrong 26.7% (`rowbound_v3`
re-decode, additive audit). state_directed: 0% missing, 100% correct.

Readings (descriptive):

- Directed text over the rendered post-update table reaches the ceiling, in
  both per-channel and shared-reader evaluations — at this interface and
  decoder.
- With the true prior shown, spontaneous "what changed" reports carry the
  bit barely above the floor (14% vs 12.5%) — the predicted referent benefit
  is again not observed. Whether this is writer failure or residual parser
  failure is **not decided**: parser agreement is not semantic annotation,
  and a unique-form audit of the saved corpus remains open.
- Shared = per-channel equality is expected here (converged identical
  actors) and is labeled as such; it does not isolate channel quality from
  reader fitting.

Limits kept: rendered (not controller-emitted) tables; requested-mask priors,
not realized-error-corrected priors; decoded bits, not raw text, cross the
channel; the registered pole-free/actual-tensor test stays unexecuted.

## S-WRITE sealed rerun (`write5/`): realized-pattern prior lookup

Addressing the "requested-mask prior is not the realized table" point directly:
`state_prior_true` reports are now indexed by the **realized carried weight
pattern** — `argmax` member per row, packed as a mask — computed from the
actual `weights` arriving at each step, including after pointer errors.
A deliberate-error fixture (`tests/test_realized_prior.py`) verifies a wrong
carried bit *does* diverge the realized prior from the requested mask, so the
lookup path is exercised and distinct.

| channel | all-four (3 seeds) | twins intact/erased | shared-pointer all-four |
| --- | ---: | --- | ---: |
| supplied / reward | 100% | 1.000 / 0.000 | — |
| state_directed | 100% | 1.000 / 0.000 | 100% |
| state_prior_true | 14.0% | 0.179 / 0.000 | 14.0% |
| no message | 12.5% | 0.000 / 0.000 | — |

Identical endpoints to `write4` — but per re-review, equal scores do **not**
establish zero divergence (post-failure divergence cannot change an already
failed all-four). `rollout_w` now emits `prior_mismatch` — a per-edit count of
realized ≠ requested prior on matched routes — so future runs measure the
state-level divergence directly. Fixtures establish the mechanism is real
(`tests/test_realized_prior.py`: a message-steered wrong pick flips the
carried argmax whenever rows disagree); whether it fires at this channel
quality is recorded, not assumed.

## Reproduce

```bash
python3 scripts/fit_message_boundary.py --checkpoint <gemma-2-2b-it snapshot> \
  --out provenance/message_boundary_20261106/run/produce
python3 scripts/role_memory_native.py --checkpoint <snapshot> \
  --packet provenance/message_boundary_20261106/run/produce/packet_soft \
  --out provenance/message_boundary_20261106/run/decide_soft
python3 scripts/role_memory_native.py --checkpoint <snapshot> \
  --packet provenance/message_boundary_20261106/run/produce/packet_hard \
  --out provenance/message_boundary_20261106/run/decide_hard
python3 provenance/message_boundary_20261106/summarize.py \
  provenance/message_boundary_20261106/run/produce
```

Artifacts: [plan](PLAN.md) · [raw run](run/produce) · [scored packets](run) · [summary](run/summary.json) · [controller](../src/selffold/message_boundary.py) · [fit script](../scripts/fit_message_boundary.py) · [summarizer](summarize.py)

## RoundTrip sealed run (`write6/`): realized (prior, mask, out) reports

The report channel is now keyed by the **realized transition triple**
`(prior, mask, out)` — the writer sees the actual post-update table,
including pointer errors, and the decoded bit feeds the next message
verbatim (`rt_report_bits_v1`). Coverage is the exact `feasible_outs`
pick bound: every (prior, mask, out) any pick can produce is generated,
so a lookup outside the cache is a protocol error, never a silent zero.
Generation dedupes over unique (group, pole) pairs (12,992 prompts/style).
First attempt stopped at 28,455/40,320 forwards-budget-exhausted and is
preserved off-repo; the closed run used budget 1.5M.

| channel | all-four (3 seeds) | twins intact/erased | shared-pointer all-four |
| --- | ---: | --- | ---: |
| supplied / reward | 100% | 1.000 / 0.000 | — |
| rt_directed | 100% | 1.000 / 0.000 | 100% |
| rt_natural | 7.3% | 0.071 / 0.000 | 7.3% |
| state_prior_true | 14.0% | 0.179 / 0.000 | 14.0% |
| no message | 12.5% | 0.000 / 0.000 | — |

### Decoder-consensus audit (`audit_roundtrip.py`: 3 decoders + unique-form view)

Per-report decoder-consensus categories on the write6 corpus. These are
NOT writer/parser attribution: the three decoders share lexical scoring
(two wrap the same `decode_state_slot`), so shared blind spots are
possible and all-decoder missingness does not prove omission.

- **rt_directed** (12,992): consensus-correct 89.0%, consensus-wrong
  11.0%, unresolved 0%. Split by realized divergence: when out == mask
  the bit is 100% consensus-correct; when out != mask the report names
  the *instruction's* bit (mask&1) in 78% — it answers the question
  asked. The carried bit thus behaves like the supplied convention on
  instruction faithfulness, consistent with the ceiling all-four. Note
  the directed prompt asks about the instruction, not the realized out:
  100% end-to-end is not 100% faithful reporting of erroneous outputs.
- **rt_natural** (12,992): consensus-correct 33.7%, consensus-wrong
  11.7%, unresolved-all-missing 53.4%, disagreement 1.2% — over 333
  unique unresolved forms. Correctness concentrates on uniform outs
  (52% vs 14.6% on mixed outs); off/off is named in 8.7% of reports.
- **state_prior_true** (14,336): unresolved 44.9%, consensus-wrong
  23.8%, consensus-correct 26.2%, disagreement 5.1%.

Readings (descriptive, this interface/decoder only):

- A **directed** question over the realized triple reaches the ceiling —
  a working structured report→decode→next-step loop under its specified
  question and decoder. An actual-output question is a distinct arm.
- **Spontaneous** natural reports remain decoder-consensus-poor on the
  realized state (~7% ≈ below floor); whether the gap is unwritten
  content or shared parser blind spots needs an independently annotated,
  stratified sample — not asserted here.
- The rt-vs-state missing-report fallback was asymmetric at write6
  (slot^pole on missing vs bit 0); it is declared and unified to 0 in
  `rowbound_v4` for future runs, so the 14.0% vs 7.3% comparison is a
  report+decoder+fallback system comparison, not a pure writer effect.

### Post-review metadata correction (`lm_matrix_rt.json`)

The committed `lm_matrix.json` selected the *output* axis of the 4D rt
matrices with `[..., masks]` — the feasible collapsed outputs (0, 15)
were excluded, and `report_counts` counted broadcast per-source cells
rather than unique generated records. The saver now selects the
instruction axis (axis 2) for 4D and reports `broadcast_cells` and
`report_counts` (unique records) separately. Recomputed from the sealed
JSONL corpus into `write6/produce/lm_matrix_rt.json` without touching
the archived file: 12,992 unique reports per rt style; per-instruction
bit-correctness including all feasible outs is in that file.

## Raw-text channel sealed run (`write7/`): undecoded reports cross the boundary

The carried payload is the report's frozen hashbag feature vector
(`hashbag_v1`, 32 dims), not a decoded bit; the recipient learns a
`feat_addr` projection by reward. Answers the writer/parser separation:
if the learned reader recovers what the decoder missed, the information
was in the text. Generation policy `kv_capped_argmax` (declared; GPU
parity vs full-re-forward was 25/30 — a distinct deterministic path).

| channel | all-four (3 seeds) | twins intact |
| --- | ---: | ---: |
| supplied / reward | 100% | 1.000 |
| rt_directed | 100% | 1.000 |
| state_prior_true | 13.8% | 0.170 |
| rt_natural | 13.3% | 0.161 |
| lmtext_rt_directed | 0.0% | 0.281 |
| lmtext_rt_natural | 0.0% | 0.335 |
| no message | 12.5% | 0.000 |

Readings (descriptive, this interface only):

- The learned text reader transmitted real but weak signal (twins
  0.28-0.33 vs none's 0) while all-four collapsed to 0. Training curves
  show near-flat second-step reward (0.51-0.56) — `feat_addr` barely
  learned under 512 updates on hash features.
- This does NOT resolve writer-vs-parser: outcome is consistent with the
  information being absent AND with an under-powered feature/learner.
  Decisive next steps: richer frozen features or longer training, an
  ordinal/pattern feature ablation, and the independent-form annotation.
- Compute: 631k forwards but 7.1M processed tokens (~15x fewer than
  write6's re-forward path) — the sealed KV path.
