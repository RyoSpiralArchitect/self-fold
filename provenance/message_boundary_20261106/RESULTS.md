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
