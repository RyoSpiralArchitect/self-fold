# What the symbolic residue of an intermediate state can and cannot carry

**Development result, executed on SpiralReality-furnace (RTX 5090, torch 2.13.0+cu132, transformers 4.57.1). Frozen source compiler `ab3dd6…` and frozen native row reader `427e13…` exactly as bound in `role_memory_20261104`; checkpoint `f3503907…`. Not a qualification run.**

**Conclusion: within this structured interface, verbalizing the intermediate table costs nothing and a learned one-bit message restores everything the words alone could not carry. Soft and hard (named-action) boundary forms are indistinguishable on every endpoint; the message channel alone decides success, and reward-trained writers again converge to the convention coordinate — off/off selects the switched row — in all seeds.**

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
3. **For the original T/L/B question:** this experiment predicts B ≈ L on this task family *provided* the role bit is allowed to cross as a symbol. The strict word-only route (`hard_none`) fails on exactly the odd-parity histories — the same failure the paper's table_only control showed. The open question is therefore not whether verbalization preserves the state, but whether the model's *own* verbalization writes the bit it needs — i.e. whether D emits the self-referential symbol. That is a writer question, addressable next on the native route once the S/N/C diagnostics separate reader from carrier.
4. **Limits.** Sixteen policies and familiar action menus; the `hard` boundary models verbalization as "named argmax action", an idealization justified by the diagnostic's finding that E(D) preserved hard actions — real generation could write less faithful text. Message bits are symbols at the interface, not natural language. The near-categorical source rows (above) mean the soft-versus-hard contrast tested a weak version of the question. No new wording panel, no qualification inputs, no claim beyond this structured interface.

## Next measurement suggested by this analysis — executed

The mixture sweep ran on furnace (`mixture/`): member rows blended β·own + (1−β)·partner at β ∈ {1.0, 0.8, 0.6}, plus a `traced` channel that writes the off/off role violation into row-0 mixture weights (ε = 0.25, invisible to argmax). All realized row vectors were scored exactly through the frozen reader. Verdict on the preregistered predictions:

- (a) confirmed: `hard_none` and `soft_none` stay at the never-flipped ceiling at every β.
- (b) **rejected in the negative direction**: the `traced` weight channel recovered nothing — `soft_traced` ≈ `hard_traced` ≈ the no-message control (12.5% all-four, twin separation ~0.29 both ways at β=1.0, worse at β≤0.8). The role *was present* in the weights (the mark is there), but this supplied pointer interface could not read it. Weight-carried memory is unusable at this interface even when it exists.
- (c) **rejected**: `soft_reward` never exceeded `hard_reward`; at β=0.8 both are 100%, at β=0.6 soft *falls below* hard (0.656 per depth vs 1.0) — because the reader itself fails on heavily mixed rows (the learned bit still wrote the convention perfectly, agreement 1.0). The only soft-vs-hard gap found is readout fidelity, not stored content.

Updated conclusion: there is no regime at this interface where the soft state beats its verbalization. Either the needed residue is symbolic (carried by a learned bit), or the residual written into weights is unreadable by the fitted readers. Self-verbalization loss, if it exists, lives upstream in what generates the state — not in this memory channel.

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
