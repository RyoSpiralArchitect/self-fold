# A trainable lexical-slot interface and two bounded closures

A rank-16 transformation of the lexical control's eight rows made a frozen Gemma recipient choose the correct native option on 22/24 decisions. It learned on four topics (16/16) and reached 6/8 on two topics excluded from this fit. Swapping to the opposing phrase's rows reduced correctness to 2/24. The pooled engineering screen passed. The two-topic transfer screen did not. This is a measured connection from native lexical rows to a downstream native answer. The ordinary E-description rows still lack an admitted recipient. [Prospective plan](PLAN.md) · [Every metric](summary.json)

## Lexical rows, gain and learned transformation

Each of six original Table 2 topics supplied two option phrases. Each phrase was captured at the native eight-slot positions, with the question replaced by neutral trailing tokens and the forward length matched to the recipient. The twelve phrases yielded 24 decisions across both option orders. Neutral replay reproduced every vocabulary logit, and every opaque lexical capture equaled its natural lexical counterpart. Native-word references, transformed rows and null controls used the same questions and native output head.

| Input | Native correct / 24 | Forced A/B correct / 24 | Both poles correct under both option orders / 6 |
| --- | ---: | ---: | ---: |
| Neutral rows | 8 | 12 | 0 |
| Lexical rows, gain 1 | 8 | 13 | 0 |
| Gain 2 | 11 | 14 | 0 |
| Gain 4 | 14 | 15 | 1 |
| Gain 8 | 12 | 15 | 1 |
| Gain 16 | 14 | 17 | 1 |
| Random directions at gain 4 | 8 | 13 | 0 |
| Learned lexical transformation | **22** | **22** | **5** |
| Same transformation, opposed rows | 2 | 2 | 0 |
| Same transformation, zero rows | 10 | 12 | 0 |
| Natural lexical words in the native forward | 23 | 23 | 5 |

All seven gains, including 0.5, and the reversed-row control are in the full summary. No gain crossed the declared screen. Gain was never selected to initialize or tune the learned transformation. The bridge was the existing shared row-wise residual z + (z A) B, rank 16, with 73,728 parameters, zero initial B, and 64 updates. Only those parameters received gradients. The recipient and backbone stayed fixed. Its loss used the native A/B logits for archive, route, notes and temperature only.

The four training topics supplied 16 decisions and four strict pairs. The bridge solved all of them. Shelf and directions supplied eight decisions and two strict pairs, with 6/8 and 1/2 correct. The two remaining errors both selected recognizable landmarks for the street-name donor. The transfer result missed the registered 80% native and 75% strict-pair requirements. These topics were already exposed in earlier experiments. This cycle supplies one seed and no independent qualification. The lexical rows explicitly contain an option phrase supplied by the fixture. They have a different provenance and coordinate system from E(s0).

## D-only copying did not recover source dependence

All 24 original descriptions were training examples. We loaded their archived E states and required exact re-encoding hashes for all 24. A single D-only q/v adapter covered layers 13 through 25 and trained for 64 updates to copy the source and emit the native end-of-turn token. E, all cached rows, the backbone and the native head stayed fixed.

| D endpoint | Source-copy NLL | Exact originals / 24 | Unique free outputs | EOS / 24 | Reconstructed-text Table 2 donors correct / 24 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | 5.7373 | 0 | 1 | 24 | 12 |
| Final | 0.3140 | 1 | 1 | 24 | 12 |

Every final output was the same training-bank sentence, “I record a meeting in the speakers' exact words.” Matched-state NLL averaged 0.313990, versus 0.315978 for the opposed state and 0.331762 for zero rows. Matching the state beat both controls on 10/24 references. None of the twelve opposed-description pairs had both original strings reconstructed. Free generations ended at EOS without censoring. The loss decrease therefore did not recover the distinct source texts. The original literal descriptions scored 24/24 donor-averaged Table 2 judgments, and the unadapted E rows 12/24. The reconstruction screen failed and this fit ended at its cap. All strings, matched/opposed/zero NLLs and downstream scores remain available. [D records](run/decoder/)

## Mixed-order policy training

The second historical training tag now presented the reverse case list. The first tag kept forward order. Every other optimizer setting, initial adapter, group sequence and question matched the preceding case-list fit. We evaluated its frozen endpoint and the fresh mixed endpoint on the two trained orders and two new permutations. Both historical tags duplicated each evaluation text and were collapsed only after checking agreement.

| Training input | Reader | Forward / 64 | Reverse / 64 | New order 1 / 64 | New order 2 / 64 | Strict opposed pairs correct in all four orders / 32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Forward only | Literal | 58 | 47 | 57 | 54 | 8 |
| Mixed | Literal | 57 | 56 | 56 | 57 | 14 |
| Forward only | E rows | 35 | 31 | 33 | 34 | 0 |
| Mixed | E rows | 30 | 34 | 33 | 34 | 0 |

These are the same four training groups. In the two previously used dev groups, the literal all-four-order joint count changed from 8/16 to 9/16. In two additional dev groups that were absent from the earlier small-set fit it stayed at 8/16. Those groups were exposed by earlier broader recipient experiments. Both latent dev panels remained at zero strict all-order pairs. Neither reader passed the preregistered training or permutation screen. Each fit stopped at 64 updates. [Per-order decisions](run/order/decisions.jsonl)

## Scope and next anchor

All four fits completed their fixed 64 updates. All 256 recorded gradient norms were finite and positive. Backbone parameter hashes matched before and after each arm, and the four saved artifacts passed exact checked save/load replay. The learned lexical interface passed its pooled development screen, while its two-topic transfer screen, D reconstruction screen and mixed-order recipient screens stayed closed. This is one bounded positive interface result alongside the retained failures. It establishes no learned G or two-stage semantic recurrence.

The next question is whether an E(s0) state can be mapped to the lexical interface that now produces a source-dependent native answer. That comparison should freeze this recipient, retain opposed/zero controls and use fresh source and topic combinations. Another source-copy fit would need a criterion that demands state dependence as well as token prediction. The current successful lexical endpoint and the three failed configurations stay fixed as controls.

## Reproduction

Source commit ad2d1a2e9016fa5fb368549ef1ee20320f1ffb2e was pushed before the model ran. Each arm used the same pinned Gemma-2-2B-IT and native hook parity. The raw record archive was returned with SHA-256 f087a5d30ce3bb5db2f2054ea188124544e072ef96a4d5c2fc89981fe44c8b6f. Its three arm directories and learned arrays are included here under their applicable conditions. [Artifact licenses](run/ADAPTER_LICENSE.md)

~~~bash
PYTHONPATH=src python scripts/probe_slot_readout.py --arm decoder --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/readout/decoder
PYTHONPATH=src python scripts/probe_slot_readout.py --arm lexical --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/readout/lexical
PYTHONPATH=src python scripts/probe_slot_readout.py --arm order --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/readout/order
python3 provenance/slot_readout_20261103/summarize.py runs/readout
~~~

From the published study directory, verify the archived results with the following commands.

~~~bash
shasum -a 256 -c SHA256SUMS
python3 summarize.py | diff - summary.json
~~~
