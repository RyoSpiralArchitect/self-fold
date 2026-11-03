# Margin continuation improves fitting accuracy and leaves the strict conditional gate closed

The sealed continuation increased literal fitting correctness from 3,687/3,840 to **3,751/3,840** (97.7%). Every stage exceeded the 95% native fitting threshold. Source and exhaustive-case strict correctness remained **23/48** and **28/48**, below the fixed 90% requirement. Reintegrated rules passed with 47/48 strict units. The combined literal gate failed, so the frozen-E and joint-E branches stayed closed. The complete newly authored fresh panel scored **923/1,280** native judgments (72.1%). [Prospective plan](PLAN.md) · [Complete reduction](summary.json)

## Exact continuation of the closed endpoint

The parent study ended at the fixed adapter SHA-256 `4fb28b1c00ccebf89387f3dfb1793bef3c9ea8bedf782cb0d359e23943cd7d5c` with its source and case gates failed. Those records and that outcome remain in the [original curriculum study](../conditional_curriculum_20261103/RESULTS.md). This follow-up loaded that exact artifact, reset AdamW, and ran 384 additional updates cycling source, cases and rule, 128 updates per stage. The fit pool retained every previous source and question byte-for-byte. Before continuation, all 3,840 native fitting observations matched the parent's final A/B log probabilities, greedy tokens and labels, source hashes and question hashes.

The full-vocabulary cross-entropy term retained its prior all-on-world weight of three and other-world weight of one. The added A/B hinge used a fixed target margin of two with equal weight for all worlds. Learning rate was 0.0003, gradient clip one and weight decay zero. Each step included both policies, all four worlds and both label mappings for four fitting sources. The predetermined continuation therefore changed the objective, optimizer state and training budget together. The measured outcome belongs to this complete recipe. No fresh score selected the final checkpoint.

## Scalar and strict outcomes

A strict unit fixes topic, flag pair, wording family and stage. It requires both opposed policies, all four worlds, every genuine source order and both A/B mappings to answer correctly. Rules have one source order. Every topic and unit remains in the denominators.

| Literal panel | Initial native | Final native | Final strict units | Final forced A/B | Native A/B coverage | A/B ties |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fit source | 1,482/1,536 | 1,488/1,536 | 23/48 | 1,487/1,536 | 1,536/1,536 | 1 |
| Fit exhaustive cases | 1,443/1,536 | 1,496/1,536 | 28/48 | 1,495/1,536 | 1,536/1,536 | 2 |
| Fit reintegrated rule | 762/768 | 767/768 | 47/48 | 767/768 | 768/768 | 0 |
| Fresh source | — | 309/512 | 0/16 | 307/512 | 512/512 | 4 |
| Fresh exhaustive cases | — | 448/512 | 1/16 | 459/512 | 496/512 | 1 |
| Fresh reintegrated rule | — | 166/256 | 0/16 | 167/256 | 255/256 | 0 |

Native correctness uses the complete vocabulary's greedy next token. Forced A/B correctness requires a strictly positive correct-label minus foil margin, so ties are retained as incorrect under that measure. All 3,840 final fitting outputs were A/B. On the fresh panel, native A/B coverage was **1,263/1,280**, leaving 17 outputs outside the requested label pair. The fresh forced count was 933/1,280 with five ties. [All native observations](run/decisions.jsonl)

On the same fitting decisions, the new endpoint recovered 144 answers and lost 80, for a net increase of 64. It preserved 3,607 correct answers and missed nine that the parent also missed. Source strict units improved from 17/48 to 23/48, case units from 13/48 to 28/48, and rule units from 44/48 to 47/48. The fixed 90% strict threshold requires at least 44/48 in each stage. Only rule met that condition. Aggregate fitting accuracy therefore advanced while the required conjunction remained false. [Recorded gate](run/warm_gate.json)

The fresh bank introduced two wholly new flag-name pairs, a new syntactic family for every source stage, and new question operations. It has the same eight canonical action topics, truth ledger and number of observations as the first curriculum. Its wording and names differ from that earlier fresh bank. The earlier 1,117/1,280 and current 923/1,280 fresh counts describe different panels and cannot estimate a transfer change caused by the hinge. Both newly exposed panels can inform later development. Independent confirmation requires a separate seal and fresh material.

## Post-hoc label-order diagnostic

Analysis of the saved native logits found that **85 of the 89** final fitting errors had gold A and predicted B. A separate exploratory readout subtracts the A-minus-B margins from the two displayed label orders, then selects a canonical action from that difference. It consumes two saved native forwards for one external forced decision, retaining every observation including non-A/B native answers and ties. Its decision denominators are therefore half the native denominators. [Reproducible diagnostic](diagnose.py) · [Complete diagnostic output](diagnostic.json)

| External two-forward readout | Fit forced actions | Fit strict units | Fresh forced actions | Fresh strict units |
| --- | ---: | ---: | ---: | ---: |
| Source | 767/768 | 47/48 | 157/256 | 0/16 |
| Exhaustive cases | 757/768 | 38/48 | 241/256 | 6/16 |
| Reintegrated rule | 384/384 | 48/48 | 92/128 | 0/16 |

Pairing the label orders improves the fitting action readout, while fresh source and rule use remains weak. This post-hoc calculation does not change the native admission gate, establish a hinge-specific causal effect, or isolate the effects of fresh flag names, source syntax and question operations. No further model forwards or training were used for it.

## Boundaries and durable receipts

The first failed study and the fixed continuation endpoint are both preserved. No slot adapter, source adapter, decoder or G was trained in this continuation. The literal fitting failure closed both conditional slot arms under the prospective rule. The remaining stage-specific issue is complete conditional use across every world, policy, source order and displayed answer mapping. The later G composition and shared-boundary language intervention retain their separate prerequisites.

The run completed all 384 additional literal updates, with finite connected task gradients throughout. It used **5,315 forwards**, **3,090,054 processed tokens** and **184.45 seconds**, within the registered bounds. The frozen backbone hashes agree before and after. Saving and reloading the endpoint reproduced the checked native logits exactly. [Training trace](run/training.jsonl) · [Terminal receipt](run/terminal.json) · [Serialization receipt](run/fits.jsonl)

The clean execution source was `d5d152f4dfae21f0fccf60fe864a5e0e90cf84f7`. The returned run archive SHA-256 was `ec48aa2d4cee55d251a6eb4c617ba79cd14be1d87044bf0ac982e94e5a22561c`. The saved task adapter is [warm_task.npz](run/warm_task.npz), SHA-256 `91b7e96a22c43d6c9e9b626b8d4c3131a2d9a563bc7a5191f9a9517062727845`. Its parent binding and initial parameter hash are recorded in [parent_initialization.json](run/parent_initialization.json). These Gemma-derived factors carry the accompanying [artifact conditions](ARTIFACT_LICENSE.md), terms and modification notice. Original code and synthetic material retain Apache-2.0.

To rerun from the sealed source, provide the pinned local Gemma checkpoint and a new output directory.

~~~bash
HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src python scripts/fit_conditional_margin.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/conditional-margin
python3 -S provenance/conditional_margin_20261103/summarize.py runs/conditional-margin
~~~

Recompute the published reduction from the repository root with `python3 -S provenance/conditional_margin_20261103/summarize.py`. Verify every study file by running `shasum -a 256 -c SHA256SUMS` in this directory.
