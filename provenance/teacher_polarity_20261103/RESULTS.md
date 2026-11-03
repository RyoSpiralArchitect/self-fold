# Reading a supplied lexical policy and recovering a preference from its source

A calibrated renderer carried the correct lexical policy to 113/128 native answers on four new question families. A separate reader recovered the preferred action from 188/256 question-free source states on two new sentence frames with familiar actions. Both development screens missed their prospective thresholds. The two scores measure different operations on different inputs. [Sealed design](PLAN.md) · [Registered reduction](summary.json)

## Correct teacher supplied first

The sixteen teachers cover both preferences for each of eight topics. The shared rank-four renderer started from the released Section 6.13 four-style endpoint. It learned from all four previously exposed question families for 64 updates. The base model, source encoder and downstream lexical transformation were fixed. Every new question was scored in both displayed option orders under each teacher pole. There are 128 distinct decisions across eight topics. Each strict topic must answer all sixteen of its decisions correctly.

| Teacher input | Fitting decisions | Fresh decisions | Fresh strict topics | Fresh opposed margin flips |
| --- | ---: | ---: | ---: | ---: |
| Canonical lexical teacher | 112/128 | 102/128 | 2/8 | 40/64 |
| Archived Section 6.13 row renderer | 118/128 | 101/128 | 1/8 | 40/64 |
| Calibrated renderer | 128/128 | 113/128 | 1/8 | 49/64 |
| Common mean through the calibrated renderer | 64/128 | 64/128 | 0/8 | 0/64 |

All 128 fresh calibrated outputs were native A/B labels, with zero forced ties. Calibration recovered 15 answers and lost three relative to the archived renderer. With the same inputs and renderer, the two generic question families scored 24/32 and 27/32. The two topic-specific families scored 31/32 each. Only the walking-route topic answered all sixteen questions correctly. The screen required at least 95% in each family, seven strict topics and 90% opposed-margin flips. The observed 62/64 on topic-specific questions is a within-panel contrast. These generic and specific stems differ in phrasing as well as topic cues.

Current evaluations batch the two label orders for one stem. In the exposed canonical teacher comparison, 127/128 native tokens agree with the earlier four-item batches across two stems. The archive-plus teacher on the fourth stem with reversed options changed from A to the correct B despite identical injected rows and question. All newly reported routes use the same pair layout. The 24 original lexical controls replayed exactly in their original layout. [Raw native pairs](run/native_pairs.jsonl) · [Historical comparison and paired counts](paired_diagnostics.json)

## Source action and clause role

Each source describes both actions. Reversing which action is accepted leaves its word multiset intact. The model received one ordinary E tensor. A fitted topic score chose one of eight topics. A within-topic score read which action appeared first, while a shared score read whether the first clause accepted its action. The signs jointly selected the prescribed action. The direct comparator instead used one fitted affine preference score for the inferred topic. Both readers used the same 512 fitting descriptions and the same training-only standardization. Neither had access to a question, source-order label or topic ID at inference.

Both readers classified all 512 fitting policies correctly, including all 128 strict units. The fresh panel crossed two new constructions with exactly the same four action-expression pairs per topic as fitting. All 256 fresh topics were correctly inferred. Supplying the gold topic to the factor reader gave the same 188/256 policy decisions.

| Question-free source measurement | Priority frame | Verdict frame | Both |
| --- | ---: | ---: | ---: |
| Which action comes first? | 121/128 | 84/128 | 205/256 |
| Does the first clause accept its action? | 118/128 | 97/128 | 215/256 |
| Complete policy from the two signs | 111/128 | 77/128 | 188/256 |
| Direct within-topic polarity | 79/128 | 62/128 | 141/256 |
| Released Section 6.13 flat mixture | 82/128 | 70/128 | 152/256 |
| Always follow the inferred first action | 59/128 | 62/128 | 121/256 |
| Strict complete policies from the two signs | 20/32 | 5/32 | 25/64 |
| Strict policies from direct polarity | 4/32 | 0/32 | 4/64 |
| Correct opposed pairs with first clause role held fixed | 49/64 | 17/64 | 66/128 |
| Correct opposed pairs with action mention order held fixed | 49/64 | 23/64 | 72/128 |

A strict source unit must recover both poles in both clause orders. Both members of each opposed pair must have the correct complete policy. The two-sign reader recovered 81 states that the direct score missed and lost 34 it had classified correctly. On 176 states both component signs were correct. On 12 more, both signs were wrong and their errors canceled. All 12 occurred in the verdict frame. This distinguishes correct decisions from correct recovery of both roles. In the priority frame the preference cue precedes the action. In the verdict frame it follows the action. Vocabulary and syntax differ between frames, so cue position has no isolated effect estimate here.

Neither fresh frame met the complete-policy target of 90%, the strict target of 75%, or the 90% paired-exchange target. Topic identification survived this test, and the choice of how to read role and action affected policy recovery. The 188/256 is an internal policy classification on familiar actions. The end-to-end native route stayed closed under the registered rule. [Per-source scores, strict units and by-order results](summary.json) · [Source scores and hashes](run/sources.jsonl)

## Linking back to Section 6.13

Joining its 512 exposed four-style native decisions to their 128 source assignments gives 278/328 native correct when the highest-weight policy was correct and 30/184 when it was wrong. The exact-policy renderer on those two subsets scored 292/328 and 156/184 respectively. The old map used a continuous mixture. Its most heavily weighted policy does not uniquely determine the native choice. These observed joint counts explain why multiplying marginal accuracies cannot recover the earlier result. The 14.27% projected difference energy applies to 16 Section 6.12 opposed pairs on that study's old withheld wording and rank-31 span. It does not estimate the fraction of binding information lost on either later panel.

## Closure and reproduction

Both component results used one pinned Gemma-2-2B-IT and eight known topics. Source fitting included 224 E hashes that could be checked against published captures. All matched. The 256 evaluation texts were new. All 16 canonical teacher hashes matched, the calibrated renderer replayed its rows and checked fitting logits exactly, and all 512 reference source scores replayed after serialization. The run used 1,539 forwards and 211,844 processed tokens. The independent qualification set remained sealed. The conditional source-to-native composition and two-stage G were not run. [Terminal record](run/terminal.json) · [Artifact terms](ARTIFACT_LICENSE.md)

The next controlled comparison can hold the choice prompt fixed and vary only the presence of a neutral topic cue. On the source side, matching action words and evaluative vocabulary across cue-before and cue-after constructions would isolate what the two observed frames confound. New material and a new seal are needed for those tests. The continuous intermediate state, distinct stage targets, operation omissions and same-boundary L/B comparison remain prerequisites for a recurrence claim.

The plan, fixtures, runner, source reader and scorer were published at 270fe5b64fea86639a53b643bc48103b170fa072 before model execution. On that source commit, using the pinned local Gemma checkpoint and package environment, run the following from the project root.

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 PYTHONPATH=src python scripts/fit_teacher_and_polarity.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/teacher-polarity
python3 -S provenance/teacher_polarity_20261103/summarize.py runs/teacher-polarity
~~~

To verify the published archive and both reductions, run these commands from this experiment directory.

~~~bash
shasum -a 256 -c SHA256SUMS
python3 -S summarize.py
python3 -S paired_diagnostics.py
~~~
