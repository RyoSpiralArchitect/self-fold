# Native recipient learning for two-stage functional continuity

This study makes the original SELF-FOLD question operational at its first interface. A later two-step reconstruction needs a question-independent state that a recipient can use. The new development bank provides conditional policies, supported explicit-case and reintegrated teachers, and self/other/impersonal frames. Each teacher has the same exhaustive four-world truth table as its source. Human transition support and usefulness review are pending. [Integrated protocol](../../docs/FUNCTIONAL_CONTINUITY.md) · [Public material](../../data/continuity_20261103/manifest.json)

## The frozen recipient comparison

We froze raw Gemma-2-2B-IT workspace E, eight slots at pre-cut layer 13, from a description-only forward of 256 tokens. The source and both teacher stages supply 1,152 train variants and 288 dev variants each, across 96 train and 24 dev groups. Construction grammars, action/flag words and sites are held out, while Boolean truth functions and target rendering grammars overlap. Source questions are constructed only after all 4,320 logical donor records have been captured.

A task-scoped rank 16 q/v adapter in the final four decoder layers minimizes full-vocabulary native A/B cross entropy for 256 AdamW updates. Each effective batch has 32 questions, balanced over both policies, both source orders, four worlds and both label mappings. Only the receiver is active in task forwards. A second candidate uses the same initialization, schedule and budget with fixed train-only coordinate standardization at task injection. Its execution was conditional on the raw candidate's fixed final endpoint missing the screen.

| Matched teacher/state readout | Raw native correct /4,608 | Standardized native correct /4,608 | Literal base native correct /4,608 |
| --- | ---: | ---: | ---: |
| Source | 2,307 (50.07%) | 2,299 (49.89%) | 3,269 (70.94%) |
| Explicit-case teacher | 2,286 (49.61%) | 2,305 (50.02%) | 3,448 (74.83%) |
| Reintegrated teacher | 2,294 (49.78%) | 2,308 (50.09%) | 2,886 (62.63%) |

The receiver's primary compound unit requires correct native greedy answers under all four polarity/order tags at a fixed group, frame, world, question wording and label mapping. The raw stage2 candidate met 52/1,152 units (4.51%), and standardized met 17/1,152 (1.48%). Stage2 rendering is canonical and its two order tags are byte-identical. This compound counts two real opposed policies with duplicated order inputs. Clause-order robustness must be examined in source and case-stage panels, and eventually on the actual learned G chain.

Both candidates missed the registered 90% native accuracy, 10-point null advantage and 75% joint screen. Their strongest fixed null scored 50%. All native argmax outputs were A or B, but many A/B likelihoods tied exactly. In stage2 there were 3,872 raw and 3,238 standardized ties out of 4,608, which counted against forced-choice success. Native greedy output uses the model's deterministic argmax, so its score can differ from the tie-as-failure forced-choice score. [Every decision and control](recipient/functional.jsonl) · [Recomputed summary](summary.json)

Both fits are complete at their cap. The backbone parameter digest matched before and after; the run used 512 total optimizer updates, 12,249 batched/donor forwards and 7,674,574 processed tokens. It completed in 397.93 seconds on one RTX 5090. The training loss rapidly approached log2. The model-source contribution targeted by the protocol remains unestablished in these recipients. The actual two-step G fit, middle-state exchange and language-insertion experiment have not entered their next phase.

## Execution and remaining claims

Source 77c4ab6623cd843b730bc2b0635b62f998aef1c5 fixed the corpus, loss, architecture, candidate order, stage-specific scoring and negative closure before the pretrained run.149 engineering tests, a tiny T/L/B smoke and all 4,320 native-tokenizer layouts passed before fitting. The longest rendered donor used 137 tokens inside the fixed 256 geometry. Logical donor rows shared 2,808 unique source forwards.

The old sealed 24-group qualification material stayed outside this experiment. The new material contains no confirmation panel yet. A new source or numerical diagnostic can follow these recorded failures under a separate plan. A successful final-teacher recipient would still be an interface result, preceding the actual learned recurrence and independent qualification. The two adapters and train-state statistics have the [Gemma conditions](recipient/ADAPTER_LICENSE.md).

To regenerate the full study summary from its receipts, run the following from the repository root.

~~~bash
python3 provenance/continuity_20261103/summarize.py provenance/continuity_20261103/recipient
~~~

## Zero-update numerical audit

A separately fixed audit at source 35c71a1aada6e2503e02e98e64f45ddba9b60706 replayed 48 source rows and evaluated 384 tasks for each of the initialized, raw and standardized recipients. It observed the native head without replacing its output. All original source-state hashes matched, all backbone/adapter parameter hashes matched before and after, and the audit made zero optimizer updates. It retained 1,152 task records and 576 opposed-source hidden-state pairs. [Audit contract](AUDIT_PLAN.md) · [Per-case numerical observations](numerical-audit/cases.jsonl) · [Summary](numerical-audit/summary.json)

| Audit endpoint /384 tasks | Initial | Raw learned | Standardized learned |
| --- | ---: | ---: | ---: |
| Exact ties after native softcap | 0 | 308 | 316 |
| Ties introduced after unequal BF16 head values | 0 | 174 | 146 |
| Two-column FP32 projection, strict correct | 192 | 185 | 203 |

No observed opposed-source hidden-state pair was exactly identical. The native stored-tanh derivative was nonzero for all observed label entries. Thus this panel found additional rounding ties, while the stronger zero-derivative saturation hypothesis was not observed. Recomputing two head columns in FP32 removed exact ties but yielded 48.18% and 52.86% strict accuracy for the trained recipients. This computation did not demonstrate a working recipient, and it did not alter the recorded native outputs.

The raw audit's 174/384 post-only ties met the separately declared 10% condition for one fresh conditional-label objective comparison. That next phase uses the same untouched native BF16 softcap and evaluates full-vocabulary native greedy output. It neither resumes the failed weights nor uses the diagnostic head to break ties. [One-candidate follow-up contract](BINARY_PLAN.md)

## One matched conditional-label follow-up

The numerical condition admitted the one-candidate follow-up at source 1dc37cdaee70212bd2f449931e1bfde2adecaf2e. It independently initialized the raw recipient and used conditional A/B cross entropy from the unchanged native capped logits. All 4,320 donor hashes, the initial parameter digest, ordered schedule, training forwards and processed-token count matched the original raw fit.

At the fixed 256-update endpoint, every matched native output was A in all three representation stages. Each stage obtained 2,304/4,608 native answers correct (50.00%), all 4,608 A/B likelihood comparisons tied, and zero of 1,152 compound units were correct. It missed the original screen. The run completed in 196.51 seconds with 6,809 forwards and 3,359,310 processed tokens, including replayed donors and evaluation. No trained weight was resumed. [Full follow-up record](binary) · [Recomputed summary](binary_summary.json)

This closes three configurations of the frozen-E recipient. The separately proposed next construction jointly trains a source-only encoder and task-only recipient, while retaining source/question isolation. That architecture has its own fixed budget and entry criteria. [One joint candidate](JOINT_PLAN.md)
