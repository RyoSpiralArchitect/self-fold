# One topic word and the position of a verdict

Adding the correct topic word to four fresh questions gave the frozen lexical teacher 99/128 correct native decisions, versus 101/128 for the identical questions without a heading. Swapping the verdict and action fields in preference records changed how well the base model read the original text. With the verdict after the action, it answered 452/512 choices correctly, versus 358/512 with the verdict first. The frozen source reader recovered the policy on 130/256 and 134/256 states respectively. The two primary effect criteria were unmet. [Prospective plan](PLAN.md) · [Registered reduction](summary.json)

## Teacher with the same question below each heading

We kept the calibrated Section 6.14 renderer, the canonical teachers and the native recipient fixed. Each of the eight topics appeared under both policies, both A/B assignments and all four new stems. The three headings each occupied one native token plus a newline. Across 64 prompt layouts, substituting a heading changed exactly one token. Removing the heading token and newline reproduced the entire bare input, with the injected teacher slots at the same positions.

| Heading | Native correct | Strict topics | Opposed margin reversals | Correct by question family |
| --- | ---: | ---: | ---: | --- |
| None | 101/128 | 2/8 | 37/64 | 25, 27, 24, 25 of 32 |
| Correct topic | 99/128 | 3/8 | 35/64 | 24, 25, 25, 25 of 32 |
| General | 96/128 | 1/8 | 32/64 | 24, 25, 24, 23 of 32 |
| Rotated topic | 96/128 | 1/8 | 32/64 | 25, 24, 24, 23 of 32 |

The correct heading recovered two decisions and lost four relative to bare. Both were right on 97 and both missed 25. Its mean change in the gold-signed A/B margin was −0.269 log units. Removal of one topic at a time gave net accuracy changes from −2.68 to 0 percentage points. The correct word gained three net decisions over each one-word control. Neither comparison reached five points. All calibrated outputs were A/B, with zero ties. Common-mean teacher accuracy was 64/128 bare, 63/128 named, 64/128 with General and 63/128 with a rotated topic. The named teacher also missed the registered reliability thresholds.

In a descriptive recount, the named teacher made the correct opposite choices on 35/64 opposing-policy pairs and gave the same label to both teachers on the other 29. The corresponding bare counts were 37 and 27. The earlier 62/64 for questions that described the topic in a complete task sentence remains specific to those question forms. A heading alone did not reproduce that advantage in this panel. [Question token layouts](run/question_layout.jsonl) · [Native choices and log probabilities](run/teachers.jsonl)

## Keeping the verdict and action words identical

Each source used two records. For example, the first record could read `My preference record: [ I accept this ] [ placing the newest reports at the front of my archive ].` Its paired placement swapped only those two fields, and did the same to the second record carrying the opposite verdict. The two verdict vocabularies, four familiar action-expression pairs, both chosen poles and both clause orders were crossed in all eight topics. All 256 swaps preserved the complete encoder token multiset, sequence length and eight slot positions. The source reader received one E tensor and no question. The literal control used the complete records with one fixed bare question in both label orders.

| Measurement | Verdict before action | Verdict after action |
| --- | ---: | ---: |
| Topic from E | 256/256 | 256/256 |
| First action within the topic | 142/256 | 144/256 |
| Role of the first record | 132/256 | 132/256 |
| Topic and both component signs correct | 76/256 | 75/256 |
| Correct policy from the two signs | 134/256 | 130/256 |
| Correct policy with two wrong signs | 58/256 | 55/256 |
| Direct within-topic policy score | 130/256 | 128/256 |
| Strict source units across both orders and both poles | 0/64 | 1/64 |
| Correct opposed pairs at fixed action order | 6/128 | 4/128 |
| Correct opposed pairs at fixed first role | 10/128 | 9/128 |
| Literal native decisions | 358/512 | 452/512 |

Placing the verdict first gained nine policy decisions and lost five on the same 256 pairs, a net 1.56 percentage points. For recovery of both component signs it gained five and lost four, a net 0.39 points. Neither reached the registered ten-point effect. The policy difference was two decisions in each verdict vocabulary. All four source reliability screens, one per vocabulary and placement, failed on these records.

In contrast, the literal control gained 109 native decisions and lost 15 when the verdict moved after the action. Literal accuracy improved in all eight topics. Removing any one topic gave an improvement between 16.96 and 19.64 points. The two layouts scored 69.92% and 88.28% and differed by 18.36 points, so the predefined literal comparability guard failed. The prior Section 6.14 prose comparison used different constructions. This controlled result concerns the order of bracketed fields in a record and one literal question. [Source components and state hashes](run/sources.jsonl) · [Paired literal choices](run/literal.jsonl)

## What the opposed sources distinguish

A descriptive analysis after scoring paired sources with the same two action phrases in the same mention order, exchanging which action was accepted. The E tensor hash changed in all 128 pairs in each placement. Yet the frozen source reader assigned the **same policy to the two opposite sources in 122/128 pairs in each placement**. With the verdict last, 84 pairs received all four correct literal native choices across poles and label assignments. In 82 of those 84, the source reader still assigned the same policy to both inputs. It switched correctly on one and in reverse on one. With the verdict first, the 24 literal-complete pairs all received the same source-reader policy on both inputs. All primary counts above include all pairs. [Descriptive exchange counts](exchange_counts.json) · [Recount script](exchange_counts.py)

These observations locate a failure of the tested reader to act on exchanged source roles. A change in E bytes shows sensitivity to the input. Whether another reader can recover the role binding from that change remains open. A subsequent probe could fit a polarity direction on action-order-matched role exchanges alone and evaluate it on separately authored action pairs, both placements and all eight topics. Its fitting and held-out decisions should use a single E state without the opposing mate. A complementary teacher test can hold a topic name fixed and vary the operation requested in the question sentence.

## Execution and reproduction

The full experiment, dataset and scorer were sealed at 95db5846d7b5f78a0a36dd9b1e850051b3907e81 before any fresh model score. All sixteen canonical teachers, 128 prior native A/B score pairs and 256 prior E hashes and policy predictions replayed before the new panels. The run used 1,745 forwards, 242,378 processed tokens and zero parameter updates. It wrote 512 fresh source observations, 1,024 lexical-teacher readouts including common nulls, and 1,024 literal choices. The teacher readouts refer to 384 unique batches of two labels. Source-to-native composition, learned G and the independent qualification set stayed outside this study. [Replay](run/replay.json) · [Compute and frozen hashes](run/terminal.json) · [Source of the two frozen readouts](../teacher_polarity_20261103/ARTIFACT_LICENSE.md)

From the experiment source commit and pinned local Gemma environment, run the model with a fresh output directory.

~~~bash
HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src python scripts/probe_matched_cues.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/matched-cues
python3 -S provenance/matched_cues_20261103/summarize.py runs/matched-cues
~~~

From this published evidence directory, verify the recorded files and reproduce both reductions.

~~~bash
shasum -a 256 -c SHA256SUMS
python3 -S summarize.py
python3 -S exchange_counts.py
~~~
