# A single E state identifies the first verdict on unfamiliar action wording

The trained role head correctly identified whether the first record accepted or rejected its action on **2,043/2,048** held-out states. It read both members correctly on **1,019/1,024** fixed-action-order exchanges. This passed the separately registered first-role criterion in all four combinations of cue vocabulary and field placement. Recovering the **identity of the preferred action** remained harder. The primary reader classified 1,311/2,048 complete policies correctly and switched correctly on both members of 627/1,024 opposed pairs. The full source-recovery criterion failed in every cell. [Sealed plan](PLAN.md) · [Registered reduction](summary.json)

## What the reader had access to

One source contained the two Section 6.15 preference records. In an opposed pair, the actions occurred in the same order and traded verdicts. Both members contained exactly the same native tokens, as a multiset, and their eight workspace slots occupied the same positions. The two verdict vocabularies and two field placements were familiar. Evaluation introduced 64 new action strings across eight known topics. Fully crossing four plus with four minus expressions per topic, both preferred poles and both mention orders yielded 2,048 states. These are repeated realizations of 16 known policies in eight topic clusters.

Gemma-2-2B-IT and the topic router were frozen. The training bank crossed all six previously exposed action expressions per pole, giving 4,608 states in 2,304 fixed-order exchanges. One shared affine score read whether the first record accepted its action. A topic-specific affine score read which action appeared first. The product of the signs selected the preferred action. The primary fit weighted within-pair differences four times as heavily as pair midpoints. An ordinary fit used unit weight, equivalent to ordinary singleton ridge with the same examples, ridge strength and architecture. A null role head flipped half the training pair orientations within each balanced stratum.

Both readouts were serialized, reloaded and checked on all 4,608 fitting states before the first held-out source was encoded. Each evaluation call took one E tensor. It inferred the topic and received neither the opposing tensor nor any question, source text, order label or test-bank normalization. [Readout implementation](../../src/selffold/role_exchange.py) · [Saved fits](run/fits.jsonl) · [Fresh boundary](run/fit_closed.json)

## Which sign transfers?

| Held-out verdict vocabulary and placement | First role correct | Shuffled role correct | Both roles correct in opposed pair | Complete policy | Both policies correctly switched |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accept/reject before action | 511/512 | 256/512 | 255/256 | 328/512 | 158/256 |
| Accept/reject after action | 512/512 | 260/512 | 256/256 | 318/512 | 149/256 |
| First/last choice before action | 510/512 | 246/512 | 254/256 | 334/512 | 163/256 |
| First/last choice after action | 510/512 | 260/512 | 254/256 | 331/512 | 157/256 |
| **All** | **2,043/2,048** | **1,022/2,048** | **1,019/1,024** | **1,311/2,048** | **627/1,024** |

For each vocabulary and placement, the first-role criterion required 95% singleton accuracy, 90% paired correctness and a twenty-point lead over shuffled role supervision. The continuous role score ranked the accept-first member above its counterpart in all 1,024 pairs. The 1,019/1,024 paired count additionally requires both singleton scores to fall on the correct side of the fixed zero threshold.

With one accepted and one rejected action in every source, the final verdict also determines the first. The experiment establishes access to the order of familiar verdicts through the frozen E interface as action wording varies. It cannot identify which verdict token supplied the signal or establish transfer to new ways of expressing acceptance and rejection.

## Why complete policy recovery still fails

The primary fit passed its fitting check with 4,607/4,608 complete policies and 1,151/1,152 strict units. On held-out expressions it reached only 185/512 strict units, each requiring both poles in both clause orders. On fixed-action-order exchanges it switched correctly on 627/1,024, switched in reverse on 245, returned the same policy of the correct topic on 35, and left the correct topic on at least one member of 117 pairs. Its signed policy score ranked 645/1,024 pairs in the gold direction when both topics were correct, counting other pairs as misses. Only 18 additional pairs had a correct ranking without two correct singleton judgments. Holding the first-clause role fixed instead yielded 402/1,024 correctly switched pairs.

| Complete-policy readout on the same fresh bank | Correct states / 2,048 | Correct fixed-action-order exchanges / 1,024 |
| --- | ---: | ---: |
| Section 6.14 frozen prose-trained reader | 923 | 15 |
| Direct paired affine polarity within inferred topic | 973 | 81 |
| First action alone | 915 | 14 |
| Shuffled role times learned first action | 900 | 68 |
| Ordinary fit, first role times first action | 1,313 | 625 |
| Fourfold contrast weight, first role times first action | 1,311 | 627 |
| Fourfold weight with gold topic supplied, diagnostic | 1,443 | 701 |

The extra weighting recovered nine singleton decisions and lost eleven relative to the identically trained ordinary reader. It gained six correctly switched pairs and lost four. The registered ten-point weighting advantage failed for both measures. The ordinary role head obtained the same aggregate 2,043/2,048 and 1,019/1,024 as the primary head. The improvement over the older reader accompanies fitting on preference records and additional lexical combinations. It cannot be assigned to the fourfold weighting.

The 737 wrong policies separate into three groups. The frozen router assigned the wrong topic on 206 states. With the correct topic, 527 errors had the right verdict order but the wrong first action, and four had the right first action but the wrong verdict order. Among the 1,311 correct decisions, 1,310 had both signs correct and one contained two canceling errors. Giving the reader the correct topic for every state raised complete recovery to 1,443/2,048. This leaves action identity as a distinct limitation even when routing is supplied. [Descriptive error partition and ordinary-role counts](error_partition.json)

The literal-source control used the first Section 6.15 bare question and both native label orders for every source. It scored 2,905/4,096 full-vocabulary judgments, with 4,090 native A/B labels, 19 forced-logit ties and 1,113/2,048 sources correct under both labels. No literal observation was used to filter the source-readout panel. [Native observations](run/literal.jsonl) · [All singleton scores](run/scores.jsonl)

## Decision and replay

The source-only first-role result is a development result at one model and within the two known verdict vocabularies. The full source gate stays closed. A next experiment can freeze the successful role head and change which action appears first while holding the verdict sequence fixed, training the action-identity head alone. Comparing gold-topic and inferred-topic results on separately sealed action wording would isolate routing from action identity. A native recipient and the two-stage G intervention still require their own evidence.

The prospective source commit was d93b140fb743de8ffbf814ba7af1be4e4bd12b3c. The run replayed all 512 archived Section 6.15 source hashes and old decisions, used 8,705 model forwards and 993,742 processed tokens, and fitted the readouts analytically with zero backbone updates. [Terminal receipt](run/terminal.json) · [Gemma-derived readout conditions](ARTIFACT_LICENSE.md)

From the sealed source and the pinned Gemma environment, rerun the bounded fit in a new output directory.

~~~bash
HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src python scripts/fit_role_exchange.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/role-exchange
python3 -S provenance/role_exchange_20261103/summarize.py runs/role-exchange
~~~

To use the saved reader with a compatible single E tensor, load the weighted NPZ with `load_weights` and call `score_one`. The topic comes from the embedded frozen router. Multiply the returned role by the action score for that topic. A positive product selects the plus policy and a negative product the minus policy. A tied router or zero product abstains.

From the published evidence directory, verify the files and recompute both reports.

~~~bash
shasum -a 256 -c SHA256SUMS
python3 -S summarize.py
python3 -S error_partition.py
~~~
