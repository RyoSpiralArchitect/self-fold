# Four source styles yield a small within-topic gain and leave polarity transfer open

On the prospective eight-topic panel, the four-style map answered 308/512 native decisions correctly (60.16%) and the matched single-style map answered 290/512 (56.64%). The difference was 18 decisions or 3.52 percentage points. Four-style training recovered 100 decisions and lost 82 on the same inputs. Both arms answered 208 correctly and missed 122. The predeclared ten-point diversity effect and the 90% wording-transfer screen both failed. The independent 6.12 admission result retains its original scope. [Prospective plan](PLAN.md) · [Complete reduction](summary.json)

## Matched comparison

All eight previous fitting topics were retained. Each arm used 512 distinct question-free E states, the same 16 lexical teachers, four action expressions per pole, both source-clause orders and the same 64-update schedule. Every action-expression pair, pole and order had a counterpart in the other arm. The single-style arm used one syntactic frame throughout. The four-style arm distributed four frames across the same pairs in a balanced assignment.

The fresh primary sources crossed two unseen frames with two new action-expression pairs per topic. Two previously unused question stems and both label mappings were presented after each state had been constructed. This gives 128 distinct source states and 512 repeated decisions in eight topic clusters. A strict unit requires both poles, both clause orders, both tested question wordings and both label orders to be correct on one topic/style/expression pair. The primary unit count is 32. The underlying model is one pinned Gemma-2-2B-IT at one seed.

| Source and question panel | Single-style native | Four-style native | Four-style strict units |
| --- | ---: | ---: | ---: |
| Fitting states and questions | 2,012/2,048 | 1,962/2,048 | 98/128 |
| Fitting states, fresh questions | 1,728/2,048 | 1,740/2,048 | 64/128 |
| Fresh style and expressions, fitting questions | 303/512 | 311/512 | 2/32 |
| Fresh style, expressions and questions | 290/512 | 308/512 | 2/32 |
| Fresh style and questions, familiar expressions | 160/256 | 166/256 | 3/16 |
| Familiar grammar, fresh expressions and questions | 158/256 | 150/256 | 0/16 |

The four-style endpoint scored 154/256 under each fresh frame, with one strict unit out of sixteen in each. Its fitting accuracy was 95.80%, while 98/128 strict fitting units (76.56%) fell short of the 90% fitting requirement. The single-style fit also missed that strict requirement at 103/128. The separately declared syntax-only screen failed in both arms. All topics remain in these counts.

## Where the mapping failed

The map retains train-only global statistics, shared whitened policy directions with at most fifteen contrast coordinates and the sixteen canonical teacher tensors. At inference a single E tensor generates continuous mixture weights. A 16-by-16 calibration and a shared rank-four row residual contribute 18,688 gradient-trained parameters. The exported map stores no individual training E anchors. The frozen lexical transformation and the base model then produce the native answer. The highest-weight teacher gives a separate diagnostic of topic and polarity.

On fresh sources the four-style map assigned the correct topic on **128/128** states and the correct policy on **82/128**. All 46 assignment errors selected the opposite pole of the correct topic. The single-style map assigned the correct topic on 128/128 and the correct policy on 77/128. The two arms' fitting-bank policy counts at initialization were each 512/512. Their final fitting-bank policy counts were 483/512 and 508/512 respectively for four and one style. These are internal teacher assignments. Native task decisions are measured independently in the table above.

| Input to the frozen recipient on the joint panel | Native correct / 512 | Strict / 32 |
| --- | ---: | ---: |
| Literal source | 496 | 27 |
| Raw E | 157 | 0 |
| Canonical lexical teacher | 424 | 12 |
| True teacher through the four-style map's learned renderer | 448 | 16 |
| Released Section 6.12 endpoint, matched E | 277 | 0 |
| Four-style endpoint, matched E | 308 | 2 |
| Four-style endpoint, opposed E | 204 | 0 |
| Four-style endpoint, training-mean E | 256 | 0 |
| Four-style endpoint, zero E | 256 | 0 |
| Four-style endpoint, norm-matched fixed directions | 256 | 0 |
| Four-style endpoint, reversed E rows | 256 | 0 |

All 512 matched four-style outputs were native A/B labels. Opposing donors flipped the A/B margin sign on 106/512 decisions. The strongest fixed null reached 256/512 and the opposed donor reached 204/512, both scored against the original gold. Direct canonical teachers gave 82.81% on these new question stems, and supplying the correct teacher to the new renderer gave 87.50%. Both known-domain teacher references missed the 90% native and 75% strict thresholds on this particular panel. The 6.12 teacher count on older wording (116/128) could not guarantee the same reliability on new questions. The 16-policy closed vocabulary limits the scope of all claims here. New-topic extrapolation remains a separate unresolved result.

## Checking the proposed old-anchor explanation

We recaptured the 64 archived 6.12 E tensors in these eight topics and matched their hashes. Using that experiment's own training mean and scale, the old 32-anchor row span had numerical rank 31 at the specified cutoff. Its fitting states and paired opposed differences had essentially all their squared norm in the span by construction. For the withheld old wording the mean state projection fraction was **30.65%** (range 15.50–43.01%, 32 states). The corresponding opposed-difference fraction was **14.27%** (range 8.64–17.59%, 16 pairs). This directly measures reduced overlap, particularly for the polarity contrast. Appreciable overlap remains, and these measurements alone do not identify the cause of a wrong native answer. [Individual projections](run/geometry.jsonl)

## Technical integrity and closure

The first attempt stopped before fitting when serialization changed the projection's storage layout. Its values matched and its checked native logits matched, but a projected BF16 row differed by at most 0.0001220703125. The repaired loader preserved the saved array order. Before the first update the new run replayed all initial weight values and strides, request, source captures, teacher records, schedule, earlier lexical controls and projection diagnostics exactly. The original records remain public. [Stopped attempt](attempt_01/terminal.json) · [Repair and numerical receipt](TECHNICAL_REPAIR.md) · [Checked replay](run/repair_replay.json)

Each arm completed 64 updates, 512 training forwards and 161,536 training tokens. All 128 gradient norms were positive and finite. The four saved endpoint checks reproduced projected rows and native logits exactly. Both fits were completed and serialized before any new endpoint was scored. The repaired run used 9,761 forwards and 2,800,170 processed tokens. It retained all 35,328 native decisions, with separate coverage, forced preferences and ties. Base and lexical parameter hashes remained fixed. [Terminal receipt](run/terminal.json) · [Artifact conditions](ARTIFACT_LICENSE.md)

The many-to-one construction preserved topic identity on the tested source bank. The registered policy-transfer gate failed and stage B stayed closed. Two measurements remain necessary. The frozen recipient must use the supplied correct lexical policy across fresh questions. The source map must also discriminate preference direction within each known topic across fresh source wording. A later study can measure those separately on a newly sealed panel. A two-stage continuity claim also requires distinct intermediate reconstructions and omission controls, because a per-policy teacher is shared across meaning-preserving stages. G, intermediate exchange, L/B insertion and independent qualification remain open.

## Reproduce

The scientific plan, fixture and reducer were published at 3a7d4fc98752023423b4be75a32f415b77f5525d. The loader repair and pre-update replay check were published at a813911f6c4dd725b7ad3e8d03bdf3057af272a4. To replay the repaired fit at the latter commit, use a local copy of the published stopped attempt.

~~~bash
PYTHONPATH=src python scripts/fit_within_topic_wording.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --first-attempt "$ARCHIVED_FIRST_ATTEMPT" --out runs/wording-repaired
python3 provenance/within_topic_20261103/summarize.py runs/wording-repaired
~~~

In the published study directory, verify the files and recompute the recorded measures.

~~~bash
shasum -a 256 -c SHA256SUMS
python3 summarize.py
~~~
