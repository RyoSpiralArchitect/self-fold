# E reaches the frozen recipient on the fitted source bank

The final source-only map let the frozen Gemma recipient answer all 64 training decisions correctly. Opposing E donors reduced that count to 0/64 and flipped every A/B margin sign. Eight strict units required both source-clause orders, both donor poles and both option assignments; all eight passed. On the same source states with a question wording absent from training, native correctness was 62/64 and forced A/B correctness 64/64. This is a fitted-bank connection from ordinary E states to native answers. [Preregistered plan](PLAN.md) · [All scores and controls](summary.json)

The separate source-generalization screens failed. New source wording within the training topics scored 57/128 native correct, and the two new topics scored 30/64. No strict unit passed on either panel. The fixed endpoint passed its training screen, while complete interface admission remains false. The run ended at 64 updates.

## Fixed architecture and supervision

E, Gemma's base parameters and the successful lexical transformation stayed frozen. The interface hash was fcedb9502207336dc00fd307b82d2c6632aa7078d392e9322f8cebb2a9e2b096. Its original 24 matched-length decisions replayed exactly, including every A/B score and native token. The new canonical lexical teachers used a question-free, fixed 256-token capture. Their geometry differs from the original recipient-length-matched control.

The new source map first fitted a centered dual-ridge map from 32 training E states to their lexical teachers. This is a stored linear map with training anchors. A rank-16 residual then learned for 64 updates with 589,824 trainable parameters. It mixes the flattened eight input rows and can produce all eight output rows. The complete mapping includes the fixed ridge component. Only its residual parameters were optimized. Neither source identifiers nor source text enter the mapper call, which receives one E tensor alone.

The objective combined conditional native A/B loss, teacher-relative representation error and a teacher-relative variance floor, weighted 1, 0.1 and 0.01. Each batch contained four topic/variant units with both poles, giving eight sources and sixteen native decisions. The plan fixed all weights, the ridge constant, batches, endpoints and gates before scoring. There was one ridge solve and one residual fit, with no checkpoint selection. This cycle has no loss ablation, so the contribution of each term remains unresolved.

## Material and cross-clause controls

The prospective development fixture has twelve topics and 96 newly authored descriptions. Each description contains a preferred alternative and a rejected alternative. Its paired source presents exactly those two clauses in reverse order. Thus an answer based only on the first-mentioned alternative cannot pass the strict unit. Eight topics supply 32 fitting states through their first wording style and both orders. A second wording style remains outside fitting. Four of the training topics reuse the earlier lexical meanings, while four concern length units, clocks, mail and music.

Shelf and directions are excluded from this source-map fit but were exposed in the previous lexical study. Photo organization and recipe weights are new development topic combinations. None of these four topics enters the ridge solve, normalization, residual loss or training schedule. Human semantic review is pending, and the independent qualification material remains sealed. [Fixture](../../data/source_lexical_20261103.json)

| Panel | Literal source | Raw E | Canonical lexical teacher | Ridge map | Final map | Final strict units across both source orders |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fitting states, fitting question | 64/64 | 14/64 | 58/64 | 58/64 | **64/64** | **8/8** |
| Same states, withheld question wording | 64/64 | 22/64 | 58/64 | 58/64 | **62/64** | **7/8** |
| Withheld source style, training topics | 124/128 | 36/128 | 116/128 | 60/128 | 57/128 | 0/8 |
| Previously exposed dev topics | 63/64 | 24/64 | 48/64 | 29/64 | 29/64 | 0/4 |
| New development topics | 64/64 | 20/64 | 56/64 | 30/64 | 30/64 | 0/4 |

All entries are full-vocabulary native greedy correctness. The raw E fitting panel had only 28/64 A/B outputs, with 32/64 forced A/B correctness. The final fitting panel had 64/64 A/B outputs, no forced ties and a mean pole gap of 13.3789 nats. Its withheld-question panel had 64/64 correct forced preferences, but two greedy outputs were outside A/B and remain incorrect.

The teacher's new-topic native correctness was 56/64, with only 2/4 strict units across both source orders and both question wordings. It missed the registered strict transfer threshold. The literal source was correct on all 64 new-topic decisions. There are therefore limits both in transferring E to the lexical coordinates and in the frozen lexical control's reliability under these new conditions. The teacher's 116/128 on the withheld source-style panel versus the map's 57/128 separately exposes the source-map generalization gap.

## Donor interventions and null bias

| Final map input | Fitting questions / 64 | Withheld question wording / 64 | Withheld source style / 128 | New topics / 64 |
| --- | ---: | ---: | ---: | ---: |
| Matched E | 64 | 62 | 57 | 30 |
| Opposed E | 0 | 0 | 51 | 30 |
| Training-mean E | 32 | 32 | 64 | 32 |
| Zero E | 32 | 32 | 64 | 32 |
| Fixed directions with the source's row norms | 32 | 32 | 64 | 36 |

The matched/opposed A/B margin sign flipped on 64/64 fitting and 64/64 withheld-question decisions. Those counts dropped to 22/128 for withheld source wording and 8/64 for new topics. Mean and zero inputs had zero donor pole gap on every panel, while their absolute option margins retained topic-specific biases. Null correctness averages over the balanced donor poles, and signed option margins are reported by topic. The reversed-row control, all signed option biases, coverage and exact ties are included in the standalone reducer output.

## Closure and next anchor

The completed endpoint passes only the preregistered training screen. Source-style and new-topic screens both fail, so the route is not admitted for the later G experiment. The measured result is that the original E bank can drive a frozen native decision path after fitting a source-only map, including a new question wording and both clause orders. Generalization to different descriptions of those preferences remains unresolved. The stored training anchors also make fitting-set interpolation a live explanation.

A next development design can use several source styles during fitting while reserving a newly authored style and topic panel before execution. It also needs a fixed lexical teacher with reliable strict-pair performance on that prospective panel. The current failed evaluation material stays exposed and cannot become a fresh confirmation set. The fixed ridge and residual endpoints remain controls. No further fitting occurred in this cycle.

## Execution and reproduction

The source was publicly sealed at 9c0c4965e0d555d7ebc20c9f8a3581273c56e0b2. All 64 residual gradient norms were positive and finite. The backbone and frozen lexical transformation hashes matched before and after. The two saved endpoints passed checked row and native-logit save/load replay. Training used 256 native forwards and 80,128 processed tokens. The complete run used 1,869 forwards, 551,434 processed tokens and no generated tokens. All 5,760 decisions are retained, including every control and failed transfer cell. [Execution receipt](run/terminal.json) · [Artifact conditions](run/ADAPTER_LICENSE.md)

The returned raw archive had SHA-256 37b37ad51c26bc72d4b15d0405feea73f3063a77dc59cf7d7279f8ed89a4ec8a. Run from the repository root at the source commit above.

~~~bash
PYTHONPATH=src python scripts/fit_source_lexical.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/source-map
python3 provenance/source_lexical_20261103/summarize.py runs/source-map
~~~

From this study directory, verify the archived files and recompute the result.

~~~bash
shasum -a 256 -c SHA256SUMS
python3 summarize.py | diff - summary.json
~~~
