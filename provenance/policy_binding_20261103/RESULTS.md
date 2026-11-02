# When the donor policy reverses

Each of two Gemma-2-2B-IT encoder bridges saw the same 512 training descriptions in the same order over 64 updates, starting from identical parameters. Each processed 44,804 training tokens with 512 backbone forwards. The second loss added one binary choice at the first source token that differs between opposed policies. The rest of the source up to that token is identical. Both endpoints were fixed before scoring. All measurements below use the development families. [Frozen comparison](PLAN.md) · [Per-case records and learned arrays](run) · [Summary](summary.json)

| Development endpoint | Copy | Copy + opposed-token loss |
| --- | ---: | ---: |
| Mean source NLL, nats/token | 1.8207 | 1.8289 |
| Correct first differing token | 44/96 | 44/96 |
| Both opposed donors correct at this token | 0/48 | 0/48 |
| Group-macro functional accuracy (pooled count) | 49.35% (283/576) | 50.00% (288/576) |
| Correct action for both opposed policies, fixed world/order | 0/288 | 0/288 |
| Both wrappers correct, group macro | 46.35% | 50.00% |
| Both contrasting worlds correct, group macro | 30.90% | 33.33% |

The absolute functional difference was +0.65 percentage points (scope +0.00, priority +1.30). The descriptive paired site-resampling interval was +0.26 to +1.04 points, conditional on just two constructions. The practical gain threshold was ten points. Literal text reached 87.89% group macro (483/576). The strongest donor-independent control was 50.00% for both arms. The copy endpoint tied on 13 functional comparisons, the opposition endpoint on zero. Equal scores count against strict success. Replacing the divergent token and everything later with placeholders at identical forward length reproduced all 192 lexical margins exactly. Neither trained bridge met the functional or lexical screen.

## A direction without a single-source decision

A nearest-centroid classifier fitted on raw **unadapted** train slots ordered the two opposed dev donors by first-mentioned tray correctly in 48/48 pairs. Its fixed decision boundary classified only 48/96 single donors. The within-family shuffled-label control gave 14/48 ordered pairs and 48/96 individual decisions. Comparing both members of a labelled pair reveals relative separation that a model presented with one unknown donor cannot directly use. These labels describe mention order. The result does not establish that a rule or its consequences can be read from an individual state.

The comparison ended at the registered cap. Both bridges failed the strict swap endpoint. The single-source raw probe also missed its 80% screen. The next registered branch is a distinct capture/geometry experiment. Both trained arrays remain available with their [Gemma conditions](run/ADAPTER_LICENSE.md).

## An action-order control for the geometry test

In the original dev descriptions, the first-mentioned action is always the action for the construction's trigger. A deterministic rule that knows the family's trigger and chooses first when true, second otherwise gets 288/288 description-world cases correct. We constructed two renderings of each opposed policy whose clauses are identical sentences in opposite orders. The same shortcut gets 144/288 correct. Both versions were checked against the original truth tables. The two panels expose the same 24 groups. They supply a controlled wording change, with no new independent meaning families. These are text/ledger checks, with no native model score on the new wording in the two-arm fit. [Audit](order_audit.json) · [Crossed cases](order_controls.jsonl) · [Generator](order_controls.py)

Fitting used source `5e4bd86f9ca45fd0e3f0192fb165cc3eac76c7e5`; the scientific contract was fixed at `4c1c47189e4ea5ab7dfee8a1fa4acfbc58f1a97b`. The initial CPU preflight found an integration failure replaying the historical separator study from a newer tree (130 passed, one failed). Its original study bytes and receipt were kept, and its verifier ran on its exact study commit. The full prefit suite then passed 131 tests, plus a tiny T/L/B smoke and native-tokenizer checks on all 144 training/dev pairs. No pretrained fit was attempted before those checks passed. The backbone parameter digest matched before and after training, gradients were finite at all 128 updates, and neither G teaching nor qualification cases entered the losses.

## Testing a single source against its neutral rendering

The geometry follow-up was sealed at `65457166eb22d450802c2cc8f18ba09560ddb44a` after both fits, before observing this panel. It used the frozen Gemma with zero optimizer updates. Each source was encoded twice, once with its tray names and once with both names replaced by the same “metal” token. Native lengths and slot positions matched. Raw slots minus these reference slots supplied a feature for a train-only linear nearest-centroid decision. The reference requires the text of this source at initialization. It is not available from an arbitrary latent state later in recurrence. [Frozen follow-up](GEOMETRY_PLAN.md) · [Per-case geometry scores](geometry/scores.jsonl) · [Full breakdown](geometry_summary.json)

| Crossed wording, 24 groups | Raw slots | Source minus neutral | Shuffled train labels, centered | Reference only |
| --- | ---: | ---: | ---: | ---: |
| First-mentioned tray, single description | 48/96 | **93/96** | 21/96 | 0/96 (96 ties) |
| Trigger action, single description | 48/96 | **51/96** | 49/96 | 0/96 (96 ties) |
| Trigger action correct in both clause orders of a policy | 24/48 | **6/48** | 2/48 | 0/48 |

The raw decision predicted copper on every crossed description. With centering, first-mention accuracy reached 47/48 on scope and 46/48 on priority. It met the registered lexical screen. No feature met the trigger-action screen. Sorting the centered trigger scores by clause order after the run gives 45/48 correct with the trigger clause first and 6/48 with it second. Both word orders prescribe exactly the same action in the trigger world. This isolates a wording sensitivity in these readouts. It does not establish a unique mechanism for the frozen model.

All 288 earlier raw donor hashes and all 192 labelled/sham margins replayed exactly. The geometry run recorded 576 unique captures for 384 sources and their references. No recipient question entered a donor capture. The train-only linear decision has access to explicit lexical contrasts and a controlled reference. This result is a positive development finding about mention order. Functional rule use, the semantic quality of a G step, and transfer to a later independent task remain open.

## Next experiment and stopping point

The next proposed learning target is the association between **condition and action**, with both clause orders represented on the training side. Freeze E before introducing a separately supervised recipient. Cross policy polarity with mention order, hold out whole constructions, expose the world only to the recipient, and compare matched, opposed, shuffled and absent sources in both A/B mappings. A new development panel is needed for selection. These 24 groups have exposed outcomes. The current cycle ends here. [Positive and negative routes to G and the independent challenge](../../docs/SEMANTIC_GATE_PLAN.md)

Rebuild both summaries and check all committed evidence from this directory.

~~~bash
python3 summarize.py | diff - summary.json
python3 summarize_geometry.py | diff - geometry_summary.json
python3 order_controls.py --panel | diff - order_controls.jsonl
shasum -a 256 -c SHA256SUMS
~~~
