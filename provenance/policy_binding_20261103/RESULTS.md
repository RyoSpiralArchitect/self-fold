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
