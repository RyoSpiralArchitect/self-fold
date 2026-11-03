# Testing the grammar gap and learning two actual state transitions

## Three literal curricula

The previous component reader had recovered 41/48, 38/48 and 48/48 strict fitting units for descriptions, cases and rules. Its jointly new panel retained 1/16, 0/16 and 3/16 respectively. All 44 ordinary case errors on that panel were in forward table order. Of the 16 case units, four missed one decision and six missed two. The 23/48, 28/48 and 47/48 counts in Section 4 describe an earlier checkpoint. Each ordinary description/case unit requires 32 native answers and a rule unit 16, because the compact rule has only one surface order. [Previous panel](../two_flags_20261104/RESULTS.md)

Three continuations started from the identical Section 5 component-reader weights. They shared 400 updates, 12,800 labels, 3,200 training forwards and 1,886,052 padded training tokens. Each five-update cycle contained two description, two case and one rule block. The first two arms used 3,200 labels per component. The focus arm used 5,600 reported-bit actions, 4,000 conjunctions, 2,400 ordinary actions and 800 individual flags. Unmasked token counts were 1,465,344, 1,481,424 and 1,571,182 for repeat, vary and focus. Snapshot replay succeeded for all three readers before any fresh scoring. One initialization, eight known menus and one new primary grammar bound these comparisons. [Fixed comparison](PLAN.md) · [Endpoint hashes](literal/fit_closed.json)

| New grammar, names and question | Ordinary choices, three stages combined | Ordinary strict units, description / cases / rule | Composed strict units, same order |
| --- | ---: | ---: | ---: |
| Unchanged parent | 1,149/1,280 | 2 / 5 / 2 | 2 / 2 / 0 |
| Original grammar, equal tasks | 1,169/1,280 | 2 / 8 / 0 | 3 / 6 / 0 |
| Added grammar, equal tasks | 1,187/1,280 | 2 / 7 / 0 | 4 / 6 / 0 |
| Added grammar, more binding and conjunction | 1,182/1,280 | 3 / 9 / 0 | 4 / 6 / 0 |

Each strict count is out of 16. With equal task proportions the added grammars recovered 37 ordinary answers and lost 19. At fixed varied grammar, reweighting recovered 16 and lost 21. The focus arm was fixed in advance. On the original two fitting styles it attained 44/48, 45/48 and 48/48 strict units. On the complete four-style fitting bank it attained 44/96, 47/96 and 91/96. The larger training bank includes styles in which action precedes the condition. The original-bank gain did not meet the expanded fitting gate.

| Focus arm, fresh joint panel | Description | Exhaustive cases | Reintegrated rule |
| --- | ---: | ---: | ---: |
| Ordinary native / strict | 471/512 · 3/16 | 502/512 · 9/16 | 209/256 · 0/16 |
| Two-call native / strict | 958/1,024 · 4/16 | 986/1,024 · 6/16 | 238/512 · 0/16 |
| Original and complemented-bit choices correct | 840/1,024 | 872/1,024 | 32/512 |
| Original and opposed-policy choices correct | 892/1,024 | 948/1,024 | 80/512 |

The new conjunction was correct in both label orders (8/8 each) for the unchanged parent and every trained arm. On the previously exposed questions reweighting gave 7/8 in both orders. Thus the fresh 16/16 cannot be attributed to added conjunction practice. On the new joint panel every first bit was correct, so its composed action errors all remained after the world test. No arm passed the Section 5 constructive criterion. The rule's high fitting score was a property of specific renderings.

With only names changed, focus retained 14/16 description, 16/16 case and 16/16 rule ordinary strict units. With only grammar changed those counts fell to 1/16, 10/16 and 1/16. The matched familiar counts were 15/16, 15/16 and 16/16. In the new joint condition all ten residual ordinary case errors were in forward order. [Complete preserved endpoints](literal-summary.json) · [Miss counts per strict unit and source order](diagnostic.json)

## A frozen cross of grammar, bit wording and question

After observing the failed primary endpoint we fixed the focus reader and the two new lamp pairs and crossed three wording factors without training. Supplying the known Boolean directly isolates the second call. In the table below the decision question and its topic predicate are familiar in every cell. Counts weight the two bit values equally, unlike the world-weighted 512 rule paths above.

| Compact rule grammar | Familiar bit report | New bit report |
| --- | ---: | ---: |
| Familiar if/else rule | 127/128 · 15/16 strict | 126/128 · 14/16 strict |
| New notebook rule | 88/128 · 2/16 strict | 60/128 · 0/16 strict |

The rule bank has 32 sources, two bit values and two displayed label orders. With the new rule and the new bit report, 60/64 false-bit choices and **0/64 true-bit choices** were correct under the familiar question. Changing the question as well shifted those counts to 30/64 and 27/64. Thus a literal reader reliable on one rule frame can largely lose its switched action in a different wording combination. Restoring the familiar bit report helped the new rule by 28 choices but the familiar rule by one. The design supports this contrast for the sampled texts. It does not isolate a single token or grammatical feature. [Sealed wording audit](WORDING_AUDIT.md) · [All eight crossed cells](wording/cross.json)

A numerical check explains the small difference between the exposed parent rescoring and the archived Section 5 answers. All 1,280 ordinary inputs and raw scores replayed exactly with the original four-example batches. At current eight-example batching, full versus position-selected logits agreed on all stored A/B values and winners; both also reproduced the current results. Between batch geometries eight winners changed on the identical inputs. All three continuation arms used identical evaluation grouping within the current study. The crossed wording audit batches only branch prompts, so its raw counts have their own recorded geometry and denominator. [Raw batch audit](wording/batch.json)

## Expansion and reintegration on a frozen native reader

We captured question-free eight-slot description, case and rule states and fitted two real maps. The first predicted a forward exhaustive case representation from each of 384 descriptions. The second targeted compact rules using 384 actual first-map outputs and 384 independently encoded case teachers. The feature fit used two rank-32 reduced-rank ridge maps. Its 16-policy decoder was fitted separately on 192 genuine training rule states. No original source, world, question, correct policy or teacher state was passed from the first map to the next in the test chain. [Two-map design](CHAIN_PLAN.md)

A separately sealed fixed continuation added rank-eight residuals to both maps. Only the residuals' 589,824 parameters received updates, using 192 training steps with the native action losses at the actual intermediate and final outputs and an auxiliary reconstruction loss. The native reader and Gemma backbone were frozen. In every step the downstream native loss had a nonzero measured gradient at the G1 output, and both fitted residuals received gradients. This is a development learning experiment. The original qualification panel was never opened. [Native-gradient design](NATIVE_GRADIENT_PLAN.md) · [Gradient run binding](guided/guide_closed.json)

| Native eight-slot route | Feature fit | Reader-gradient continuation |
| --- | ---: | ---: |
| Independently encoded description / cases / rule | 256 / 256 / 256 | 256 / 256 / 256 |
| G1 only | 257 | 255 |
| G2 only, from the original state | 254 | 253 |
| G2 from the actual G1 output | 255 | 252 |
| Correct original and opposed-midpoint reversal | 0 | 0 |
| Zero / fitting-mean rule slots | 256 / 256 | 256 / 256 |

Every count above is out of 512. Each complete two-map route attained 0/16 strict units. The 64 held-out input descriptions represent 16 policies on eight topics, with four worlds and both later label orders. For the independent rule teacher the two description orders refer to the same compact rule. The frozen native recipient could not use even the supplied rule teacher above its balanced nulls. This is an observed bottleneck before attribution to G.

The G1 output was closer to its correct case teacher than the opposed teacher on 31/64 descriptions after feature fitting and 32/64 after native-guided continuation. The final rule preference counts were 33/64 and 31/64. Neither operation showed the required omission advantage together with the correctness thresholds. Through a separate discrete decoder and a fixed canonical rule renderer the complete chains scored 264/512 and 251/512. Giving this renderer the correct class instead produced 508/512. This last oracle supplies the correct policy explicitly. It diagnoses the final literal reader under the canonical wording. It does not validate a state-only native readout. [Feature-only native and decoded routes](maps-summary.json) · [Native-gradient endpoints](guided-summary.json)

Both raw gates failed at the fixed endpoints. The experiments establish an implemented and differentiated two-stage path, a measurable gap between rule wording and its slot representation, and a separate supervision-dependent synthetic role-memory result in the preceding study. An independently qualified source-state reader and the language-insertion comparison remain open. [General continuity protocol](../../docs/FUNCTIONAL_CONTINUITY.md)

A fresh process replayed the saved tensors and trained reader for eight fixed sources spanning every menu, eight routes and both readout modes. All 1,024 native winners and both A/B log probabilities matched per map endpoint. [Feature-fit replay](replay-maps/receipt.json) · [Guided replay](replay-guided/receipt.json)
