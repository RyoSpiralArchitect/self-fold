# Changing the condition after discarding the policy text

A walker takes the ridge by default and the river when both lamps are on. Suppose the condition changes to *either lamp*. If just one is on, the revised policy should send the walker to the river. The old policy and the revised policy agree when both are on. This distinction lets us ask what work the update did.

Gemma-2-2B-IT encoded each of two first-person clauses separately. Supervised heads bound the default and joint-trigger actions into a compact memory. A learned operation received only that memory, changed the condition, and produced the entire revised policy before a world was supplied. A second operating-system process made the native choices from a numeric packet, without access through its inputs to source text, source attention, gold choices or donor type. Its readout received an addressed soft row through an eight-position mixture of the 16 familiar action embeddings and the two displayed options. No intermediate argmax was decoded to text. Learned weights were saved before fresh encoding. [Prospective protocol and thresholds](PLAN.md) · [Exact execution source](https://github.com/RyoSpiralArchitect/self-fold/tree/effaaef64af873f8e32342049470558bfeb8c0a9)

| Result | New names, familiar wording | New names and new wording |
| --- | ---: | ---: |
| Complete four-row policy in the learned midpoint | 128/128 | 32/64 |
| Native action after updating | 3,072/3,072 | 1,140/1,536 |
| Original and opposed answers correct, A/B margin reversed | 3,072/3,072 | 744/1,536 |
| Original and equivalent donor both correct | 3,072/3,072 | 1,104/1,536 |
| Uniform memory, original gold | 1,536/3,072 | 768/1,536 |
| Correct one-hot original policy passed through update and reader | 3,072/3,072 | 1,536/1,536 |
| Strict topic/name/style units | 32/32 | 3/16 |

For each paired exchange we retain the identical updated operation, world, displayed alternatives and A/B order. A strict unit contains 96 decisions covering both original policies, both source orders, all four worlds, all three operations and both label mappings. There are 16 underlying policy bindings across eight known topics, repeatedly rendered and scored. Exchanging a pair in the other direction is another logical observation of those same policies. These counts describe one initialization and one fixed bank.

| Trigger requested | Familiar wording, native | New wording, native | New wording, paired opposition |
| --- | ---: | ---: | ---: |
| Both on | 1,024/1,024 | 374/512 | 236/512 |
| Either on | 1,024/1,024 | 386/512 | 260/512 |
| Exactly one on | 1,024/1,024 | 380/512 | 248/512 |

**Selective intervention.** On familiar wording, replacing only the joint branch from the opposed donor changed the choice correctly on all 1,536 affected judgments and preserved it on all 1,536 others. Replacing only the default branch gave the same counts. On new wording, counts for the joint replacement were 420/768 affected and 552/768 unaffected. For the default replacement they were 336/768 and 588/768. Each count requires the original answer also to be correct.

**Omitting the update.** On familiar wording, the 1,280 judgments for which the new condition changed the truth were all correct after updating and all wrong when the state bypassed the update. On the 1,792 judgments whose truth was unchanged, both paths were completely correct. With new wording the corresponding changed-world counts were 484/640 and 172/640. On unchanged worlds they were 656/896 and 660/896. The changed-world gain met its cutoff in both panels. The familiar-wording panel passed all seven registered criteria. The new-wording panel passed only that gain criterion.

**Where the new wording failed.** The new phrases distinguished a pair of lamps that “reaches full illumination” from one that “falls short of full illumination.” Of the 32 incomplete midpoints, 20 assigned both branches the same action on the correct menu. Twelve assigned at least one branch an action outside that menu, including two that also collapsed both branches. The correct one-hot policy produced all 1,536 native answers through the identical learned update and reader. Source encoding is therefore one observed point of failure. A soft incorrect memory can still yield a correct binary choice. This diagnostic uses argmax only to inspect the saved memory after inference.

**Training and independent reconstruction.** The compiler used 288 individual labelled clauses and formed all 288 fitting policy tables correctly. Its two learned role rows have a prescribed layout (three copies of default and one joint). The learned 4-by-4 update for each operation was supervised on these tables. The reader began from the preceding study's component adapter and saw all sixteen canonical actions and all eight menus during calibration. The one-hot embedding path reproduced ordinary literal-token logits exactly on the registered fitting check. The saved reader replayed its native fitting logits exactly. The producer and consumer both verified the same frozen backbone hash. The full eight-route study comprised 36,864 logical requests, 334 distinct input memories and 1,740 distinct native inputs. They took 109 batched decision forwards. The producer used 131 instrumented forwards including calibration and capture. The native consumer was rerun in a fresh process for each immutable packet. Choices, both logits, input fingerprints and dispatch indices matched byte-for-byte. [Replay hashes and terminal records](native_replay.json) [Fitting receipt](run/produce/fit_closed.json) · [Packet bindings](run/produce/packet/binding.json) · [Native choices and A/B scores](run/decide/answers.npz) · [Complete scores and gates](summary.json)

The result establishes a supervised, externally structured policy update without the original text on the familiar grammar. The bank fixes the actions, delineates the two clauses, assigns the ordinary four-world layout and supplies the binary world address. The operation family and the lexical readout were trained in advance. Freely learned eight-slot recurrence, reintegration into a compact rule, autonomous self-improvement, the effect of first-person framing and the language insertion remain separate experiments. [General protocol and scaling comparisons](../../docs/FUNCTIONAL_CONTINUITY.md) · [Reproduction](../../docs/REPRODUCING.md)

## Can the same update use its own output?

We fixed a second, zero-training diagnostic after inspecting the single-edit results and the routing weights. It uses the same 128 primary descriptions, with each of the three trigger settings followed by each possible second setting. The second request replaces the trigger and retains the default and switched actions. In every admissible first table, off/off still identifies the default and some other world identifies the switched action, so enough information remains in a correct first table. The second update received that whole soft table in its original numerical form, with no first-operation label, policy text, or reset to the ordinary AND layout. [Follow-up seal](https://github.com/RyoSpiralArchitect/self-fold/tree/0405d2bce6bb9e60432ed7d1b1618558e4808bee) · [Fixed comparison](COMPOSITION_PLAN.md)

| Incoming trigger | Request both on | Request either on | Request exactly one on |
| --- | ---: | ---: | ---: |
| Both on | 1,024/1,024 | 1,024/1,024 | 1,024/1,024 |
| Either on | 256/1,024 | 768/1,024 | 512/1,024 |
| Exactly one on | 0/1,024 | 0/1,024 | 0/1,024 |

Counts are full-vocabulary native choices after the second operation. In each cell, correct original-plus-opposed reversals had the identical count. Each first-row cell passed 32/32 strict units. All six other cells had 0/32 strict units. All the changes across the first row were realized on the affected worlds, and a bypass missed them all. In the other four nonredundant cells, the requested edit and the bypass both solved zero affected worlds. Three of the nine cells passed, giving 4,608/9,216 correct overall. The full composition criterion failed. After exactly-one-on, the model still changed its letter on 3,072/3,072 opposed-donor exchanges. Both native choices in every pair were wrong for their respective policies. Every output was A or B, with zero A/B margin ties. There were 27,648 logical requests including opposed and bypass controls, 1,120 distinct native inputs and 70 batched decision forwards. The original compiler, update and reader were frozen.

The learned update for each operation gave the first three input rows exactly the same weight within each output row. Those three rows had always described the default during fitting. After either-on, two of the three carry the switched action. After exactly-one-on, two again carry the switched action and the last row carries the default. The first table is correct, but the learned read rule now misidentifies both branch roles. This localizes a failure of repeated use even on familiar wording. A future learner can retain explicit branch identities alongside the mutable case table or learn to recover the roles from its own edited outputs. A fresh comparison should remove the original policy from each step and distinguish the contribution of the added state from a supervised baseline at the same cost. [All nine cells and changed-world controls](chain-summary.json) · [Exact native outputs](chain/decide/answers.npz)

## Can the two actions be recovered from the same table?

After a paper-only review we sealed a diagnostic of the failed compositions. It used the first update's actual FP32 outputs, the same nine instruction sequences, questions, frozen update and native reader. The external layout rule took the off/off row as default and the row farthest from it in squared Euclidean distance as the switched action. It restored the three default rows and one switched row expected by the trained updater. It saw no flag reading, menu, prior operator, source, donor type or target. A second condition deliberately assigned the two recovered roles in reverse. An omitted edit always returned the original first table untouched. Both transformations had zero training updates. [Prospective recovery test](ROLE_RECOVERY_PLAN.md) · [Execution source](https://github.com/RyoSpiralArchitect/self-fold/tree/4b8b54bcf1970e11c7f95337d6fa3bd6702ba525)

| Second-edit input | Native correct across nine pairs | Correct original and opposed reversals | Passing cells |
| --- | ---: | ---: | ---: |
| Unmodified first table | 4,608/9,216 | 4,608/9,216 | 3/9 |
| Restore default and switched roles | 9,216/9,216 | 9,216/9,216 | 9/9 |
| Deliberately exchange the recovered roles | 0/9,216 | 0/9,216 | 0/9 |

Restoration raised the six cells starting with either-on or exactly-one-on from 1,536/6,144 to 6,144/6,144, a gain of 75 percentage points. It preserved all 3,072 correct choices starting with both-on. Every restored cell passed 32/32 strict units and the relevant changed-world comparison. Each new condition used 27,648 logical requests including opposed and bypass controls. Restoration needed 574 distinct native inputs and deliberate reversal needed 570. Each took 36 batched Gemma decision forwards. Both verified the unchanged backbone. [Complete criteria and raw counts](recovery-summary.json) · [Restored choices](recovery/restore_roles/decide/answers.npz) · [Wrong-role choices](recovery/exchange_roles/decide/answers.npz)

The original failure can therefore be reversed by a change in how the same numerical table is presented. All three conditions share the prior that off/off selects the default and a distinct second action appears elsewhere. The rule is hand-specified and uses that property. Learning a stable representation or recovering the roles from recurrent experience, including families without a common default address, remains the next training question.
