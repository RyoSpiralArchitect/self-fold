# Self Without a Transcript

## Reading, changing and reusing a conditional policy

**Research working paper · November 4, 2026**

### Abstract

Can a model change a policy after discarding the words that specified it? On eight familiar topics, a probe of frozen Gemma states recovered the order of verdicts on 2,043/2,048 rephrased records, but the complete preference on only 1,311/2,048. We then trained a compiler to store two conditional actions, a small map to revise the trigger and a native Gemma reader to choose an action. With familiar clause patterns and new signal names, all 3,072 choices were correct. Replacing the midpoint with the opposed policy reversed every choice. Bypassing the update missed all 1,280 judgments whose correct action changed. New clause wording reduced correct choices to 1,140/1,536. Applying the same updater twice on the familiar wording gave 4,608/9,216. Thus a learned policy memory supported a single source-free revision within its trained format. Rearranging the same tables by a fixed two-role rule restored 9,216/9,216. The actions survived even where the learned updater could no longer locate them.

## 1. A rule for choosing a route

A walker says, “I take the river path when signals Alder and Birch are both on. Otherwise I take the ridge path.” Tomorrow Alder is on and Birch is off. The walker should take the ridge path. Now suppose the original sentence is gone. Could a model work from its current representation, expand the rule into cases, turn those cases back into a rule, and still choose the ridge path when asked tomorrow? What would change if it had to put its intermediate state into words?

| Form of the same policy | What the model would need to preserve |
| --- | --- |
| Original description | Take the river path when both signals are on. In every other situation take the ridge path. |
| Exhaustive cases | Off/off, ridge. Off/on, ridge. On/off, ridge. On/on, river. |
| Reintegrated rule | Take the ridge path by default. Switch to the river path exactly when both signals are on. |

Writing out the cases above preserves the route in every world. The intervention we test in this paper changes the rule itself. If either lamp now suffices for the river, the walker switches routes on days with just one lamp on and keeps the old route when both are on. This gives us a positive check on the update and a check on what it must preserve. Exchanging river and ridge gives an opposed policy. A future experiment will ask whether the model itself can rewrite an equivalent rule and how putting its state into words affects its next decision.

We first ask what a question-free state preserves about a preference and whether exchanging such a state changes a native answer. We then follow the walker's policy through two interfaces. The first passes a conjunction bit and lets the second call reread the policy. The second passes a learned numeric policy and removes the original wording. The first-person frame supplies the policy. An effect specific to self-attribution would require matched third-person and impersonal descriptions.

## 2. What is read, and by whom?

**Reading a state.** An encoder reads a description before any evaluation question appears. We retain eight vectors at a specified layer and call this state $E(s)$. A source probe learns a score from these vectors to predict a feature of the description. A high probe score establishes that the fitted reader can recover that feature from one state. In a separate test, we inject a state into a task prompt and read the language model's own next token. We call this the *native answer*. An *opposed donor* supplies the state of the other policy while the question and its displayed options stay fixed.

**Reading a choice.** Every binary action question is presented twice, with the options exchanged. Native correctness requires the most likely token over the full vocabulary to be the requested letter. Other tokens count as errors. A strict unit in the conditional experiments fixes topic, flag-name pair, wording and representation stage. It passes only when both policies are correct in all four worlds, in every genuine source order and in both label orders. For example, a reader that always follows the last clause can score many individual decisions and still fail the entire unit. The original description and the table each have two orders. The reintegrated rule has one realization per wording family.

Earlier Mistral-7B-Instruct-v0.3 and Gemma-2-2B-IT probes explain the ordering of our tests. Both models answered 24/24 preference questions from literal descriptions. Twenty-four distinct updated slot tensors nevertheless produced one verbalization per checkpoint, and all 12 re-encoded donor contrasts tied. Replaying states from positions that had already processed both options could reproduce native lexical choices. A recurrence needs a state formed *before* the question. The original tests of uncalibrated transport, output positions and stopping conditions are indexed in the supplement. [Development experiments 1–8](DEVELOPMENT_RECORD.md#study-1)

## Related work

The Quine Test studies whether descriptions reproduce preference profiles in fresh instances [Babu et al. (2026)](https://arxiv.org/abs/2610.38726v1). Propositional probes distinguish facts and their relational bindings in context [Feng et al. (2024)](https://arxiv.org/abs/2406.19501v2). Our verdict probe asks which of two familiar judgment roles survives a change in the action words. The donor and branch exchanges address the separate question of causal use. Coconut trains reasoning through continuous hidden-state feedback [Hao et al. (2024)](https://arxiv.org/abs/2412.06769), and recurrent-depth models allocate repeated computation within a network [Geiping et al. (2025)](https://arxiv.org/abs/2502.05171). Natural Language Autoencoders provide a learned activation-to-language-to-activation interface [Fraser-Taliente et al. (2026)](https://transformer-circuits.pub/2026/nla/index.html). The planned language insertion compares later use on the two paths from the very same intermediate state.

## 3. A frozen state preserves the order of familiar verdicts

Consider a two-record description that accepts the river route and rejects the ridge route. Hold the two actions and their mention order fixed, then exchange the verdicts. The two descriptions contain the same tokens and the same slot positions, yet which action is accepted changes. A reader using only one question-free state can be tested on each member independently.

We froze Gemma-2-2B-IT and trained a linear score with a fitted offset on 4,608 such records. The readout predicted whether the first record accepted its action. A second readout predicted which action was mentioned first within an inferred topic. Combining the two predictions selected the preferred action. Evaluation used 64 unfamiliar action expressions across eight known topics, two familiar verdict vocabularies, both field placements, both action orders and both policies. The 2,048 states are repeated realizations of 16 policies within those topics.

| Question asked of one held-out state | Correct | Comparison or consequence |
| --- | ---: | --- |
| Is the first action accepted? | 2,043/2,048 | Shuffling the training role supervision gave 1,022/2,048. |
| Are both members of an opposed pair read correctly? | 1,019/1,024 pairs | Each member was read separately. |
| Which specific action is preferred? | 1,311/2,048 | Topic routing and action identity also have to be correct. |

The role result passed its registered criterion in all four combinations of verdict vocabulary and field placement. A fit without extra weight on opposed pairs was equally accurate on the role. Each description contains one acceptance and one rejection, so a cue in either record can reveal their order. The result establishes a recoverable *order of familiar verdicts* under action rephrasing. The complete-policy errors locate the next difficulty. In 527 of the 737 errors, topic and verdict order were correct but the reader identified the wrong first action. Another 206 involved an incorrect topic and four identified the right action but assigned it the wrong verdict within the correct topic. [Probe, shuffled control and error partition](../provenance/role_exchange_20261103/RESULTS.md)

## 4. When a source state changes a model answer

A separate intervention asks whether a source can drive a native choice. We first established that eight vectors derived from explicit option words could influence a frozen Gemma reader. We then fitted a map from a question-free source encoding into this working input space. At test time the map received one source state. The downstream question contained the displayed alternatives, and the model produced its ordinary next-token scores. Source and recipient were frozen during fitting of the map. [Word-derived input control](../provenance/slot_readout_20261103/RESULTS.md)

On the 32 fitting states the mapped sources gave 64/64 native choices. Opposed donors reduced the score against the original gold to 0/64 and reversed the A/B margin on all 64 judgments. The fitting mean and zero inputs each gave 32/64. Wording of the question transferred better than wording of the source. The same states gave 62/64 under a withheld question wording, whereas changed source wording yielded 57/128 and two new topics yielded 30/64. This establishes causal use of the supplied preference on its fitting bank. General conditional reading requires a further operation. The same stored policy now has to yield *different* actions depending on the later world. [Source exchange and transfer panels](../provenance/source_lexical_20261103/RESULTS.md)

We tested that operation on the eight known action topics by supplying each original description, case table and reintegrated rule independently. These were experimenter-authored teachers. None was generated by a learned update. Native choices from ordinary slots were 127/256, 128/256 and 64/128, respectively, against balanced zero-state counts of 128/256, 128/256 and 64/128. Literal text gave 219/256, 215/256 and 88/128. Because even the original text left errors, the next experiment trained a reader on text before admitting slot learning. [Independent teacher controls](../provenance/action_operation_20261103/RESULTS.md)

Literal training and one fixed continuation raised fitting accuracy to 3,751/3,840 (97.7%). The corresponding strict counts were 23/48 for the original description, 28/48 for the table and 47/48 for the rule. Only the rule cleared the fitting requirement of 95% native and 90% strict. New names, source wording and questions gave 923/1,280 native choices. Of 89 fitting errors, 85 returned B when A was correct. This discrepancy motivates the curriculum comparison. It keeps both label orders, separates changes in names, grammar and questions, and asks where the decision fails when its parts can be tested independently. [Literal curriculum](../provenance/conditional_curriculum_20261103/RESULTS.md) · [Margin continuation and label-order diagnostic](../provenance/conditional_margin_20261103/RESULTS.md)

## 5. From two flags to an action

We compared two curricula from the same saved literal Gemma reader. *Practise the final choice* (`direct`) trained on actions given the policy and two flag values. *Learn the parts of the choice* (`parts`) divided its targets equally among reading a flag, deciding whether both flags were on, choosing from a supplied condition bit and answering the full action question. Both trained rank-16 query/value adapters for 384 updates in the final four layers, with 12,288 labels and 1,747,152 padded tokens. We froze the endpoints before crossing familiar and new signal names, source grammars and question wordings. The primary comparison used the cell in which all three changed.

For the two-call route, the first call saw just the two readings and answered whether both were on. A fixed rendering of that Boolean and the original policy text went to the second call. The two calls had independently swapped A/B labels. We also complemented the bit at fixed policy, exchanged the policy at fixed bit and supplied the true bit. These interventions distinguish errors in reading the world from errors in applying the policy.

| Training | Direct action | Two calls, predicted bit | Both bit settings correct | Strict composed units |
| --- | ---: | ---: | ---: | ---: |
| Final choice | 1,165/1,280 | 1,932/2,560 | 1,718/2,560 | 0/48 |
| Learn parts | 1,159/1,280 | 2,234/2,560 | 2,049/2,560 | 11/48 |

Compared with final-choice training, component practice lost 6 of 1,280 direct answers on net and gained 302 of 2,560 composed paths. Those paths reuse 16 distinct conjunction judgments and 640 distinct second-call responses per arm. Supplying the true bit gave 2,370/2,560 for component practice and 2,328/2,560 for final-choice practice. The registered five-point direct-transfer advantage and the full absolute intervention criterion both failed. Most importantly for the next experiment, even a correct conjunction says nothing about which action the walker takes. The second call still had the original policy in front of it. [Crossed panels, label controls and criteria](../provenance/two_flags_20261104/RESULTS.md)

## 6. Changing the policy after its words are gone

The walker takes the ridge by default and the river when both lamps are on. Now change the trigger to either lamp. With just one on, the walker should switch from ridge to river. With both on, the route should stay the same. We use these changed and unchanged worlds to test whether a later computation actually revises a remembered policy.

A frozen Gemma encoded the two clauses separately, before any later world or choices appeared. Two supervised heads, trained on 288 labelled clauses, identified each clause's role and action among 16 familiar actions. The two resulting soft action distributions populated the ordinary case table. Off/off, off/on and on/off received the default distribution, and on/on received the switched distribution. This initial layout was prescribed. For each requested trigger, a four-by-four map learned to revise the whole table from 288 fitting policy realizations. Its targets were the correct action identities in all four worlds under that trigger. A world address entered only after the revised table was complete. The native Gemma reader received that row as a continuous mixture of action embeddings, together with the two displayed alternatives. It received no original words or source attention.

```mermaid
flowchart LR
    A["Learned · Bind two clauses to two actions"] --> B["Given · Place the actions in four world rows"]
    B --> C["Learned · Revise all rows with the requested map"]
    C --> D["Given · Address one row with the later world"]
    D --> E["Learned · Read row and answer A/B"]
```

*Figure 1. Only the current table and the requested operation enter the revision. The world and displayed alternatives arrive later.*

Both panels used new lamp names, both clause orders and both A/B orders. The primary panel used two familiar clause patterns. The challenge used a new contrast between reaching full illumination and falling short of it. A correct opposed exchange required the original and donor choices to be correct with a reversed A/B margin.

| Outcome | Familiar wording | New wording |
| --- | ---: | ---: |
| Correct complete midpoint | 128/128 | 32/64 |
| Native action after one edit | 3,072/3,072 | 1,140/1,536 |
| Correct original and opposed reversal | 3,072/3,072 | 744/1,536 |
| All 96 judgments in a strict unit | 32/32 | 3/16 |

*Both panels repeat 16 policy bindings. The primary total consists of 32 topic/name/wording units of 96 judgments each. Each unit crosses both policies, both clause orders, four worlds, three operations and both label mappings. The new-wording panel has 16 such units.*

On familiar wording, exchanging only the switched-action row changed the correct choice on all 1,536 affected judgments and preserved it on all 1,536 unaffected judgments. Replacing only the default rows gave the same counts. Updating solved all 1,280 judgments whose correct choice changed, while skipping the update solved zero. On the other 1,792, both paths were completely correct. Uniform memory gave 1,536/3,072. All registered primary criteria passed.

On new wording only 32 of 64 midpoints recovered both branch actions. In 20 incomplete memories both branches named the same action on the correct menu. In another 12 at least one branch pointed outside the menu. The correct one-hot policy drove all 1,536 answers through the same update and reader. The wording challenge missed its gate. Across both panels and eight interventions, 36,864 requests shared 1,740 distinct native inputs. [Native outputs and intervention scores](../provenance/state_only_20261104/RESULTS.md)

We then froze a follow-up test on the already scored primary panel. A second edit replaced the first trigger while preserving the two branch actions. The updater received its own complete output and the new operation, without the first operation's name. When the first trigger was both-on, the second edit was correct on 3,072/3,072 judgments across the three requested triggers. After either-on the count was 1,536/3,072. After exactly-one-on it was 0/3,072. Opposed donors still flipped all 3,072 answers in that last condition, but both answers in each pair contradicted their policies. Only three of nine sequences met the reuse criterion. All first edits on these policies had been correct. [Nine sequences and frozen follow-up design](../provenance/state_only_20261104/COMPOSITION_PLAN.md)

The fitted map gave the first three input rows equal weights when reading the default. All three had held the default during fitting. After an exactly-one edit, two held the switched action, and the fourth row held the default. We tested whether those roles remained recoverable in the same intermediate tables.

In a separately sealed diagnostic, a fixed rule took the off/off row as the default and selected the most different of the other three soft rows as the switched action. It restored the original layout and passed it to the same learned update and native reader. Only the incoming table was consulted. Across the nine sequences, accuracy rose from 4,608/9,216 to 9,216/9,216. Deliberately exchanging the two recovered roles gave 0/9,216. The two actions survived the first edit and were accessible under the known convention that off/off always selects the default. The trained updater required external help to recover them after its input layout changed. [Role-restoration test and controls](../provenance/state_only_20261104/RESULTS.md#can-the-two-actions-be-recovered-from-the-same-table)

## 7. From one revision to learning across revisions

The constructed memory establishes source-free conditional use on a fixed action vocabulary. It also exposes two distinct requirements for recurrence. A new source wording must bind the correct actions, and the updater must preserve a format that it can consume again. The single-bit reader in Section 5 had retained the literal policy. Here the next computation used the numeric table alone. The table format, world address and action vocabulary were supplied by design. Independent confirmation requires newly sealed policy families and additional seeds.

Our next test would train on whole sequences of revisions with exceptions, redundant edits and rule reintegration. Rollouts must include the learner's own intermediate errors. Branch a saved midpoint into an edit, a bypass and a donor exchange, and score them on independent changed and unchanged worlds. A reward for improvement alone can favor a poor first answer. Reward correctness before and after the edit, audit a separately held verifier, and measure loss of previously correct choices. These comparisons connect the observed failure of reuse with work on multi-turn self-correction and reward overoptimization. [Kumar et al.](https://arxiv.org/abs/2409.12917v2) · [Gao et al.](https://arxiv.org/abs/2210.10760)

For the original eight-slot hypothesis, first verify independently encoded descriptions, case tables and reintegrated rules. Then compare rewriting in words (`T`), continuing in slots (`L`), and verbalizing and re-encoding the same intermediate before continuing (`B`). The last two routes share their first updated state and their next updater. Record native end of sequence or censoring for each verbalization.

For a scaling claim, compare recurrent reinforcement learning with direct-answer and supervised recurrent baselines at equal training and inference compute, including verifier calls and verbalization. Vary depth, task diversity and model size separately. Hold out whole operation and action families. Test independent planning and language tasks, retention and variability across seeds. [Protocols and scaling comparisons](../docs/FUNCTIONAL_CONTINUITY.md)

## Supporting material

The [development record and experiment key](DEVELOPMENT_RECORD.md) names each earlier experiment and preserves its run identifiers, results tables and links to the raw records. The [reproduction instructions](../docs/REPRODUCING.md) give the execution seals. The [reference audit](REFERENCE_AUDIT.md) records source and version information.
