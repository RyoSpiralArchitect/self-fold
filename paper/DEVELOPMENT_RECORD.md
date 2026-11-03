# Development record and experiment key

The [main paper](main.md) follows the scientific argument. This supplement preserves the nineteen earlier results sections and their internal labels. The [reviewed edition](https://github.com/RyoSpiralArchitect/self-fold/blob/b96f7f3a76660230039ba2b2c422269df9ce75d9/paper/self-fold.pdf) preserves their original pagination, methods and bibliography. Legacy section references in archived plans and the text below refer to that edition.

## Paths, operators and training arms

| Reader-facing name | Internal ID | What is passed or changed |
| --- | --- | --- |
| Rewrite in words | `T` | The next step reads the model's new description. |
| Continue in slots | `L` | The next step receives the updated hidden state. |
| Say it, encode it, then continue | `B` | The model verbalizes the same updated state and encodes its words before continuing. |
| Encode the description | `E` | A source-only pass produces eight hidden vectors. |
| Say what the state carries | `D` | A decoder verbalizes the current slots. |
| Update the current state | `G` | An operation consumes the current slots. In the proposed learned composition, `G1` expands cases and `G2` reintegrates a rule. |
| Align output and input layers | `R` | Ridge transport candidate in the qualification plan. |
| Learn to store and recover a description | `C` | Codec and transport candidate in that plan. |
| Learn one state update | `U` | Operation-conditioned update candidate in that plan. |
| Ask the original slots | `raw` | Feed the frozen source representation to the task reader. |
| Normalize slots using training statistics | `standardized` | Use training means and scales before injection. |
| Fit the two displayed labels | `native_binary` | Train on the native A/B contrast. |
| Adapt both ends | `joint` | Fit source-encoding and task-reading adapters in separate scopes. |
| Teach the literal reader | `warm` | Train on the policy text before admitting slot learning. |
| Keep the source encoder fixed | `frozen` | Adapt only the recipient of the original source slots. |
| Practise the final choice | `direct` | Spend every answer in the closing study on an action. |
| Learn the parts of the choice | `parts` | Divide answers among flag value, joint test, branch-to-action and final choice. |

In task outputs, **A** and **B** label the two displayed alternatives, whose order is counterbalanced. The B answer is distinct from the language-roundtrip arm. The qualification plan also uses *positive completion* (route A) and *bounded diagnosis* (route B) as decision labels. Those labels name protocol outcomes.

## Experiment directory

| Scientific purpose | Legacy section | Record key and evidence |
| --- | ---: | --- |
| [Does waiting for the model to stop preserve a difference?](#study-1) | 6.1 | [mistral_eos_20261101](../provenance/mistral_eos_20261101/RESULTS.md) |
| [Ask the archived states which decision they favor](#study-2) | 6.2 | [slot_content_20261101](../provenance/slot_content_20261101/RESULTS.md) |
| [Test two descriptions of each of six opposed preferences](#study-3) | 6.3 | [slot_content_20261101](../provenance/slot_content_20261101/RESULTS.md) |
| [Separate the eight word positions from the later question positions](#study-4) | 6.4 | [slot_content_20261101](../provenance/slot_content_20261101/REPLAY_PLAN.md) |
| [Narrow the question region that carries a lexical choice](#study-5) | 6.5 | [downstream_span_20261101](../provenance/downstream_span_20261101/RESULTS.md) |
| [Exchange the period after the second displayed option](#study-6) | 6.6 | [downstream_span_20261101](../provenance/downstream_span_20261101/CLAUSE_PLAN.md) |
| [Capture a description before showing any question](#study-7) | 6.7 | [carriers_20261102](../provenance/carriers_20261102/RESULTS.md) |
| [Train the description copier and then ask it to choose](#study-8) | 6.8 | [carriers_20261102](../provenance/carriers_20261102/RESULTS.md) |
| [Distinguish first mention from the action selected by a condition](#study-9) | 6.9 | [policy_binding_20261103](../provenance/policy_binding_20261103/RESULTS.md) |
| [Test four readers of independently encoded cases and rules](#study-10) | 6.10 | [continuity_20261103](../provenance/continuity_20261103/RESULTS.md) |
| [Separate word-slot amplification, text copying and mixed-order policy use](#study-11) | 6.11 | [slot_readout_20261103](../provenance/slot_readout_20261103/RESULTS.md) |
| [Exchange the source while keeping the native reader fixed](#study-12) | 6.12 | [source_lexical_20261103](../provenance/source_lexical_20261103/RESULTS.md) |
| [Change source and question wording within eight topics](#study-13) | 6.13 | [within_topic_20261103](../provenance/within_topic_20261103/RESULTS.md) |
| [Separate the question reader from the source preference](#study-14) | 6.14 | [teacher_polarity_20261103](../provenance/teacher_polarity_20261103/RESULTS.md) |
| [Move the verdict beside the action and add a topic heading](#study-15) | 6.15 | [matched_cues_20261103](../provenance/matched_cues_20261103/RESULTS.md) |
| [Read the order of verdicts on unfamiliar action expressions](#study-16) | 6.16 | [role_exchange_20261103](../provenance/role_exchange_20261103/RESULTS.md) |
| [Swap actions, vary the question operation and test literal conditions](#study-17) | 6.17 | [action_operation_20261103](../provenance/action_operation_20261103/RESULTS.md) |
| [Teach the model to read complete conditional descriptions](#study-18) | 6.18 | [conditional_curriculum_20261103](../provenance/conditional_curriculum_20261103/RESULTS.md) |
| [Increase answer margins and count complete policies](#study-19) | 6.19 | [conditional_margin_20261103](../provenance/conditional_margin_20261103/RESULTS.md) |
| From two flags to an action | Matched-curriculum experiment | [two_flags_20261104](../provenance/two_flags_20261104/RESULTS.md) |
| Change the remembered condition after discarding the words | Current paper §6 | [state_only_20261104](../provenance/state_only_20261104/RESULTS.md) |
| Reuse the same updater on its own edited table | Frozen composition follow-up | [state_only_20261104/chain](../provenance/state_only_20261104/RESULTS.md#can-the-same-update-use-its-own-output) |
| Recover default and switched roles from the same table | Paper-only review follow-up | [state_only_20261104/recovery](../provenance/state_only_20261104/RESULTS.md#can-the-two-actions-be-recovered-from-the-same-table) |
| Learn the role bit with native-choice rewards | Current paper §7 | [role_memory_20261104](../provenance/role_memory_20261104/RESULTS.md) |
| Balance the parity of every four-edit history | Frozen exhaustive diagnostic | [role_memory_20261104/full](../provenance/role_memory_20261104/RESULTS.md#why-final-correctness-hid-errors) |
| Rebalance literal supervision and cross rule wording | Current paper §5.1 | [stage_grammar_20261104](../provenance/stage_grammar_20261104/RESULTS.md) |
| Expand cases and reintegrate a rule under a frozen native reader | Current paper §6.1 | [stage_grammar_20261104/guided](../provenance/stage_grammar_20261104/RESULTS.md#expansion-and-reintegration-on-a-frozen-native-reader) |

<a id="study-1"></a>

## Does waiting for the model to stop preserve a difference?

Original §6.1 · Native EOS and the language roundtrip


Two opposed descriptions emphasized revisable or prompt decisions. Each underwent two plain updates in T, L and B, once with native EOS and once with an eight-token ceiling. The four trials used greedy BF16 Mistral-7B-Instruct-v0.3, eight slots, a 1024-token context, a pre-layer-16 cut and a post-layer-31 writer (zero-indexed). All 18 generations per policy include main-path, observer and branch readouts. All 18 reached EOS with the full context available, and all 18 reached the explicit ceiling under the cap. With EOS, each of the four B updates verbalized the same 13-token sentence about workspace contents. The four B states matched within each policy, although the two policies gave different states. The four L states differed within each policy and matched step for step across policies. [Plan, raw states and output text](../provenance/mistral_eos_20261101/RESULTS.md)

<a id="study-2"></a>

## Ask the archived states which decision they favor

Original §6.2 · Reading the original state donors


For the previously studied Mistral pair, the literal revisability and speed descriptions yielded plus margins of +16.766 and -16.344 nats. The two initial slot states both gave +2.844. The first and second L updates gave distinct fingerprints but positive margins for both donors. B gave +2.813 for both donors at both steps. This particular question separated the two literal descriptions and favored revisability for every archived latent donor.

<a id="study-3"></a>

## Test two descriptions of each of six opposed preferences

Original §6.3 · Six contrasts and matched controls


On both checkpoints all 24 literal descriptions had the correct sign. At E and G, none of the 12 matched pairs had both signs correct on either model. The B margins tied in all 12 pairs on each model. Each checkpoint had 24 distinct E tensors and 24 distinct G tensors, but just one E readout string and one G readout string per checkpoint and stage. Re-encoding D(G) produced one B tensor on each checkpoint. All 50 free generations per checkpoint ended at native EOS. Gemma's G readout requested a description to summarize, as did its zero-state readout. Mistral's G readout referred to the contents of the work areas.

| Route | Mistral ordered/tied (of 12) | Gemma ordered/tied (of 12) | Mistral correct (of 24) | Gemma correct (of 24) |
|---|---|---|---|---|
| Literal | 12/0 | 12/0 | 24 | 24 |
| E | 6/2 | 5/1 | 12 | 12 |
| G | 3/0 | 7/1 | 12 | 11 |
| B | 0/12 | 0/12 | 12 | 12 |
| G, reversed | 5/2 | 5/0 | 12 | 13 |
| G, magnitude only | 9/0 | 3/2 | 12 | 12 |

The row-reversal and fixed-direction magnitude shams produced some positive rankings even when most donor signs were unchanged. The absolute magnitude of a state's response matters for interpreting these rankings. For ordinary G the median signed pole gap was -0.016 nats on Mistral and approximately zero on Gemma, compared with +29.680 and +20.766 for literal text. Per-topic opposite-pole and same-pole paraphrase differences, exact ties and both raw label orders are included in the supplement.

<a id="study-4"></a>

## Separate the eight word positions from the later question positions

Original §6.4 · Matched lexical donors and replay geometry


A lexical positive control placed the words of one answer option at the eight designated positions, padded on the right with the neutral period token. No answer letter was included. We compared native scoring, ordinary E on the same words, and injection of residuals captured at the same positions in a prefix ending at the last slot. Native words yielded 11/12 correct donor signs on Mistral and 12/12 on Gemma. Ordinary E gave 6/12 on each. The short-prefix aligned donor gave 6/12 and 7/12. Replaying neutral slots captured from a shorter forward changed individual label log probabilities by up to 0.1903 and 0.1613 nats. Those discrepancies motivated a separate length-matched experiment.

We appended one neutral dummy token to every scoring prompt and read the logit vector immediately before it. This reproduced all earlier native label scores exactly. The eight-slot donor now used a full-length forward, with neutral tokens replacing the question and alternatives after the slots. Neutral replay then matched every native vocabulary logit in all 12 question/order combinations on both checkpoints. The opaque-tail lexical donor's slots also matched its natural full-length counterpart exactly in all 24 donor/orders per checkpoint.

| Intervention | Mistral correct | Gemma correct | Mistral median gap (nats) | Gemma median gap (nats) |
|---|---|---|---|---|
| Native words | 11/12 | 12/12 | 28.219 | 19.672 |
| Eight slots | 6/12 | 7/12 | 0.008 | 0.219 |
| Downstream span | 11/12 | 12/12 | 27.594 | 19.734 |
| Both regions | 11/12 | 12/12 | 28.219 | 19.672 |

We next transplanted the natural lexical donor's middle-layer residuals from the positions after the slots through the scoring position. These downstream residuals had already processed the question and alternatives. Transplanting only this region into a neutral recipient reproduced the native preference sign in all 12 cases per checkpoint, including Mistral's error on the author-shelving donor. Its full logit vectors could still differ. Transplanting the slots and downstream region together recovered the entire native scoring-position logit vector exactly in 24/24 donor/orders per checkpoint. [Plans, raw scores, state files and all control results](../provenance/slot_content_20261101/RESULTS.md)

<a id="study-5"></a>

## Narrow the question region that carries a lexical choice

Original §6.5 · Localizing the downstream span


We split the interval from the end of the slots through the scoring position into the generic task instruction, topic question, labeled alternatives, closing text before scoring, and the scoring position. Native tokenizer offsets were checked against the exact recipient IDs. We transplanted each block alone into a neutral recipient, left it out of an otherwise complete downstream transplant, and replaced it with neutral residuals in a native donor. We also tested windows of equal width. The six original topics and six prospectively fixed new topics used the same lexical task, both label orders and both donor poles. Each checkpoint and panel provided 24 donor/order measurements per study. The two panels are reported separately.

For each topic, let the pole gap be the plus donor's order-averaged margin minus the minus donor's margin. The reported percentage is 100 times the sum of those six gaps under a given intervention divided by their sum under whole-downstream transfer. A negative value indicates a reversed aggregate direction. Only-window and leave-window-out values need not add to 100%.

Transplanting the labeled alternatives alone yielded 49.5% to 84.8% of the full-span contrast across the four checkpoint/panel combinations. Omitting them left 4.8% to 22.9%. On the new panel the last eight positions retained 90.8% on Mistral and 8.7% on Gemma. In the Mistral template the closing span is six tokens long, so the last eight include two option positions. In Gemma the closing span is ten tokens long, so the last eight contain no option token.

Percentage of the full downstream pole gap. Each window is transplanted alone except in the omission column.

| Panel | Model | Options | Omit options | Score | First 8 | Last 8 |
| --- | --- | --- | --- | --- | --- | --- |
| Original | Mistral | 70.4 | 20.4 | 20.7 | 0.0 | 90.6 |
| Original | Gemma | 84.8 | 4.8 | 1.6 | -0.1 | 5.7 |
| New | Mistral | 49.5 | 22.9 | 26.6 | 0.0 | 90.8 |
| New | Gemma | 79.3 | 8.4 | 0.6 | 0.2 | 8.7 |

<a id="study-6"></a>

## Exchange the period after the second displayed option

Original §6.6 · What reaches the period after the second alternative?


The original Mistral last-eight result motivated the option-clause follow-up, fixed before inspecting prospective-panel outcomes. Both clause-ending periods were separate from the phrases and had the same token ID within each model. On new topics the first period alone carried 9.2% and 11.9% of the full contrast on Mistral and Gemma, respectively. The second period carried 21.8% and 27.3%. Omitting this period retained 87.9% and 88.8%. Leaving one topic out gave ranges of 18.5% to 23.9% and 25.6% to 28.4% for the second-period-only measurement.

Percentage of the whole downstream pole gap at the second displayed option's period. Context columns transplant the single period into the real recipient. The omission column keeps all other downstream positions from the matched donor.

| Panel | Model | Matched | Hide stem | Hide phrases | Reverse | Omit period |
| --- | --- | --- | --- | --- | --- | --- |
| Original | Mistral | 37.8 | 44.5 | 0.7 | -32.9 | 76.6 |
| Original | Gemma | 38.8 | 31.5 | 0.2 | -33.3 | 82.2 |
| New | Mistral | 21.8 | 16.4 | 0.0 | -22.3 | 87.9 |
| New | Gemma | 27.3 | 20.7 | 0.0 | -26.4 | 88.8 |

We held the target's true wording and option order fixed. Masking only the donor's question stem with neutral tokens of the same length preserved a positive contrast at the second period. Masking the two option phrases while retaining their printed labels and separators reduced the absolute raw contrast below 0.7% of the full reference in all four cells. Reversing the donor's natural option assignment changed the aggregate sign in every cell, including after subtracting the source neutral background and adding the lexical difference to the target background. On the new panel, among individual orders where the donor-native and target-native preferred letters differed, the raw second-period transplant followed the donor letter in 19/22 Mistral and 19/24 Gemma cases. The first period sometimes shifted under order reversal and was excluded whenever its absolute position or token ID differed. The second period was compatible in all 24 donor/orders per cell.

All 192 donor/order records across the eight jobs passed exact neutral and combined slot-plus-suffix vocabulary-logit replay. Original-panel native and whole-suffix label scores reproduced the earlier experiment exactly. The computations used no free generation. [Frozen plans, raw likelihoods, all windows and integrity checks](../provenance/downstream_span_20261101/RESULTS.md)

<a id="study-7"></a>

## Capture a description before showing any question

Original §6.7 · Question-independent carriers before learning


We built all donors before constructing any question and reused each unchanged state for two question wordings and both option orders. The twelve exposed topics had two opposed descriptions and two phrasings each. Each checkpoint supplied 48 donors per route, eight rows and a fixed 256-position donor forward. Workspace captures the original E slots. Native tail captures the last eight source-token positions in a description-only prompt. Both use the earlier midpoint and eager BF16. No layer or row-count search followed the scores.

| Checkpoint | Carrier | Group accuracy | Best fixed null | Literal source |
| --- | --- | --- | --- | --- |
| Mistral | Workspace | 50.0% | 50.0% | 100.0% |
| Mistral | Native tail | 50.5% | 49.5% | 100.0% |
| Gemma | Workspace | 50.0% | 50.0% | 96.9% |
| Gemma | Native tail | 59.9% | 50.0% | 96.9% |

Accuracy averages within twelve groups and then over groups, with 192 dependent readouts per cell. None passed the predeclared 80% accuracy plus ten-point null advantage. All 192 free readouts reached EOS. Native-tail D produced 45 strings on Mistral and five on Gemma, while each workspace produced one. Diversity alone did not identify the source. Both native hook checks and all 96 full-vocabulary neutral replays were exact. An initial Gemma attempt captured 82 donors and scored no questions before stopping on a seven-token source. Its failure is retained. Both pole-zero photo phrasings were lengthened, and native shape validation was moved before model construction.

<a id="study-8"></a>

## Train the description copier and then ask it to choose

Original §6.8 · The first codec-only learning run


Under the plan's bounded negative closure, the next scope was a description encoder bridge, with G teachers deferred. Forty-eight synthetic policy groups in four construction families supplied 192 training descriptions; 24 groups in two other families supplied 96 dev descriptions. Another 24 groups in two further families were committed by hash and kept outside the training checkout. Qualification outcomes and independent human transition review remain pending. The first codec teacher copied the source description exactly. It did not use the proposed G targets or qualification content.

We retained workspace because neither Gemma carrier passed. The bridge z + (z A) B has rank 16 and 73,728 trainable parameters; B began at zero. AdamW made 64 updates with eight accumulated examples, seed 47, learning rate 0.0003 and no weight decay. Teacher forcing used the source text and native turn-ending token 107. The lowest dev NLL among steps 0, 16, 32, 48 and 64 selected step 64. Functional questions were introduced after this choice.

| Dev endpoint | Before | Selected |
| --- | --- | --- |
| Mean source NLL, nats/token | 4.309862 | 1.857181 |
| Functional group-macro accuracy | 49.9% | 50.0% |
| Distinct D strings / 48 | 1 | 4 |

NLL averages token loss within each description and then across 96 descriptions. Functional macro accuracy weights each of 24 meaning groups equally. Groups have different numbers of context cases. The pooled counts are 287/576 before, 288/576 after and 483/576 for literal text; literal macro accuracy is 87.9%. All 96 pre/post free readouts reached EOS. The four post-fit strings concerned signal policies, while dev references concerned visitor mode or request/override. These outputs have not received blind human semantic review.

Only bridge parameters received gradients, all 64 gradient norms were positive and finite, and full backbone parameter hashes matched before and after fitting. Active differentiable and inference NLLs agreed exactly at the checked initial and selected states. The serialized bridge is bound to its host, capture geometry, source and data.

A separately fixed post-hoc read-only replay reproduced all 480 per-description NLLs at five checkpoints, with a maximum mean discrepancy of 3.48e-8 nats. It performed no optimizer updates or qualification evaluation. At step 64, keeping each target fixed but supplying its opposed donor gave mean NLL 1.858054, versus 1.857181 with the matching donor and 4.327749 with zero rows. The matching donor had lower loss in 49/96 descriptions. The large loss reduction therefore coexisted with very little aggregate opposed-policy sensitivity and no functional accuracy improvement. This identifies a failure of the tested fit, without proving that content is absent from its states. [Contracts, failures, learned artifacts and raw replay](../provenance/carriers_20261102/RESULTS.md)

<a id="study-9"></a>

## Distinguish first mention from the action selected by a condition

Original §6.9 · Source-specific learning and the order of a policy


We started two new rank-16 Gemma encoder bridges at identical weights. Each saw both opposed donors per pair, 64 AdamW updates, 512 description forwards and 44,804 training tokens. Both computed whole-source NLL and a two-token cross entropy at the first divergence (copper versus silver) following an identical prefix. The latter term was weighted zero for copy and one for copy plus opposition. Step 64 was fixed in advance. Replacing the divergent token and its entire suffix with placeholders at identical length reproduced all 192 dev decision margins exactly.

Copy and copy plus opposition obtained 49.35% and 50.00% group-macro functional accuracy, respectively. Each recorded zero correct opposed switches in 288 world, wording and label-order comparisons, and zero in 48 shared-prefix lexical pairs. Literal text reached 87.89%. The functional difference was +0.65 percentage points (scope +0.00, priority +1.30). The paired 95% site-resampling interval was +0.26 to +1.04 points, conditional on these two templates. The prespecified practical gain was ten points.

In the unadapted slots, a train-only centroid direction correctly ordered 48/48 original dev donor pairs by first-mentioned tray. Its fixed threshold classified 48/96 single descriptions. This prompted a separately sealed geometry test. We reversed the order of the two policy clauses while preserving each truth table. A shortcut that knows the construction's trigger and treats the first-mentioned action as its answer went from 288/288 correct description-world cases to 144/288. Each new description was encoded alone and again with both tray adjectives replaced by the same single token, metal. Source-minus-neutral features and raw features used separate train-only centroids.

| Fixed readout, reordered clauses | Raw slots | Source minus neutral | Centered, shuffled labels |
| --- | ---: | ---: | ---: |
| First-mentioned tray | 48/96 | 93/96 | 21/96 |
| Tray prescribed at the trigger | 48/96 | 51/96 | 49/96 |
| Prescribed tray correct in both orders | 24/48 | 6/48 | 2/48 |

On crossed wording the raw threshold predicted copper for every source. Centering supplied single-source access to mention order. The centered trigger readout was correct on 45/48 descriptions with the trigger clause first and 6/48 with it second. The registered joint endpoint counted the policies for which both orders were correct. These two encodings require the original input text. Availability of such a reference during later latent updates remains open. [Comparison, follow-up and raw scores](../provenance/policy_binding_20261103/RESULTS.md)

<a id="study-10"></a>

## Test four readers of independently encoded cases and rules

Original §6.10 · Native recipient learning before the two-stage reconstruction


The integrated question next required a native recipient of a question-independent state. A new bank has 96 train and 24 development policy groups, crossing polarity, clause order and self/other/impersonal frames. Flag/action bindings and source construction grammars differ across splits. Boolean truth functions and teacher rendering grammars overlap. Three representations per variant supply a source, exhaustive cases and an equivalent reintegrated rule. Each is encoded independently. These teacher states are interface controls before learning G.

Four bounded Gemma configurations each completed 256 updates. The first two trained task-only rank-16 q/v factors in the final four layers, with raw or train-standardized workspace inputs and full-vocabulary label loss. A separately sealed conditional-label follow-up matched the raw source hashes, initialization, schedule and training compute. A final one-candidate construction jointly trained source-only factors in layers 9--12 and task factors in layers 22--25 at learning rate 0.0003. Its source forward retained eight gradient-carrying rows at pre-cut 13 and received no question.

| Native final-teacher readout | Correct / 4,608 | Joint correct / 1,152 |
| --- | ---: | ---: |
| Frozen E, raw recipient | 2,294 (49.78%) | 52 (4.51%) |
| Frozen E, standardized recipient | 2,308 (50.09%) | 17 (1.48%) |
| Frozen E, conditional-label loss | 2,304 (50.00%) | 0 (0.00%) |
| Joint source E and recipient | 2,336 (50.69%) | 57 (4.95%) |
| Literal base-model control | 2,886 (62.63%) | 335 (29.08%) |

The fixed entry screen used full-vocabulary native greedy answers, requiring 90% final-teacher accuracy, a ten-point advantage over the strongest empty/zero/random/unrelated null and 75% joint polarity/order correctness. Each configuration supplied 4,608 decisions per stage per route. Final-teacher order tags share a canonical text, so this conjunction duplicates that axis. Source- and case-stage panels retain actual order differences. No candidate passed.

A post-hoc zero-update audit observed the native hidden state and pre-/post-softcap head values on a fixed train/dev subset. Among 384 raw-recipient tasks, 174 acquired an A/B tie after distinct BF16 head values. Recomputing two head columns in FP32 removed ties but yielded 48.18% strict correctness for raw and 52.86% for standardized. No observed label had a zero stored-tanh derivative. The audit found output quantization without recovering a successful policy readout. These diagnostic computations left the native output path intact.

The joint construction received positive finite gradients in both modules at all updates, replayed its zero-encoder native captures, and preserved full backbone hashes. Its source/state readout remained at 50.00%, and its final-teacher result was 50.69%. Across these four configurations, 1,024 optimizer updates did not admit a native recipient under the fixed criterion. Actual G composition, operator omission and intermediate language insertion therefore remain prospective. The intended positive continuity contribution is unestablished. [Phase seals, all outcomes and numerical audit](../provenance/continuity_20261103/RESULTS.md)

A separately sealed small-set control then trained identical fresh recipients for 64 updates on 128 fixed training tasks. Literal source input improved from 86/128 to 112/128 native correct, while raw slots improved from 48/128 to 58/128. On 64 exposed dev tasks the corresponding final counts were 48 and 34. Final literal output had no forced A/B ties, versus 74/128 training ties for slots. Neither condition met the predeclared 95% training accuracy and 90% joint engineering screen. This descriptive four-group control directs further work to trainability and the source/recipient connection. It supplies no learned G or qualification result. [Paired trainability control](../provenance/continuity_20261103/RESULTS.md)

A separately fixed presentation experiment used the same four policy groups, worlds, labels and 64-update schedule. Both new recipients trained on four exhaustive case sentences in one order. We scored each reader on the disjunctive rule, the trained list and its sentence reversal. Two historical order tags represented identical texts within each rendering, leaving 64 distinct training decisions and 32 opposed-policy pairs. On the forward list, the case-trained literal reader obtained 58/64 correct versus 48/64 for the prior rule-trained reader. On the reversed list their scores were 47/64 and 52/64. Both obtained 13/32 pairs correct across both orders. The case-trained slot reader obtained 35/64 forward and 31/64 reversed, with zero pairs correct across both orders. Neither reached the small-set training screen. On the two previously exposed development groups, the case-trained literal reader scored 27/32 forward and 28/32 reversed. The literal case-list fit processed more tokens than the literal rule fit. This comparison estimates sensitivity to the whole rendering on exposed development material. [Matched presentations and per-decision scores](../provenance/policy_form_20261103/RESULTS.md)

<a id="study-11"></a>

## Separate word-slot amplification, text copying and mixed-order policy use

Original §6.11 · Bounded readout tests


A new bounded cycle trained four further interfaces for 64 updates each. Its lexical control captured the two option phrases per topic in the native eight-row positions, neutralized the trailing question, and matched the donor and recipient forward lengths. Every neutral replay matched the full native vocabulary logits, and opaque lexical rows equaled the corresponding natural lexical rows. A predeclared gain sweep from zero to sixteen reached at most 14/24 native decisions, with at most one of six strict pairs correct across both option orders. No gain passed the fixed screen.

A shared rank-16 residual transformation then learned from sixteen native A/B decisions in four topics while the recipient and base model stayed frozen. It reached 16/16 on those topics and 6/8 on two topics excluded from its fit, for 22/24 overall and 5/6 strict pairs across both orders. Passing opposing lexical rows through the same transformation yielded 2/24, and passing zero rows yielded 10/24. The pooled development screen passed, while the two-topic transfer screen failed at 6/8 decisions and 1/2 strict pairs. These exposed lexical phrases explicitly supplied an option's wording.

The D-only arm loaded and exactly re-encoded the original 24 E states. All 24 source strings were training targets for one rank-16 q/v adapter in layers 13 through 25. Its mean copy loss fell from 5.7373 to 0.3140, but every state decoded to the same training-bank sentence. Exact source reconstruction was 1/24, with zero complete opposed-description pairs. Matched, opposed and zero-state final losses averaged 0.313990, 0.315978 and 0.331762. Ten of 24 references favored the matched state over both controls. All free outputs reached EOS, and their literal downstream donor correctness remained at 12/24.

The order arm replaced the duplicated training presentation with forward and reversed case lists at the same 64-update budget. The fresh literal reader scored 57/64 and 56/64 in these orders and 56/64 and 57/64 in two new permutations. Strict opposed pairs correct in all four orders increased from 8/32 for the earlier forward-trained reader to 14/32. The two previously used dev groups changed from 8/16 to 9/16, and two additional exposed dev groups stayed at 8/16. The E-slot recipient retained zero all-four-order strict pairs in every panel. Neither recipient passed its registered training or permutation screen. These four fits ended at their caps with positive finite gradients and unchanged backbone hashes. [Plan, learned interfaces and complete readout records](../provenance/slot_readout_20261103/RESULTS.md)

<a id="study-12"></a>

## Exchange the source while keeping the native reader fixed

Original §6.12 · From E to the frozen lexical recipient


A subsequent experiment froze E, the lexical transformation and the native recipient. Its newly authored bank contained twelve topics, two poles, four source variants, two question wordings and both option assignments. Each source paired a preferred and a rejected alternative, and a second source reversed those exact clauses. Eight topics and one source style supplied 32 fitting states. The other source style and four topics stayed outside fitting. Photo organization and recipe weights were new development topic combinations, while shelf and directions were previously exposed.

The source-only interface first solved a centered dual-ridge regression from the 32 E tensors to question-independent lexical teachers captured at a fixed 256-token donor length. Its complete map included these stored training anchors. Only a rank-16 residual on the flattened eight rows then learned for 64 updates, with 589,824 trainable parameters. The loss combined conditional native A/B cross entropy, lexical-teacher error and a teacher-relative variance floor. Neither source identifiers nor questions entered the mapper call. The prior lexical endpoint replayed all 24 original decisions exactly before fitting.

The final map solved 64/64 fitting decisions and all eight strict units spanning both source orders, both poles and both option assignments. Opposed E inputs yielded 0/64 and reversed all 64 margin signs. The training-mean and zero E inputs each scored 32/64 with zero donor pole gap, while retaining topic-specific option biases. With the same states and a withheld question wording, native correctness was 62/64 and forced A/B correctness 64/64. Two native outputs fell outside A/B. Thus the fitted bank could control a frozen native output path without a source text in the task forward.

The source-generalization screens failed. Withheld source wording in the fitting topics scored 57/128, previously exposed dev topics 29/64, and new topics 30/64, each with zero complete cross-order strict units. The canonical lexical teacher scored 116/128 on withheld source wording and 56/64 on new topics, but its new-topic strict count was only 2/4. Both the source map and the frozen lexical interface therefore have generalization limits under this protocol. The residual used 256 training forwards; its gradients were finite and positive, base and lexical hashes stayed fixed, and both saved endpoints replayed their checked rows and logits exactly. One seed and stored training anchors leave interpolation as a live explanation. Full interface admission and learned G remain open. [Sealed design, raw decisions and source maps](../provenance/source_lexical_20261103/RESULTS.md)

<a id="study-13"></a>

## Change source and question wording within eight topics

Original §6.13 · Within-topic transfer across source and question wording


A separate prospective study fixed all eight Section 6.12 fitting topics and their sixteen canonical lexical teachers. It evaluated transfer within this finite policy vocabulary. New-topic extrapolation retained its earlier outcome. The one-style and four-style arms each fitted 512 distinct E states using identical action expressions, poles, clause orders and optimizer schedules. Four syntactic frames were balanced over the same pairs in the multi-style arm. Each assembled a shared contrast space of at most fifteen dimensions from training-only within-policy covariance. Soft mixture weights selected proportions of the sixteen lexical teachers, followed by a shared rank-four row residual. Each arm trained 18,688 parameters for 64 updates. Questions and style identifiers were absent from mapper inference.

The primary panel jointly crossed two fresh source styles, two fresh pairs of action expressions, both clause orders, both poles, two fresh questions and both option assignments. Its 512 repeated decisions belonged to 128 source states and eight topics. Four-style training scored 308/512 native correct versus 290/512 for one style. Of the paired decisions, 100 favored four-style training and 82 favored one style, an aggregate difference of 3.52 percentage points. It missed the ten-point effect criterion. Four-style strict correctness was 2/32, with 1/16 units in each fresh style. Each unit required both poles, both source orders and all question and label mappings. Fitting correctness was 1,962/2,048 but only 98/128 strict units, below the fixed fitting screen.

The four-style map assigned the correct topic on 128/128 fresh source states and the correct pole on 82/128. Under opposing donors, native correctness fell to 204/512 and 106/512 margin signs reversed. Mean, zero and norm-only inputs each gave 256/512. Literal input scored 496/512. Direct canonical teachers scored 424/512 and the fitted true-policy renderer scored 448/512. Thus the tested source map still confused preferences within topics, and the teacher references also missed the 90-percent native criterion on the new questions. Neither the primary gate nor the narrower syntax-only screen passed.

The older ridge anchor span had numerical rank 31. The standardized Section 6.12 withheld states had mean squared projection energy of 30.65 percent in this span, and their opposed-donor differences had 14.27 percent. This measured partial overlap leaves the causal explanation for an individual error open. A stopped technical attempt exposed a memory-layout change on loading the initial projection. The repair replayed the original arrays, storage order and source captures before any optimizer update. Both final fits and their initializations passed exact checked row and native-logit replay. Two-stage G remained outside the closed gate. [Registered screen, all decisions and donor geometry](../provenance/within_topic_20261103/RESULTS.md)

<a id="study-14"></a>

## Separate the question reader from the source preference

Original §6.14 · Separating teacher reliability from source polarity


We first supplied the correct one of sixteen lexical policies for the eight known topics. A shared rank-four renderer trained for 64 updates on all four exposed question families with the base model and lexical transformation frozen. It answered 128/128 fitting decisions correctly. On four new families it reached 113/128 native correct, compared with 101/128 for the archived renderer and 102/128 for direct canonical teachers. The two generic families scored 24/32 and 27/32. The two topic-specific families each scored 31/32. Only one of eight topics was correct across all four questions, both poles and both label orders. Opposed teachers flipped 49/64 margin signs. A common mean teacher gave 64/128 with no sign reversals. These results missed the registered teacher screen.

We then held the action expressions fixed and varied how a source expressed acceptance and rejection. Each sentence pair mentioned both actions using identical word multisets for opposite preferences. Training crossed four familiar grammatical frames with four familiar action pairs per topic, both poles and both clause orders. A question-free reader inferred the topic and used the signs of two fitted contrasts. The first identified the first-mentioned action within the inferred topic. The second identified whether the first clause accepted it. Their product selected the preferred action. A comparator read the preferred action with one fitted affine score per inferred topic. Both readers scored 512/512 on the fitting sources.

| Fresh source frame | First action | First role | Two-sign policy | Direct policy |
| --- | ---: | ---: | ---: | ---: |
| Priority cue before the action | 121/128 | 118/128 | 111/128 | 79/128 |
| Verdict after the action | 84/128 | 97/128 | 77/128 | 62/128 |

The two-sign reader recovered all 256 topics and 188/256 complete preferences across the two new frames, versus 141/256 preferences for the direct score. Both component signs were correct on 176 sources. On 12 more, two wrong component signs canceled. A unit requiring both poles in both clause orders passed in 25/64 cases for the two-sign reader and 4/64 for the direct score. Holding the action mention order fixed, the two-sign reader recovered both members of 72/128 opposed pairs. Holding the first role fixed gave 66/128. Neither fresh frame cleared the source screen. The native teacher and internal source-policy counts have different endpoints. The conditional end-to-end test stayed closed.

[Sealed protocol, raw native choices and source components](../provenance/teacher_polarity_20261103/RESULTS.md)

<a id="study-15"></a>

## Move the verdict beside the action and add a topic heading

Original §6.15 · Matched topic headings and verdict placement


We held the Section 6.14 lexical teacher and source reader fixed. Four new generic questions were asked bare, with the correct one-word topic heading, with the word General, or with a different topic heading. Substitution changed exactly one native token. Removing that token and the newline reproduced the bare prompt. Across both policies and both option orders in eight topics, the true teacher scored 101/128 bare, 99/128 with the correct topic and 96/128 under each control heading. The correct heading recovered two decisions and lost four against bare. Opposed teachers switched the native answer correctly on 35/64 named pairs and 37/64 bare pairs. The cue-benefit and teacher-reliability criteria both failed.

A separate source manipulation used two bracketed fields per record. Each field contained either a familiar action phrase or an explicit verdict such as “I accept this”. We exchanged the fields in both records, crossing eight topics, four action pairs, two verdict vocabularies, both poles and both record orders. All 256 swaps preserved the encoder token multiset, length and workspace positions. The source reader inferred every topic in both placements. The literal control read the original text with the same bare question and both label orders.

| Matched source measurement | Verdict before action | Verdict after action |
| --- | ---: | ---: |
| Both first-action and first-role signs correct | 76/256 | 75/256 |
| Complete policy from the signs | 134/256 | 130/256 |
| Correct policy from two wrong signs | 58/256 | 55/256 |
| Native choices from literal source | 358/512 | 452/512 |

Moving the verdict after the action recovered 109 literal decisions and lost 15. All eight topics favored this placement on the literal task. The prospective literal comparability guard failed. For the slot reader the net difference favored verdict first by four policies and by one state with both component signs correct. Neither met the ten-point criterion. A descriptive recount held action mention order fixed and exchanged which action was accepted. All 128 opposed E pairs in each placement had distinct tensor hashes, yet the source reader chose the same policy on 122/128 pairs in both placements. With the verdict last, literal answers were correct for both poles and both label assignments on 84 pairs. The slot reader kept the same policy on 82 of those 84. This isolates a failure of the tested readout. Whether another reader can recover the distinction from E remains open.

[Registered paired contrasts and descriptive exchange counts](../provenance/matched_cues_20261103/RESULTS.md)

<a id="study-16"></a>

## Read the order of verdicts on unfamiliar action expressions

Original §6.16 · Reading verdict order across unfamiliar action expressions


In the preceding record study, opposing E states differed but the frozen prose-trained reader usually chose the same policy. We fitted a new reader on 4,608 records in eight known topics. Each training pair held both action phrases and their mention order fixed while exchanging acceptance and rejection. A shared affine head read the first record's verdict. A second head read which action appeared first within the topic inferred by the frozen router. Their sign product selected the preferred action. The primary fit weighted pair differences four times as heavily as midpoints. A unit-weight fit was equivalent to ordinary singleton ridge. We saved and replayed both endpoints before encoding any fresh source.

Evaluation crossed four new plus and four new minus expressions per topic with both verdict vocabularies, both field placements, both preferred poles and both clause orders. Each prediction used one question-free E tensor. A complete paired decision required independent correct predictions for both opposing states. The first-role head was correct on 2,043/2,048 states and on both members of 1,019/1,024 fixed-mention-order pairs. Its continuous score ranked the accept-first state higher on all 1,024 pairs. Shuffled role supervision produced 1,022/2,048 correct singleton roles. Each of the four cells exceeded the sealed 95-percent singleton, 90-percent paired and twenty-point shuffled-control criteria. Each source contained one acceptance and one rejection, so either cue could indicate their order.

| Held-out cell | First verdict | Both verdicts in opposed pair | Complete policy |
| --- | ---: | ---: | ---: |
| Accept/reject before action | 511/512 | 255/256 | 328/512 |
| Accept/reject after action | 512/512 | 256/256 | 318/512 |
| First/last choice before action | 510/512 | 254/256 | 334/512 |
| First/last choice after action | 510/512 | 254/256 | 331/512 |

The complete-policy fit reached 4,607/4,608 on fitting states but only 1,311/2,048 on held-out wording. Of 1,024 fixed-action-order exchanges, 627 switched correctly, 245 switched in reverse, 35 gave the same policy of the correct topic and 117 included a topic error. The full source gate failed in all four cells. Of 737 singleton errors, 206 selected the wrong topic. With the correct topic, 527 had the correct verdict order and the wrong first action, and four had the opposite pattern. Supplying the correct topic lifted policy recovery to 1,443/2,048. An ordinary fit scored 1,313/2,048 policies and 625/1,024 correctly switched pairs. It matched the primary role counts and the extra weighting missed its registered advantage. Literal text gave 2,905/4,096 native choices over both label orders. The successful endpoint measures the order of familiar verdicts in the frozen source state. Transfer to new verdict language and native use of the accepted action remain open.

[Sealed role and policy criteria, readout weights and raw predictions](../provenance/role_exchange_20261103/RESULTS.md)

<a id="study-17"></a>

## Swap actions, vary the question operation and test literal conditions

Original §6.17 · Action swaps, question operations and conditional teacher states


We froze the successful verdict head and the previous topic router, and fitted new first-action directions on the same 4,608 training states. The pairs now held the verdict sequence fixed and exchanged action order. A second candidate penalized variation among paraphrases at fixed topic, first action, verdict order, vocabulary and placement. Both endpoints replayed all fitting scores before the 2,048 new states were encoded. The covariance-penalized candidate identified 1,449/2,048 first actions with gold topic supplied, compared with 1,450/2,048 for both the swap-only and previous action heads. Its strict count was 196/512 and both independently correct action-swap members occurred in 426/1,024 pairs. With the inferred topic it scored 1,438/2,048. The frozen verdict sign was correct on all 2,048 fresh states, and the router on 2,028. Neither action admission nor the additional covariance effect passed.

A separately sealed lexical-teacher panel kept the topic heading and options fixed while replacing a generic preference-selection predicate with a concrete task operation. Across four new question families, concrete wording scored 125/128 native choices and generic wording 98/128. The paired contrast recovered 28 and lost one, a net 21.09 percentage points. Every family improved and five of eight topics improved, meeting the registered operation-effect criterion. The reliability conjunction still failed. The four concrete family counts were 31/32, 32/32, 32/32 and 30/32, with six of eight completely correct topics and 60/64 opposed margin flips. A common mean teacher scored 64/128. The final family missed the 95-percent family threshold and strict topics missed seven of eight. Changed operation phrases also changed question length, so the contrast tests these declared whole predicates.

The lexical teachers encoded unconditional preferences. To test a native interface for genuine conditions, we independently authored source, exhaustive-case and rule descriptions for the same eight action topics. Two neutral flags defined four worlds. One policy chose the plus action exactly when both flags were on, and the other reversed the choice. Every saved teacher state faced all four worlds and both label orders. Source and case teachers had two order variants. The canonical rule had one unique realization. This yielded 80 independently encoded states and 640 judgments per route.

| Conditional teacher stage | Literal native | Raw E native | Zero native |
| --- | ---: | ---: | ---: |
| Source | 219/256 | 127/256 | 128/256 |
| Exhaustive cases | 215/256 | 128/256 | 128/256 |
| Reintegrated rule | 88/128 | 64/128 | 64/128 |

Every raw-E stage had zero strict topics. The native conditional interface missed its 90-percent accuracy, 75-percent strict-topic and ten-point null-advantage criteria. Literal readings also exposed difficulty with these conditions, particularly the reintegrated rule. The lexical operation benefit therefore did not establish a reader for the conditional state. The registered G prerequisite failed. The run closed after 7,617 forwards and 753,813 processed tokens, with the base, verdict head, router, lexical transformation and renderer fixed.

[Sealed component experiments, native worlds and source readouts](../provenance/action_operation_20261103/RESULTS.md)

<a id="study-18"></a>

## Teach the model to read complete conditional descriptions

Original §6.18 · A literal-first conditional curriculum


We next placed literal task learning before slot adaptation. The fixed action vocabulary retained all eight topics. Three flag-name pairs and two wording families supplied 480 fitting descriptions across source, exhaustive cases and reintegrated rule. A separate bank crossed two new flag-name pairs with a new wording family and question scaffold, yielding 160 fresh descriptions. Every description faced all four Boolean worlds and both label mappings. A strict unit fixed topic, flag pair, wording and stage and required both policies, every world, genuine source order and label mapping to be correct. Each stage had 48 fitting and 16 fresh strict units.

A task-only rank-16 q/v adapter in the final four layers received 384 fixed updates. The first 64 used cases, the next 64 used rules, and the last 256 cycled all three stages. Each update included two paired units and 32 native decisions. Full-vocabulary cross entropy weighted the both-flags-on world three times as heavily as each other world. Evaluation weighted all worlds equally. The final endpoint was fixed before scoring the fresh bank.

| Literal stage and split | Final native | Strict units | Forced A/B | A/B ties |
| --- | ---: | ---: | ---: | ---: |
| Fit source | 1,482/1,536 | 17/48 | 1,480/1,536 | 4 |
| Fit cases | 1,443/1,536 | 13/48 | 1,440/1,536 | 4 |
| Fit rule | 762/768 | 44/48 | 761/768 | 1 |
| Fresh source | 429/512 | 1/16 | 429/512 | 0 |
| Fresh cases | 485/512 | 4/16 | 485/512 | 1 |
| Fresh rule | 203/256 | 0/16 | 201/256 | 4 |

Initial fitting native counts were 1,271/1,536 for source, 1,277/1,536 for cases and 728/768 for rule. The total increased from 3,276/3,840 to 3,687/3,840 (96.0%). The fitting gate required at least 95-percent native accuracy and 90-percent strict units separately at every stage. Rule passed both conditions. Source failed strict units, and cases failed both conditions. The gate remained closed despite the aggregate native score. All final outputs were A or B. Forced A/B scores require a strict margin sign, so tied logits can differ from the native greedy token.

The post-endpoint fresh literal diagnostic scored 1,117/1,280 (87.3%). It did not select training settings or reopen the gate. The frozen-E and joint-E slot arms were unattempted, and G remained untrained. The frozen backbone hashes matched, and the saved task adapter reproduced the checked native logits exactly after reload. This bounded result localizes an incomplete conditional reader before the source-interface comparison. It preserves the literal-first anchor for further development without establishing latent continuity or recurrence.

[Sealed curriculum, fixed endpoint and complete native decisions](../provenance/conditional_curriculum_20261103/RESULTS.md)

<a id="study-19"></a>

## Increase answer margins and count complete policies

Original §6.19 · A bounded margin continuation


The first endpoint's fitting errors occupied 31/48 source units and 35/48 case units, with 141 source and 170 case decisions below a gold-minus-foil margin of two. A separately sealed continuation loaded exactly that saved task adapter and retained every fitting source and question byte. It reset AdamW and added 384 updates, cycling 128 per stage. The previous world-weighted full-vocabulary loss was supplemented by an equally world-weighted A/B hinge with margin two and coefficient one. These changes jointly altered the objective, optimizer state and training budget. This design does not isolate a causal effect of the hinge.

The initial full fitting panel reproduced the parent endpoint. The final aggregate rose to 3,751/3,840 (97.7%), and each stage exceeded the native fitting threshold. Strict source and case units remained below 90%. Rule passed with 767/768 native decisions and 47/48 strict units. The literal conjunction again failed, leaving both slot arms unattempted and G untrained.

| Literal stage and split | Final native | Strict units | Forced A/B | A/B coverage |
| --- | ---: | ---: | ---: | ---: |
| Fit source | 1,488/1,536 | 23/48 | 1,487/1,536 | 1,536/1,536 |
| Fit cases | 1,496/1,536 | 28/48 | 1,495/1,536 | 1,536/1,536 |
| Fit rule | 767/768 | 47/48 | 767/768 | 768/768 |
| Fresh source | 309/512 | 0/16 | 307/512 | 512/512 |
| Fresh cases | 448/512 | 1/16 | 459/512 | 496/512 |
| Fresh rule | 166/256 | 0/16 | 167/256 | 255/256 |

The new fresh bank replaced the exposed names, source family and question operations before continuation. It scored 923/1,280 (72.1%) native decisions. Its different material prevents a transfer-effect comparison with the earlier fresh panel. Seventeen fresh outputs fell outside A/B, retaining their denominators. Of the 89 remaining fitting errors, 85 selected B when A was gold. This descriptive asymmetry leaves label mapping and branch order as explicit diagnostic factors. Fitting A/B ties numbered one, two and zero for source, cases and rule, respectively; fresh ties numbered four, one and zero. Saving and reloading the endpoint preserved the checked native logits, and the frozen backbone hashes matched.

An exploratory [saved-logit diagnostic](../provenance/conditional_margin_20261103/diagnostic.json) combined the swapped label assignments as $\tilde m=(m_0-m_1)/2$, where $m_\ell$ is the A-minus-B logit difference under label order $\ell$. Each external forced decision uses two saved native forwards, halving the decision denominator. Source, cases and rule scored 767/768, 757/768 and 384/384 on fitting decisions, with 47/48, 38/48 and 48/48 strict units. Fresh counts were 157/256, 241/256 and 92/128, with 0/16, 6/16 and 0/16 strict units. The fitting recovery is consistent with label-calibration error contributing to native failures. Fresh source and rule transfer remained weak. This post-hoc reader used no new model calls and changed no native gate, training objective or selected endpoint.

The rule fitting result establishes a learned literal endpoint within the closed fitting bank. The complete literal criterion remained unmet, and the new fresh diagnostic showed incomplete transfer. Both bounded studies are closed with their original thresholds and failed gates retained. No further loss or budget search belongs to these studies, and independent qualification remains unopened.

[Sealed margin recipe, parent binding and complete native decisions](../provenance/conditional_margin_20261103/RESULTS.md)
