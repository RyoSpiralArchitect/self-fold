# Self Without a Transcript
## Testing Functional Continuity under Semantic Recurrence

**Research working paper · November 3, 2026**

### Abstract

Can a distinction formed by reinterpreting a current self-description support another update and a later judgment without a transcript? We study that intended continuity and the effect of verbalizing an intermediate state. Earlier Mistral and Gemma development probes found question-conditioned lexical transfer but weak question-independent carriers. On Gemma, source-copy learning reduced text loss without reliable policy switching. A neutral-reference readout exposed mention order in 93/96 reordered descriptions, while the prescribed action was correct in both orders for 6/48 policies. We next trained four native-recipient configurations on a new formal policy bank, including a source encoder learned jointly with its recipient. Their independently encoded final-teacher accuracy ranged from 49.78% to 50.69%, below the fixed admission screen. A zero-update audit found additional BF16 output ties, while FP32 diagnostics remained near chance. These results identify unsuccessful tested interfaces within a unified two-stage research design. Actual learned G composition, its causal use of intermediate distinctions and the intended positive continuity result remain open.

## 1. Problem formulation

When the meaning of a current self-description is interpreted again and folded into a new current state, can the resulting distinction support another meaning update and a judgment in a new situation without consulting the original description or a transcript? How does verbalizing an intermediate state change that continuity? We use functional continuity as an operational question about specified distinctions and tasks.

A description such as “I prioritize keeping decisions revisable” may support an interpretation about preserving correction opportunities. A successor can make a condition explicit, preserve an already useful distinction, or introduce an unsupported claim. The target process takes the meaning of the current description as its input. The experimenter's transcript stays outside the model, while the current state can carry the causal influence of previous updates.

The first constructive target is a finite reconstruction of conditional policies. One operation makes exhaustive cases explicit and another reintegrates those cases as a current rule. The resulting state should guide a later judgment when the original source is unavailable. Content preservation is a condition for this two-stage process. Intermediate-state exchanges, operator omissions and a shared-state language insertion examine which distinctions the later computation uses.

These controlled transformations preserve prescribed actions, so their final correctness criterion alone can also be met by identity. The proposed experiment therefore checks the supported intermediate reconstructions and the effect of omitting each operation. Already explicit descriptions provide justified no-change cases. Open-ended interpretations that introduce a broader decision criterion require independent human support judgments and can have several acceptable successors.

The existing development studies identify failures that this constructive target must overcome. Distinct tensors, lower source-text loss and accessible mention order each leave the tested condition–action use unresolved. Their role is to locate the first failing connection under a particular interface. “Self” denotes an experimental attribution frame. Matched other-agent and impersonal descriptions test whether that attribution changes the result.

## 2. Related work and positioning

Coconut trains continuous-state reasoning through hidden-state feedback [hao2024coconut]. Recurrent-depth models study test-time computation through repeated internal updates [geiping2025huginn]. STARS motivates distinguishing useful stable dynamics from performance collapse in its studied looped models [yang2026stars]. These works do not establish that an arbitrary untrained residual feedback route preserves the meanings of self-descriptions.

The Quine Test asks whether model-written self-descriptions reproduce preference profiles in other models or fresh instances, and reports successful transfer of preferences assigned by an explicit policy [babu2026quine]. Our proposed target follows successive supported reconstructions and asks whether an intermediate state remains usable by another meaning update and a later decision.

Persona vectors and the Assistant Axis connect persona-related behavior with activation structure and interventions [chen2025persona; lu2026axis]. Natural Language Autoencoders learn an activation-to-language-to-activation route and edit verbal explanations to construct steering interventions that change subsequent output [frasertaliente2026nla]. SELF-FOLD's target extends to reuse of a reconstructed current state by a later meaning update and to a language insertion at the same intermediate state.

Propositional probes recover lexical information together with relational bindings, including settings where the model's output is unfaithful to its encoded context [feng2024propositional]. This motivates testing a distinction through later causal use as well as decodability. A latent chain-of-thought study also distinguishes output commitment from representational commitment [li2026dynamics].

Semantic diversity and superficial wording variation are not equivalent [tao2026divergent]. Findings about intrinsic self-correction and key-condition verification motivate operation-specific rather than generic self-improvement hypotheses [huang2024selfcorrect; wu2024verification]. Dadfar's vocabulary/activation correspondence report is considered alongside McCardle's open replication critique [dadfar2026examine; mccardle2026replication]. The latter reports length, mode and cycle-related confounds. Neither report is treated as proof of an internal experience. SlimMoE informs the proposed small Phi MoE comparison family [li2025slimmoe].

This draft does not claim priority or provide a systematic literature review. Bibliographic status and source caveats are recorded in `REFERENCE_AUDIT.md`.

## 3. Conditions and recurrence

Let c be a fixed operation instruction, s a current textual description, and Z a fixed K-by-d state. E encodes a description, D verbalizes a state, and G updates a state:

\[
\begin{aligned}
\text{T: } s_{n+1}&=T_\theta(c_n,s_n),\\
\text{L: } Z_{n+1}&=G_\theta(c_n,Z_n),\\
\text{B: } Z_{n+1}&=E_\theta(D_\theta(G_\theta(c_n,Z_n))).
\end{aligned}
\]

L and B start from copies of the same E(s0). They share model weights, boundary definitions, transport, generation policy, and operator schedule. L/B is the primary contrast. T/L compares systems with different representations and computation and is not an isolated linguistic intervention.

Every outer step starts from a fresh context with no persistent KV cache. Only the current text or slots carry forward. Readouts operate on copies, restore random-number states, and do not feed their text back except where explicitly required by B. The initial s0 is not repeated after initialization. Current implementation uses fixed numbers of slots at explicit decoder boundaries and an explicit post-writer to pre-cut transport. An identity transport is an engineering diagnostic only.

Two schedules are planned: an ordinary meaning-update instruction and a fixed cycle of decomposition, boundary examination, and reintegration into a current description. Unchanged content is allowed when no justified update is available. A future history-triggered intervention would be a separate feedback condition.

## 4. Qualification, measurements and interventions

Before a latent pilot, native hook parity, content discrimination, contrastive-state swap effects, one-step update validity, readout fidelity, and positional controls must pass a separately reviewed gate. Mapping and verbalizer training may be necessary; such training must be reported and use material disjoint from confirmation. One-step qualification does not guarantee validity of long trajectories.

Meaning assessment uses blind before/after judgments of support, update type, new distinctions, and event-recovery violations. Numerical state change and lexical distance are not semantic scores. Activation measurements include selected residual, attention, MLP and, where supported, MoE gate outputs. Gate logits do not reveal actual expert dispatch frequencies. The development probes below test source discrimination and intervene on residual positions. Independent validation of specific semantic directions remains open.

Functional evaluation introduces each problem only after recurrence, using an isolated copy. Near transfer, structural transfer and less-related reasoning tasks are separate tiers. Correctness and explicitly defined utility are compared with initial-state, one-shot instruction and non-self controls. Reversal tasks distinguish useful commitment from rigidity. Frozen-weight state-conditioned improvements do not imply permanent learning.

At a saved state Z, a one-time intervention compares G^r(Z) with G^r(E(D(Z))). Continuous compression, norm/distance-matched perturbations, no-update roundtrips and meaning-preserving versus contrastive swaps are required to interpret differences beyond generic codec damage. The current harness implements the identity/language branch pair; the studies below add slot-order and magnitude controls. Distance-matched compression shams remain open.

The next development protocol tests a two-stage composition at depth two. Fixed E first supplies a question-independent state to a task-trained recipient. Training this native output path is an interface admission test. Its source, explicit-case and reintegrated teacher descriptions are encoded independently. The later G experiment must generate its own intermediate and final states from the preceding state alone. Independent teacher encodings supply supervised targets and diagnostics.

At the first stage boundary, the prescribed second operation receives the same-state replay, a same-meaning wording donor, or an opposed-policy donor. The source text and final evaluation question are absent from both updates. Omission of the first and second operation measures the contribution of each requested reconstruction. The registered compound unit must preserve its world, frame, question wording and displayed answer mapping while crossing policy polarity and clause order. Its numerical decision rules and bounded fits are in the phase-specific protocol. [Functional continuity protocol](../docs/FUNCTIONAL_CONTINUITY.md)

The primary language intervention branches once at the same first-stage state. One branch proceeds directly through the second update and the other verbalizes and re-encodes before that same update. Depths zero and one diagnose the first failed connection, and depth four is a fixed secondary analysis when admitted. A new two-stage confirmation panel and independent human transition review remain pending. The earlier sealed qualification material retains its original contract.

## 5. Development design

The first experiment compared native EOS with an eight-token ceiling on two Mistral descriptions about revisable or prompt decisions. We subsequently restored the archived literal, encoded, latent and roundtripped states and asked which priority each donor favored. Before collecting new scores, we fixed six additional topics with two opposing preferences and two phrasings per preference. They covered filing order, walking routes, meeting notes, temperature units, shelving and directions. For each topic all donors received the same two alternatives. The question and alternatives were absent during encoding, the single plain update and greedy verbalization.

For each donor we scored both answer labels and reversed their option assignments. The plus margin is the mean, over the two orders, of the log probability of the label naming the plus option minus that of the label naming the minus option. A positive margin favors the plus option. A pair is ordered if the plus donor has a larger margin than the minus donor. Correct identification of both donors additionally requires the two margins to straddle zero in the expected direction. Ties use the stored scores at their recorded precision. Each checkpoint contributed 12 matched pairs in six topic clusters. All results are descriptive. Alternative orders and phrasings are repeated measures.

We scored literal text, E, G after one update, B after re-encoding, reversed G rows, and G vectors with the donor's per-slot magnitudes on fixed directions. Zero vectors and uninjected native placeholders served as null inputs. For each of 24 descriptions we collected D(E) and D(G), and we generated from both nulls once. Both checkpoints used one RTX 5090, eager BF16, eight slots, uncalibrated identity transport, a common 1024-token context and their native EOS ids. The cut and writer were layers 16 and 31 for Mistral and layers 13 and 25 for Gemma (zero-indexed). These earlier probes used no training, prompt selection against their scores or human semantic rating.

## 6. Results

### Native EOS and the language roundtrip

Two opposed descriptions emphasized revisable or prompt decisions. Each underwent two plain updates in T, L and B, once with native EOS and once with an eight-token ceiling. The four trials used greedy BF16 Mistral-7B-Instruct-v0.3, eight slots, a 1024-token context, a pre-layer-16 cut and a post-layer-31 writer (zero-indexed). All 18 generations per policy include main-path, observer and branch readouts. All 18 reached EOS with the full context available, and all 18 reached the explicit ceiling under the cap. With EOS, each of the four B updates verbalized the same 13-token sentence about workspace contents. The four B states matched within each policy, although the two policies gave different states. The four L states differed within each policy and matched step for step across policies. [Plan, raw states and output text](../provenance/mistral_eos_20261101/RESULTS.md)

### Reading the original state donors

For the previously studied Mistral pair, the literal revisability and speed descriptions yielded plus margins of +16.766 and -16.344 nats. The two initial slot states both gave +2.844. The first and second L updates gave distinct fingerprints but positive margins for both donors. B gave +2.813 for both donors at both steps. This particular question separated the two literal descriptions and favored revisability for every archived latent donor.

### Six contrasts and matched controls

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

### Matched lexical donors and replay geometry

A lexical positive control placed the words of one answer option at the eight designated positions, padded on the right with the neutral period token. No answer letter was included. We compared native scoring, ordinary E on the same words, and injection of residuals captured at the same positions in a prefix ending at the last slot. Native words yielded 11/12 correct donor signs on Mistral and 12/12 on Gemma. Ordinary E gave 6/12 on each. The short-prefix aligned donor gave 6/12 and 7/12. Replaying neutral slots captured from a shorter forward changed individual label log probabilities by up to 0.1903 and 0.1613 nats. Those discrepancies motivated a separate length-matched experiment.

We appended one neutral dummy token to every scoring prompt and read the logit vector immediately before it. This reproduced all earlier native label scores exactly. The eight-slot donor now used a full-length forward, with neutral tokens replacing the question and alternatives after the slots. Neutral replay then matched every native vocabulary logit in all 12 question/order combinations on both checkpoints. The opaque-tail lexical donor's slots also matched its natural full-length counterpart exactly in all 24 donor/orders per checkpoint.

| Intervention | Mistral correct | Gemma correct | Mistral median gap (nats) | Gemma median gap (nats) |
|---|---|---|---|---|
| Native words | 11/12 | 12/12 | 28.219 | 19.672 |
| Eight slots | 6/12 | 7/12 | 0.008 | 0.219 |
| Downstream span | 11/12 | 12/12 | 27.594 | 19.734 |
| Both regions | 11/12 | 12/12 | 28.219 | 19.672 |

We next transplanted the natural lexical donor's middle-layer residuals from the positions after the slots through the scoring position. These downstream residuals had already processed the question and alternatives. Transplanting only this region into a neutral recipient reproduced the native preference sign in all 12 cases per checkpoint, including Mistral's error on the author-shelving donor. Its full logit vectors could still differ. Transplanting the slots and downstream region together recovered the entire native scoring-position logit vector exactly in 24/24 donor/orders per checkpoint. [Plans, raw scores, state files and all control results](../provenance/slot_content_20261101/RESULTS.md)

### Localizing the downstream span

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

### What reaches the period after the second alternative?

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

### Question-independent carriers before learning

We built all donors before constructing any question and reused each unchanged state for two question wordings and both option orders. The twelve exposed topics had two opposed descriptions and two phrasings each. Each checkpoint supplied 48 donors per route, eight rows and a fixed 256-position donor forward. Workspace captures the original E slots. Native tail captures the last eight source-token positions in a description-only prompt. Both use the earlier midpoint and eager BF16. No layer or row-count search followed the scores.

| Checkpoint | Carrier | Group accuracy | Best fixed null | Literal source |
| --- | --- | --- | --- | --- |
| Mistral | Workspace | 50.0% | 50.0% | 100.0% |
| Mistral | Native tail | 50.5% | 49.5% | 100.0% |
| Gemma | Workspace | 50.0% | 50.0% | 96.9% |
| Gemma | Native tail | 59.9% | 50.0% | 96.9% |

Accuracy averages within twelve groups and then over groups, with 192 dependent readouts per cell. None passed the predeclared 80% accuracy plus ten-point null advantage. All 192 free readouts reached EOS. Native-tail D produced 45 strings on Mistral and five on Gemma, while each workspace produced one. Diversity alone did not identify the source. Both native hook checks and all 96 full-vocabulary neutral replays were exact. An initial Gemma attempt captured 82 donors and scored no questions before stopping on a seven-token source. Its failure is retained. Both pole-zero photo phrasings were lengthened, and native shape validation was moved before model construction.

### The first codec-only learning run

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

### Source-specific learning and the order of a policy

We started two new rank-16 Gemma encoder bridges at identical weights. Each saw both opposed donors per pair, 64 AdamW updates, 512 description forwards and 44,804 training tokens. Both computed whole-source NLL and a two-token cross entropy at the first divergence (copper versus silver) following an identical prefix. The latter term was weighted zero for copy and one for copy plus opposition. Step 64 was fixed in advance. Replacing the divergent token and its entire suffix with placeholders at identical length reproduced all 192 dev decision margins exactly.

Copy and copy plus opposition obtained 49.35% and 50.00% group-macro functional accuracy, respectively. Each recorded zero correct opposed switches in 288 world, wording and label-order comparisons, and zero in 48 shared-prefix lexical pairs. Literal text reached 87.89%. The functional difference was +0.65 percentage points (scope +0.00, priority +1.30). The paired 95% site-resampling interval was +0.26 to +1.04 points, conditional on these two templates. The prespecified practical gain was ten points.

In the unadapted slots, a train-only centroid direction correctly ordered 48/48 original dev donor pairs by first-mentioned tray. Its fixed threshold classified 48/96 single descriptions. This prompted a separately sealed geometry test. We reversed the order of the two policy clauses while preserving each truth table. A shortcut that knows the construction's trigger and treats the first-mentioned action as its answer went from 288/288 correct description-world cases to 144/288. Each new description was encoded alone and again with both tray adjectives replaced by the same single token, metal. Source-minus-neutral features and raw features used separate train-only centroids.

| Fixed readout, reordered clauses | Raw slots | Source minus neutral | Centered, shuffled labels |
| --- | ---: | ---: | ---: |
| First-mentioned tray | 48/96 | 93/96 | 21/96 |
| Tray prescribed at the trigger | 48/96 | 51/96 | 49/96 |
| Prescribed tray correct in both orders | 24/48 | 6/48 | 2/48 |

On crossed wording the raw threshold predicted copper for every source. Centering supplied single-source access to mention order. The centered trigger readout was correct on 45/48 descriptions with the trigger clause first and 6/48 with it second. The registered joint endpoint counted the policies for which both orders were correct. These two encodings require the original input text. Availability of such a reference during later latent updates remains open. [Comparison, follow-up and raw scores](../provenance/policy_binding_20261103/RESULTS.md)

### Native recipient learning before the two-stage reconstruction

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

## 7. Interpretation and limitations

In these fixtures the complete downstream region reproduced the native lexical choices. The partition and masking experiments narrow this observation to an option-order-sensitive contribution at the second displayed alternative's terminal period. Its counterbalanced contrast persisted with the question stem hidden, weakened when the option words were hidden, and reversed under a natural option-order swap. The large contrast retained on omission leaves room for other positions and their interactions.

The second period is later in the causal sequence and has access to both alternatives. These interventions have not separated separator identity from order, position or preceding lexical context. Neutral-token masks alter the donor distribution. The two panels share the same explicit lexical schema, and each checkpoint was tested at one middle-layer cut in BF16. The short-prefix replay discrepancy was removed by matching forward length, but its numerical kernel has not been identified. Question-independent state portability, independent evaluation of one-step meaning updates, blind judgments of new distinctions and held-out transfer remain open. No semantic gate has been approved. The source-specific training signal and centered encoder were tested on one Gemma checkpoint, two construction families and two action words. A change in clause order preserved the truth conditions but exposed strong sensitivity to first mention. Training a recipient on crossed condition–action assignments and testing new constructions is the next functional comparison. The one-step G rubric still requires independent assessment of useful distinctions, warranted stability and unsupported additions.

## 8. Proposed two-stage study and analysis

The next bounded development experiment uses Gemma-2-2B-IT to construct a usable route through a current policy state. Recipient training is confined to task forwards, leaving the encoder and recurrent operations fixed during that phase. A recipient endpoint can establish that a learned native output path uses independently encoded policy states. Two-step G learning and the single language insertion require their own source seals and evidence.

The primary composition depth is two. Source, first-stage and second-stage observations are assessed with the operation omissions and intermediate-state exchanges described above. The interface, candidate cap, training budget and compound scorer are frozen before their respective phase is evaluated. A passing development candidate is then frozen for a separately authored confirmation panel. Human transition review is required for a reviewed semantic qualification verdict.

The principal language contrast is the paired effect of one E/D insertion at a shared first-stage state on the second update and later task. A practical-equivalence statement requires the full 95% paired interval to lie within the registered five-percentage-point margin. Meaning groups and construction families determine the scope of inference. Actual D/E and readout costs, exact ties, native-output coverage, failures and censored generations retain their denominators. The phase-specific protocol provides positive endpoints and bounded negative closure. A procedure frozen after a successful Gemma result can subsequently be examined on Mistral under a new binding.

## References

See `references.bib` (15 audited entries). All bracketed keys resolve there. For compilation, use the LaTeX version. The source audit records dated review status, not a guarantee that no later correction exists.
