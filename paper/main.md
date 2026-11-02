# Self Without a Transcript
## Semantic Recurrence Across Text and Latent States

**Research working paper · November 3, 2026**

### Abstract

Does a distinction survive when a model verbalizes its state, and can it guide a later decision? We compare text relay, latent recurrence and a language roundtrip after the same latent update. Untrained, question-independent carriers on Mistral-7B and Gemma-2-2B missed a development content screen. A Gemma encoder bridge reduced source-text loss while opposed donors retained almost the same improvement. Two fresh, compute-matched fits then gave 49.35% and 50.00% functional accuracy. Neither correctly switched actions for both opposed policies in any of 288 comparisons. In a separate train-only linear readout, subtracting the encoding of a neutral rendering made the first-mentioned tray decodable in 93/96 clause-reordered descriptions. The prescribed action was readable in 51/96, with both orders correct for 6/48 policies. These development results separate access to word order from reliable use of a rule. One-step semantic updating and transfer remain open.

## 1. Problem formulation

A description such as “I prioritize keeping decisions revisable” may be reinterpreted as a preference for preserving correction opportunities. This transition can add a useful distinction, remain a paraphrase, or drift into an unsupported claim. The object of study is the transition in meaning, not the event of having uttered the earlier description. We therefore exclude a direct transcript and instructions that infer character from the act of speaking. This exclusion does not erase the causal influence of past updates on the current state.

We ask three questions. Does repeated meaning transformation yield defensible distinctions rather than only lexical variation? Does a language roundtrip alter the formation or subsequent use of those distinctions? Do any changes improve held-out task performance or appropriate switching when conditions reverse?

Convergence is not intrinsically a failure: useful stabilization and rigid commitment must be separated by their effects. Novelty is not intrinsically a success: unsupported reinterpretation can move far while losing the initial content. “Self” denotes an experimental attribution frame. It is not assumed to establish subjective experience or a stable personal identity.

## 2. Related work and positioning

Coconut trains continuous-state reasoning through hidden-state feedback [hao2024coconut]. Recurrent-depth models study test-time computation through repeated internal updates [geiping2025huginn]. STARS motivates distinguishing useful stable dynamics from performance collapse in its studied looped models [yang2026stars]. These works do not establish that an arbitrary untrained residual feedback route preserves the meanings of self-descriptions.

Persona vectors and the Assistant Axis connect persona-related behavior with activation structure and interventions [chen2025persona; lu2026axis]. Natural Language Autoencoders provide a learned activation-to-text-to-activation construction, while distinguishing reconstruction and explanation issues [frasertaliente2026nla]. The present main contrast is an intervention with a specific codec, not “language versus pure thought.” A latent chain-of-thought causal study also motivates separating output commitment from representational commitment [li2026dynamics].

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

## 7. Interpretation and limitations

In these fixtures the complete downstream region reproduced the native lexical choices. The partition and masking experiments narrow this observation to an option-order-sensitive contribution at the second displayed alternative's terminal period. Its counterbalanced contrast persisted with the question stem hidden, weakened when the option words were hidden, and reversed under a natural option-order swap. The large contrast retained on omission leaves room for other positions and their interactions.

The second period is later in the causal sequence and has access to both alternatives. These interventions have not separated separator identity from order, position or preceding lexical context. Neutral-token masks alter the donor distribution. The two panels share the same explicit lexical schema, and each checkpoint was tested at one middle-layer cut in BF16. The short-prefix replay discrepancy was removed by matching forward length, but its numerical kernel has not been identified. Question-independent state portability, independent evaluation of one-step meaning updates, blind judgments of new distinctions and held-out transfer remain open. No semantic gate has been approved. The source-specific training signal and centered encoder were tested on one Gemma checkpoint, two construction families and two action words. A change in clause order preserved the truth conditions but exposed strong sensitivity to first mention. Training a recipient on crossed condition–action assignments and testing new constructions is the next functional comparison. The one-step G rubric still requires independent assessment of useful distinctions, warranted stability and unsupported additions.

## 8. Proposed pilot and analysis

The development bank comprises six content groups, two opposed poles, two paraphrases and self/other frames: 48 initial descriptions. Crossed with three arms and two schedules, this gives 288 proposed trajectories per model. Four candidate families are Mistral-7B-Instruct-v0.3, Llama-3.2-3B-Instruct, Gemma-2-2B-IT and Phi-mini-MoE, with Phi-tiny-MoE as a separately reported fallback. These models are not matched in parameter count, architecture or training, so cross-family comparisons are descriptive replication tests, not causal architecture comparisons.

Twelve steps and fixed assessment checkpoints are proposed, not retrospectively optimized. The current CLI evaluates downstream tasks at baseline and final state; intermediate task automation remains to be added. All 20 bundled task fixtures are exposed development material. A confirmation split, sample-size decision, thresholds and preregistration remain pending.

Primary analysis will use paired L/B utility differences at a declared depth, while reporting semantic support and switching separately. Meaning groups and task families, rather than individual steps, determine dependence-aware inference. Same-step comparisons also report actual compute; matched-compute analyses are secondary. Failures, censored generations, exact ties and excluded cases retain explicit denominators. No winner is selected using the confirmation set's best trajectory depth.

## References

See `references.bib` (13 audited entries). All bracketed keys resolve there. For compilation, use the LaTeX version. The source audit records dated review status, not a guarantee that no later correction exists.
