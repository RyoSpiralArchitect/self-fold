# Self Without a Transcript
## Semantic Recurrence Across Text and Latent States

**Research working paper · November 1, 2026**

### Abstract

What can a frozen language model recover after a description is encoded into eight hidden-state slots? We compare text relay, latent recurrence and a language roundtrip inserted after the same latent update. In Mistral-7B-Instruct-v0.3, allowing generation to reach native EOS still produced one roundtripped state across two opposed initial descriptions. Across six further development contrasts, Mistral and Gemma-2-2B-IT each identified 24/24 literal descriptions. Each checkpoint produced 24 distinct updated tensors but one greedy readout string. In a lexical positive control with matched sequence lengths, supplying only eight middle-layer donor slots to a neutral recipient gave 6/12 correct preferences on Mistral and 7/12 on Gemma. Supplying downstream donor positions that had already processed the question reproduced all 12 native preference signs on both checkpoints. These interventions measure access to explicit content at a particular boundary. Semantic qualification of the recurrent state and transfer to new tasks remain open.

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

We scored literal text, E, G after one update, B after re-encoding, reversed G rows, and G vectors with the donor's per-slot magnitudes on fixed directions. Zero vectors and uninjected native placeholders served as null inputs. For each of 24 descriptions we collected D(E) and D(G), and we generated from both nulls once. Both checkpoints used one RTX 5090, eager BF16, eight slots, uncalibrated identity transport, a common 1024-token context and their native EOS ids. The cut and writer were layers 16 and 31 for Mistral and layers 13 and 25 for Gemma (zero-indexed). There was no training, prompt selection against these scores or human semantic rating.

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

## 7. Interpretation and limitations

In these development fixtures, relaxing the token ceiling changed the text and the re-encoded state while the roundtrip still mapped opposing donors to one state. The fixed question could read explicit lexical content. The standard encoded state and the matched eight-slot transplant usually failed to recover opposite preference signs. At the measured cut, the downstream-span intervention sufficed to reproduce the native choices. That span contains more positions and comes from a donor that saw the actual question. Its success does not identify a unique storage location or establish a reusable semantic representation.

The study has two frozen checkpoints, six hand-authored topics, one selected cut per model, greedy decoding, BF16 arithmetic and an uncalibrated transport. The replay experiment isolated an input-length effect under these settings. It did not identify the responsible numerical kernel. Small direction rankings should be read with the raw margins and controls. Independent evaluation of one-step meaning updates, blind judgments of new distinctions and held-out transfer remain open. No semantic gate has been approved.

## 8. Proposed pilot and analysis

The development bank comprises six content groups, two opposed poles, two paraphrases and self/other frames: 48 initial descriptions. Crossed with three arms and two schedules, this gives 288 proposed trajectories per model. Four candidate families are Mistral-7B-Instruct-v0.3, Llama-3.2-3B-Instruct, Gemma-2-2B-IT and Phi-mini-MoE, with Phi-tiny-MoE as a separately reported fallback. These models are not matched in parameter count, architecture or training, so cross-family comparisons are descriptive replication tests, not causal architecture comparisons.

Twelve steps and fixed assessment checkpoints are proposed, not retrospectively optimized. The current CLI evaluates downstream tasks at baseline and final state; intermediate task automation remains to be added. All 20 bundled task fixtures are exposed development material. A confirmation split, sample-size decision, thresholds and preregistration remain pending.

Primary analysis will use paired L/B utility differences at a declared depth, while reporting semantic support and switching separately. Meaning groups and task families, rather than individual steps, determine dependence-aware inference. Same-step comparisons also report actual compute; matched-compute analyses are secondary. Failures, censored generations, exact ties and excluded cases retain explicit denominators. No winner is selected using the confirmation set's best trajectory depth.

## References

See `references.bib` (13 audited entries). All bracketed keys resolve there. For compilation, use the LaTeX version. The source audit records dated review status, not a guarantee that no later correction exists.
