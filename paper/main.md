# Self Without a Transcript
## Semantic Recurrence Across Text and Latent States

**Working protocol draft, 2026-11-01.** Authorship and venue are not finalized. This is not a report of completed pretrained-model experiments.

### Abstract

We propose a study of recursive semantic self-description without retaining a dialogue transcript. The question is whether successive reinterpretations produce justified new distinctions that affect later problem solving, and whether verbalization supports or constrains this process. Three conditions separate text relay (T), continuous-state recurrence (L), and the same continuous update with an interposed language roundtrip (B). We combine a fixed expansion-oriented intervention with observations of meaning, activations, and downstream effects. Evaluation problems are withheld until recurrence has finished. One-time interventions at shared states are intended to test how a particular verbalizer/re-encoder changes subsequent trajectories. A semantic qualification gate precedes any claim that an injected-state route realizes the intended operation. The present release contains a protocol and tested small-decoder software infrastructure, not evidence of reasoning improvement, semantic selfhood, or consciousness.

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

Meaning assessment uses blind before/after judgments of support, update type, new distinctions, and event-recovery violations. Numerical state change and lexical distance are not semantic scores. Activation measurements include selected residual, attention, MLP and, where supported, MoE gate outputs. Gate logits do not reveal actual expert dispatch frequencies. Independent semantic probes and causal patching are planned, not completed.

Functional evaluation introduces each problem only after recurrence, using an isolated copy. Near transfer, structural transfer and less-related reasoning tasks are separate tiers. Correctness and explicitly defined utility are compared with initial-state, one-shot instruction and non-self controls. Reversal tasks distinguish useful commitment from rigidity. Frozen-weight state-conditioned improvements do not imply permanent learning.

At a saved state Z, a one-time intervention compares G^r(Z) with G^r(E(D(Z))). Continuous compression, norm/distance-matched perturbations, no-update roundtrips and meaning-preserving versus contrastive swaps are required to interpret differences beyond generic codec damage. The current harness implements the identity/language branch pair; the fuller sham suite remains planned.

## 5. Proposed pilot and analysis

The development bank comprises six content groups, two opposed poles, two paraphrases and self/other frames: 48 initial descriptions. Crossed with three arms and two schedules, this gives 288 proposed trajectories per model. Four candidate families are Mistral-7B-Instruct-v0.3, Llama-3.2-3B-Instruct, Gemma-2-2B-IT and Phi-mini-MoE, with Phi-tiny-MoE as a separately reported fallback. These models are not matched in parameter count, architecture or training, so cross-family comparisons are descriptive replication tests, not causal architecture comparisons.

Twelve steps and fixed assessment checkpoints are proposed, not retrospectively optimized. The current CLI evaluates downstream tasks at baseline and final state; intermediate task automation remains to be added. All 20 bundled task fixtures are exposed development material. A confirmation split, sample-size decision, thresholds and preregistration remain pending.

Primary analysis will use paired L/B utility differences at a declared depth, while reporting semantic support and switching separately. Meaning groups and task families, rather than individual steps, determine dependence-aware inference. Same-step comparisons also report actual compute; matched-compute analyses are secondary. Failures, censored generations, exact ties and excluded cases retain explicit denominators. No winner is selected using the confirmation set's best trajectory depth.

## 6. Current status and limitations

No pretrained-model or Furnace result is reported. CPU tests and a randomly initialized small causal decoder establish selected software invariants only. The latent transport and codec have not passed semantic qualification. Their learning and calibration may require substantive additional work. The repository intentionally blocks unqualified L/B scientific runs.

The proposed effect of B conflates verbalization with its particular reconstruction error and additional computation. Positional encoding, prompt framing, length, norm and tokenization are potential confounds. A fluent decoder can generate an explanation not faithfully encoded in its input. New distinctions need independent functional validation. A temporary improvement on a small task does not establish general intelligence, selfhood or consciousness.

## 7. Results

**Not available.** Do not replace this section with engineering pass counts or illustrative model prose. Populate it only after frozen, reviewed empirical runs; retain negative and failed results.

## References

See `references.bib` (13 audited entries). All bracketed keys resolve there. For compilation, use the LaTeX version. The source audit records dated review status, not a guarantee that no later correction exists.
