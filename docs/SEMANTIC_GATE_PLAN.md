# A bounded route to semantic qualification

English · [日本語](SEMANTIC_GATE_PLAN.ja.md)

**Status on 2026-11-01 — proposed research plan.** This document starts from `6b054baf2571c3fa0e4f19d5d841ea760d3eaba2`. It records no new experiments, training or gate approval. Dataset manifests, the training implementation and the executable evaluation specification remain to be created before their respective stages run. Numerical targets below are proposed screening criteria for a small qualification study, not observed results or a preregistration.

There are **six work packages**. Packages 1–3 prepare a first learning run, package 4 fits and freezes a candidate, package 5 challenges it on an untouched qualification set, and package 6 reviews the evidence. Work on the data and training implementation can overlap the bounded diagnostic. Completing a package can mean either meeting its positive endpoint or closing a specified negative diagnosis. A scientific gate pass has its own stricter conditions.

## What the existing evidence settles

The [content probes](../provenance/slot_content_20261101/RESULTS.md) found distinct E/G tensors but collapsed free verbalizations and roundtrips. The [downstream interventions](../provenance/downstream_span_20261101/RESULTS.md) localized a partial lexical contrast to a state that had already seen both alternatives. Reversing the donor's option mapping reversed the transferred contrast. That intervention cannot supply a question-independent recurrent state.

The next target is a state built from a current description alone, reused unchanged for several independently authored questions. Training will address E/D, post-writer to pre-cut transport, and, if needed, the one-step G operator. The semantic gate evaluates these interfaces using independently specified truth conditions and a fixed rubric.

The identity of the period, its later causal position and the preceding lexical context remain partly confounded. A complete mechanistic account of that period is a separate research question. It is not a prerequisite for testing a question-independent learned interface.

## The two completion routes

| Package | Route A — positive completion (`done`) | Route B — bounded diagnosis, then advance |
| --- | --- | --- |
| **1. Select a question-independent carrier** | A fixed eight-row route preserves the development contrasts under fresh questions, passes native/neutral replay, and meets the development content screen. | Test the two routes below once per checkpoint, publish every result, and identify which tested interfaces lack usable readout. Freeze a route for learning even if its untrained semantic screen fails. Keep question-conditioned suffix states as a separate positive control. |
| **2. Build independent material and a review instrument** | Author and review the truth conditions, supported one-step targets, no-update cases, blinded rubric and family-disjoint train/dev/qualification manifests. | Narrow the first claim to a fully reviewable synthetic domain if open-ended meaning judgments remain ambiguous. Retain independent truth conditions and disjoint groups. Missing golds, reviewers or split integrity remain explicit prerequisites for the stages that use them. |
| **3. Implement and seal a learning contract** | Matched activation targets, trainable parameter lists, losses, controls, artifact identity and gradient/isolation checks are executable within the fixed budget. | If valid one-step targets are unavailable, start with a codec/transport-only contract and mark the G-update claim pending. A disconnected gradient path, leaked evaluation input or unbound artifact must be fixed before learning starts. |
| **4. Fit, diagnose and freeze** | Development content/readout checks pass and a one-step candidate meets the update screen. Freeze the selected weights and complete configuration. | At the cap, retain the best candidate selected using dev data and all failed candidates. Record whether the tested bottleneck is reconstruction, functional readout or updating. A finite, technically valid candidate can undergo the qualification challenge even when a semantic failure is expected. |
| **5. Run the independent six-check challenge** | Every predeclared check passes on the untouched qualification material, with uncertainty, denominators and blind transition reviews reported. | Complete the same evaluation and record failed or inconclusive checks. Do not tune on this qualification set. A technical invalidity is recorded as `invalid`; it does not count as a valid semantic test. |
| **6. Review and bind the result** | A named reviewer accepts all six evidence files and their exact source/model/artifact bindings. Only then issue `SEMANTIC_GATE_REVIEWED_PASS` and admit a bounded pilot. | Record `NOT_QUALIFIED`, identify the failed evidence and nominate one next intervention using a fresh future holdout. The challenge can be complete with a negative result. Long latent pilots remain pending. |

Route B keeps an unresolved limitation visible while making the next experiment interpretable. It never changes a failed check to `pass`. In particular, native parity, input isolation, independently defined targets, split integrity and artifact binding cannot be waived by exhausting a budget.

## 1. Close the carrier diagnostic

Use the two already tested checkpoint snapshots and middle cuts. Keep eight rows and one cut per checkpoint. Evaluate only two routes on the exposed development topics, giving four checkpoint/route cells.

1. The current E workspace slots, captured after the complete source description.
2. Eight native trailing positions in a description-only donor forward. Define their selection and any padding before the run. A successful route would be a newly specified interface, not retroactive qualification of the original workspace.

Neither donor forward receives a question, alternatives, answer labels, future operator outputs or trajectory history. Fix donor padding and forward shape independently of the eventual question. Hash each saved state and reuse that same state for multiple question wordings and both option orders. Record actual positions, token IDs and source lengths. Check neutral self-replay with each forward geometry and include direct source text, empty/zero states, unrelated donors, row permutation and norm-only controls.

The positive development screen is at least 80% group-macro functional accuracy and a gain of at least 10 percentage points over the strongest fixed null control. This is a route-selection screen on exposed material. It supplies no qualification result. Stop after these four cells or a recorded technical failure. Do not open a layer, token-count or punctuation search in this work package. If neither route meets the screen, proceed to learned interfaces after packages 2–3 are ready.

## 2. Define what must survive and what may change

The first proposed dataset has **96 meaning groups across eight authored construction families**, split by whole family into 48 train groups, 24 dev groups and 24 qualification groups (four/two/two families). A family describes how facts, conditions or exceptions are constructed, rather than a superficial topic name. Reused templates, derivations, paraphrases and polarity reversals stay with their source group and family. All existing repository fixtures remain exposed development material and are excluded from qualification.

Each group supplies two opposed descriptions, at least two paraphrases, independent questions about source content and conditions, a support ledger, and one-step targets for the applicable fixed operators. Include cases where the correct output is unchanged. Cover the same operator inventory across splits while holding the construction families apart. Keep the later recurrence/transfer confirmation benchmark uncreated and outside this qualification cycle.

At two paraphrases per pole and all four operators, the 24 qualification groups produce 384 transition pairs and 768 individual ratings from two raters. Finalize the applicable-case ledger before unblinding and report that workload alongside model compute. The six packages are work streams, not six GPU jobs.

Gold meaning is specified from the authored facts and conditions before any D output. A teacher-generated rewrite can be a training candidate only after checking it against that ledger. Store original teacher outputs and rejection reasons. An attractive self-description alone is not a target. Never derive a correct answer or meaning label from the candidate's own verbalization.

Review before/after pairs blind to arm, training variant and model identity. The rubric separately records preserved facts, preserved conditions, supported implication/distinction, unjustified additions, contradiction and autobiographical intrusion. Use two independent human raters for the qualification transitions and retain their initial judgments, disagreements and adjudication. Rater availability is a prerequisite for a positive reviewed result. A mechanically scored diagnostic can still be published while that review is pending.

Freeze train/dev content and their manifests before fitting. Seal qualification material and its manifest before candidate selection, and keep its content inaccessible to training and development. Run a family/derivation overlap audit. Review qualification reference material without examining candidate outcomes. Candidate-driven exclusions are forbidden. Any pre-unblinding correction creates a new version with a recorded reason. Once qualification outcomes are inspected, the material is spent for further model selection.

## 3. Make the learning contract concrete

Start with **Gemma-2-2B-IT** at its already tested cut. Mistral remains a separately fitted replication and does not delay the first challenge. Freeze the pretrained backbone, vocabulary and tokenizer. Within the selected route, L and B must use the same E, G, transport and initialization. All learned modules are frozen during recurrence and qualification.

Use the following fixed progression.

| Candidate | Learned object and targets | Entry condition |
| --- | --- | --- |
| **R — ridge baseline** | One post-writer to pre-cut mapping, using `fit_transport.py` with ridge `1e-3`. Pair an update's writer rows with E rows of an independently validated target description, with explicit row correspondence. | Valid description/target pairs and coordinate definitions exist. Tensor reconstruction error and functional tests are both reported. A low MSE alone cannot select a semantic pass. |
| **C — codec/transport adapter** | A small adapter for the selected E/D route and boundary transport. Use source/target description reconstruction and preservation of independently authored facts/conditions. | R is retained as the baseline. Candidate C receives only train descriptions and reviewed targets, never qualification questions or answers. |
| **U — one-step operator adapter** | Add a small operator-conditioned module for supported G updates, using reviewed single-step source/target pairs. Include valid identity targets. | Candidate C passes the dev content/readout screens. Otherwise freeze C's negative result and evaluate that failure before fitting U. |

Predeclare the exact architecture and loss coefficients in the executable run specification. Proposed bounds are one ridge fit, one C run and at most one U run. Each neural run has at most 2,000 optimizer updates, eight effective training examples per update and two GPU hours. The total neural budget is at most four GPU hours, with at most two million trainable parameters per candidate including inherited learned modules. Fix training seed 47 and the dev checkpoint-selection rule. Use no automatic restart or hyperparameter sweep. These are first-cycle bounds, not a hardware runtime estimate. Finalize the remaining optimizer settings and validation cadence before any real fitting.

The current `SlotHost.forward` uses `torch.inference_mode()` and `State` detaches tensors. A training implementation needs a separate differentiable path. Do not remove inference isolation from the experiment host merely to obtain gradients. Verify nonzero finite gradients in the intended modules, no backbone gradients or parameter changes, and numerical agreement between the trained artifact's evaluation path and the experimental host. Tiny controlled fixtures can test this before pretrained fitting.

Bind every learned E/D/G/transport artifact, architecture, template, normalization, checkpoint/tokenizer, slots, cuts and stopping policy into the host identity. The current transport-only artifact field is insufficient for new learned modules. The gate and loader must reject any changed learned artifact. Keep the training-data manifest and the source hash in the evidence chain. A new carrier is a new route and receives a new binding.

## 4. Diagnose by stage and freeze the candidate

Measure E, D(E), G(E), D(G(E)) and E(D(G(E))) separately using functional questions and gold support, alongside reconstruction loss. Score all outputs against the intended current meaning, including any supported one-step change. These comparisons identify failures of the tested interfaces. They do not prove that inaccessible information is absent from the tensor.

If R fails and C succeeds, report the specific mapping result. If functional content survives in E but its D readout fails, retain both observations. If the codec passes while G supplies no supported change beyond identity, keep the carrier and update outcomes separate. If all tested routes collapse, a failed qualification is still a useful endpoint. Preserve first failures and all intermediate artifacts.

Select one candidate with a rule fixed before training, using only dev data. Freeze weights, source, identities, evaluation specification and the six evidence destinations before unblinding. If every candidate is semantically weak, select the least-failing technically valid candidate under that same rule and label it a negative challenge. If no candidate is technically valid, stop and record an implementation failure.

## 5. Challenge the frozen candidate once

The following practical thresholds are **initial proposals**. Finalize them, the denominators, allowed perturbations, uncertainty calculation and missing-output rules using train/dev material before unblinding qualification. Any subsequent change needs a new future qualification set. Report paired group-level uncertainty, with both donors, all paraphrases, question variants and label orders kept together. With only two qualification construction families, any pass has a narrow scope.

| Existing gate check | Proposed positive endpoint |
| --- | --- |
| `native_parity` | Exact full-logit agreement for native versus observational-hook forwards at identical shapes and dtype, with learned interventions disabled. Neutral self-replay and state/observer isolation also pass for every registered geometry. A mismatch invalidates the affected semantic test. |
| `content_discrimination` | At least 80% balanced group-macro correctness and at least 10 percentage points over every fixed empty/zero/random/unrelated-state control on independently authored questions. Count exact ties and missing answers as unsuccessful. |
| `meaning_swap_effect` | Opposed-state swaps produce the independently specified answer change in at least 80% of applicable comparisons. Same-meaning paraphrase swaps preserve the intended answers in at least 90%. Include unchanged-state and unrelated-state controls, and test both label orders. |
| `one_step_update` | At least 80% of reviewed transitions preserve supported meaning, unjustified additions occur in at most 5%, and at least 70% of change-required cases realize the predeclared useful distinction. On those cases, functional correctness exceeds identity/no-update and unrelated-state controls by at least 10 percentage points. Score no-change cases separately. |
| `readout_fidelity` | Both D(G(E)) text and its re-encoded state retain at least 80% functional correctness against the intended updated meaning. Re-encoding loses at most 10 percentage points relative to G(E), and unjustified additions in the readout occur in at most 5%. A consistently uninformative roundtrip cannot pass by preserving itself. |
| `positional_controls` | Under each allowed, meaning-preserving prefix/position variation, functional correctness stays at least 75% and falls by at most 10 percentage points from the aligned route. Record the geometry. Row permutation and norm-only substitutions are diagnostic controls, not presumed meaning-preserving transformations. |

Fix a precedence rule before scoring. Leakage, mutated states, altered artifacts, failed parity or broken golds produce `invalid` evidence. Missing human review produces `pending` evidence. A valid semantic score below a threshold produces `fail`. Neither pending nor invalid evidence is a passing check.

Qualification receives at most one evaluation pass per frozen candidate, with the scorer and native controls run on the same predefined units. Never drop a group because the native host failed it after unblinding. Keep technical failures, exact ties, missing outputs and censored generations in the all-attempt denominator, and report completed-output results separately. Native EOS is the default stopping policy. A nonempty bounded output remains available for assessment with its censoring flag. Truncation alone is not an automatic semantic failure.

## 6. Review, record and decide admission

Map the results to all six keys in [the existing gate contract](QUALIFICATION.md). Preserve raw scores, references, blind reviews, checkpoints, training receipts, failures and file digests. The reviewer checks the underlying evidence and its scope as well as the artifact contract.

Only six accepted `pass` verdicts with an actual named review can produce `SEMANTIC_GATE_REVIEWED_PASS`. Bind the result to the tested source tree and all learned artifacts. Run any admitted pilot from that unchanged binding. Editing a manifest-tracked source or document requires a fresh binding under the existing contract.

If the candidate fails, preserve `NOT_QUALIFIED` and a check-by-check record of `pass`, `fail`, `pending` or `invalid`. Specify one next intervention tied to the failed interface and author a fresh holdout before a further selection cycle. Do not silently enlarge this cycle's budget. A pass admits only a bounded one-step-qualified pilot. Multi-step retention, self-reference-specific effects, held-out reasoning transfer, cross-model replication and matched-compute L/B attribution remain later questions.

## What is complete at each milestone

- **Learning-ready** means packages 1–3 have either positive completion or their permitted diagnostic closure, and every technical prerequisite for the chosen learning scope passes. Semantic success before fitting is not required.
- **Challenge-ready** means a technically valid candidate and its complete evaluation specification are frozen, qualification material remains untouched, and the required review is arranged. A candidate with known dev weaknesses may still be tested as a negative result.
- **Challenge-complete** means all attempted units and their outcomes are recorded under the frozen specification. A valid failure remains a valid challenge result. Missing review or invalid execution remains visible as incomplete scientific evidence.
- **Gate-passed** means all six scientific checks and the exact binding have actually been reviewed and accepted. Budget exhaustion or a completed negative challenge never grants pilot admission.

The immediate next work is the four-cell question-independent carrier diagnostic, alongside authoring the independent support ledgers and split manifests. Further punctuation localization stays outside this critical path.
