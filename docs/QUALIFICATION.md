# Semantic qualification before latent pilots

A finite tensor and a plausible D output do not establish semantic recurrence. The gate distinguishes the experiment's construct from an arbitrary dynamical system with a fluent decoder.

The paired-donor probes measured content discrimination, free-readout fidelity, magnitude and slot-order controls, and single-forward position interventions on Mistral and Gemma. Literal descriptions were identifiable, while a full roundtrip gave equal scores for opposing donors in all 12 paired comparisons per model. The downstream-span control had access to the question during its donor forward. A reviewed semantic gate for a reusable latent update is still open. [Development evidence](../provenance/slot_content_20261101/RESULTS.md)

Follow-up experiments partitioned those downstream positions. At the period after the second alternative, the donor's option order affected the sign of the transferred contrast on both panels and checkpoints. The other positions retained most of the contrast when that period was excluded. These controls concern a lexical readout after the alternatives are presented. A question-independent latent update still requires its own qualification. [Localized interventions](../provenance/downstream_span_20261101/RESULTS.md)

The [bounded qualification plan](SEMANTIC_GATE_PLAN.md) proposes six work packages, with positive completion and permitted negative closure for each. It separates readiness to learn, a completed qualification challenge and a reviewed gate pass. Its numerical screens are proposals to fix before evaluation, and do not change the current gate status. [日本語](SEMANTIC_GATE_PLAN.ja.md)

## Required checks

The description-only study closed all four native-interface cells below its development screen. A first Gemma encoder-bridge fit improved source NLL but left functional dev accuracy near 50%, with little aggregate matched-versus-opposed donor advantage. The codec is unqualified, qualification material is still sealed, and independent human transition review is pending. [First learning evidence](../provenance/carriers_20261102/RESULTS.md)

| Check | Evidence to collect on calibration-only data |
|---|---|
| Native parity | Unmodified host and observational hooks agree; injections disabled |
| Content discrimination | E and D preserve independently specified contrasts across held-out paraphrases; compare empty/random states |
| Meaning swap effect | Swapping between contrastive states changes the relevant task response, while same-meaning swaps have smaller/no corresponding effect |
| One-step update | G performs the requested meaning operation better than identity/no-update and unrelated-state controls; blind transition review |
| Readout fidelity | D/E roundtrip preserves specified distinctions and is not merely inventing plausible self-descriptions; compare functional tests, not D's words alone |
| Positional controls | Slot position, prefix length, norm and relocation effects are measured separately; aligned and perturbed controls do not erase the intended contrast |

Predeclare tasks, denominators, thresholds and the review procedure before qualification. Set thresholds on a separate development subset, then evaluate the calibration holdout. Keep downstream confirmation answers separate from transport fitting, cut selection and verbalizer development.

## Suggested small calibration progression

Start from contrasting short descriptions with independently authored truth-conditional questions and paraphrases. Verify that native host text use succeeds first. Then test E/D on held-out paraphrases and explicit state swaps. Test G for one update on support/contradiction/condition questions. Long rollouts come only after these checks, and still need monitored off-manifold diagnostics.

The native identity transport is allowed only as an engineering diagnostic. `scripts/fit_transport.py` accepts paired input/output boundary activations and fits a ridge mapping; it does not decide how valid pairs should be constructed, train a verbalizer, or supply the evidence above. Adapter or codec training may be required. That additional work is explicitly unfinished and cannot be described as minor environmental tuning.

A learned route is acceptable, but report what it learned. Train single-step operations on independent content, not the long trajectories or benchmark solutions to be tested. Pretrained D/E may not be transplanted across model families, layers or normalizations without separate compatibility evidence.

## Gate record

Copy `configs/qualification.example.json` to a separate evidence location. It deliberately contains only pending checks. `verify_gate` requires a reviewer, a calibration-only split, the exact host identity hash, exact source hash and six referenced evidence files with SHA256 values. Set `SEMANTIC_GATE_REVIEWED_PASS` only after an actual review. A matching status string is an access contract, not a substitute for scientific judgment; preserve the evidence for audit.

Changing checkpoint, transport, layer, slots, template, generation budget or source requires a new reviewed binding. Keep the failed gate and its traces. A passed one-step gate is admission to a pilot, not proof that twelve steps preserve meaning or improve reasoning.
