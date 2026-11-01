# Semantic qualification before latent pilots

A finite tensor and a plausible D output do not establish semantic recurrence. The gate distinguishes the experiment's construct from an arbitrary dynamical system with a fluent decoder.

## Required checks

| Check | Evidence to collect on calibration-only data |
|---|---|
| Native parity | Unmodified host and observational hooks agree; injections disabled |
| Content discrimination | E and D preserve independently specified contrasts across held-out paraphrases; compare empty/random states |
| Meaning swap effect | Swapping between contrastive states changes the relevant task response, while same-meaning swaps have smaller/no corresponding effect |
| One-step update | G performs the requested meaning operation better than identity/no-update and unrelated-state controls; blind transition review |
| Readout fidelity | D/E roundtrip preserves specified distinctions and is not merely inventing plausible self-descriptions; compare functional tests, not D's words alone |
| Positional controls | Slot position, prefix length, norm and relocation effects are measured separately; aligned and perturbed controls do not erase the intended contrast |

Predeclare tasks, denominators, thresholds and review procedure before the qualification evaluation. This starter intentionally supplies no arbitrary universal quality threshold. Choose thresholds on a separate development subset, then evaluate the calibration holdout. Never use downstream confirmation answers to fit the transport, choose a cut, or select a verbalizer.

## Suggested small calibration progression

Start from contrasting short descriptions with independently authored truth-conditional questions and paraphrases. Verify that native host text use succeeds first. Then test E/D on held-out paraphrases and explicit state swaps. Test G for one update on support/contradiction/condition questions. Long rollouts come only after these checks, and still need monitored off-manifold diagnostics.

The native identity transport is allowed only as an engineering diagnostic. `scripts/fit_transport.py` accepts paired input/output boundary activations and fits a ridge mapping; it does not decide how valid pairs should be constructed, train a verbalizer, or supply the evidence above. Adapter or codec training may be required. That additional work is explicitly unfinished and cannot be described as minor environmental tuning.

A learned route is acceptable, but report what it learned. Train single-step operations on independent content, not the long trajectories or benchmark solutions to be tested. Pretrained D/E may not be transplanted across model families, layers or normalizations without separate compatibility evidence.

## Gate record

Copy `configs/qualification.example.json` to a separate evidence location. It deliberately contains only pending checks. `verify_gate` requires a reviewer, a calibration-only split, the exact host identity hash, exact source hash and six referenced evidence files with SHA256 values. Set `SEMANTIC_GATE_REVIEWED_PASS` only after an actual review. A matching status string is an access contract, not a substitute for scientific judgment; preserve the evidence for audit.

Changing checkpoint, transport, layer, slots, template, generation budget or source requires a new reviewed binding. Keep the failed gate and its traces. A passed one-step gate is admission to a pilot, not proof that twelve steps preserve meaning or improve reasoning.
