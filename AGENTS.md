# SELF-FOLD: agent entry point

Read `docs/FRAMING_HANDOFF.md`, `docs/IMPLEMENTATION_STATUS.md`, and `docs/FURNACE_RUNBOOK.md` before changing or running anything beyond CPU tests.

This is a research harness, not a demonstration that a model has a self. Preserve the T/L/B contrast and distinguish engineering success from semantic qualification and downstream benefit.

1. No transcript/KV/evaluation feedback into recurrence. Observers and task readouts use clones and isolated RNG.
2. L and B must use the same G, same initialization, same operator schedule, same model and transport. Only B inserts E(D(.)); count its extra compute.
3. Never optimize for novelty alone or mark stable fixed points as failure by definition.
4. Do not silently change prompts, positions, normalizers, sampling, quantization, checkpoints, or gates after looking at results.
5. Source-seal runs. Preserve failed attempts under distinct output identities. No automatic retry, deletion of prior models/evidence, or unrelated GPU jobs.
6. No download or remote-code execution by default. Resolve real local paths and inspect live resources before model loading. Do not infer RAM/VRAM availability from historical context.
7. Treat `identity_unqualified` and fitted-but-unreviewed transports as engineering-only. Do not manufacture a PASS receipt to unlock a pilot.
8. No paid service, remote publication, model upload, or full matrix launch from CI. The publish helper is a separate explicit action on an authenticated machine.
9. Return inspectable traces and unresolved distinctions to Ryō. Interpretation and hypothesis revision are not to be auto-closed as an implementation task.

Primary check: `PYTHONPATH=src python -m pytest -q`. Optional architecture check on a prepared machine: `python -m pytest -q -m hf`; skipped tests are not success.
