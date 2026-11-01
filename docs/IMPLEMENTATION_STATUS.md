# Implementation status

This table records the state of the supplied starter on 2026-11-01. Its original results are in [the starter receipts](../provenance/RECEIPTS.md). The subsequent public launch ran all 85 local tests, including four random-configuration HF architecture tests, and four CPU recurrence trials. See the [launch results](../provenance/launch_cpu_20261101/RESULTS.md).

The first pretrained experiment used one Mistral-7B-Instruct-v0.3 snapshot on Furnace. All four planned trials completed. EOS stopping and an eight-token ceiling were compared with the same seeds, weights and operators. See the [pretrained results](../provenance/mistral_eos_20261101/RESULTS.md) for native parity, the 94-test preflight and separate main-path and observer readouts.

| Component | Status | Claim ceiling |
|---|---|---|
| T fresh text relay | Implemented; CPU decoder smoke tested | No pretrained quality result |
| L fixed-slot recurrence | Implemented; uncalibrated identity transport tested | Tensor plumbing, not established semantic recurrence |
| B exact G → D → E ordering | Implemented and call-order tested | Same limitation as L |
| Plain / fixed expansion schedule | Implemented | Anti-cycle effectiveness not measured |
| Clone readout, isolated RNG, no persistent KV | Implemented and tested | Single-process invariant under supported hooks |
| One-time language branch | Implemented and tested | Norm-matched / continuous-compression shams not yet automated |
| Safe NPZ state restore | Implemented; fresh-process tiny replay tested | Pretrained / GPU exactness not verified |
| Activation collection | Cut/writer residuals, attention and MLP; MoE gate where present | No causal semantics assigned to activation size |
| Post-recurrence task readout | Implemented at baseline/final in CLI | 20 development fixtures only |
| Intermediate task schedule | Library supports saved states | Matrix dispatcher and intermediate CLI scheduling not implemented |
| Semantic qualification gate | Enforced for L/B `run` | No gate has passed; no trained bridge included |
| Ridge transport fit | Implemented primitive | Paired calibration collection/training protocol still required |
| Four HF family layouts | Optional random-config tests provided | Skipped in this environment because Transformers is absent |
| Actual Mistral/Llama/Gemma/Phi weights | Profiles and local-only loader | NOT RUN |
| Furnace / CUDA | Read-only inventory + guarded script | NOT RUN; historical hardware is not current inventory |
| Confirmation dataset / preregistration | Protocol proposal | NOT CREATED / NOT REGISTERED |
| Reasoning gain / useful novelty | Endpoints defined | NOT ESTABLISHED |
| Paper | Markdown + LaTeX problem/method draft, 13 references | No empirical results, no priority claim |
| Remote GitHub repository / CI | Publish helper / workflow provided | NOT CREATED / NOT EXECUTED by this session |

The candidate HF dependency range is not an environment lock. The Phi model cards include custom-code metadata; native Phimoe loading must be verified against the actual local config before any claim of compatibility. No automatic remote-code fallback is allowed. A hash-bound explicit reviewed-local-code route is implemented but has not been exercised on real custom model code. Phi-tiny has a confirmed attention-geometry mismatch with native Phimoe v4.57.1; see the runbook.

## Not implemented as a scientific capability

A trained semantic transport/verbalizer, adaptive history-triggered cycle controller, a robust semantic cycle classifier, norm-matched and low-rank sham suites, activation patching by validated semantic direction, broad benchmark evaluation, multi-seed clustered inference, and an automatic full-matrix runner remain explicit follow-up work. Their design in the paper/protocol must not be mistaken for completed experiments.
