# Fitted-lexical-reader channel: end-to-end endpoints (Experiment B)

**Status: development measurement, fully offline. Reads sealed write7
artifacts; no generation, no model, no refit of any boundary component except
the report→bit reader under test. Not a qualification run.**

## Question

`writer_parser_20261115` (Experiment A) measured the *ceiling* of what a
fitted lexical reader can extract from the spontaneous report corpus:
~0.62–0.71 per report in familiar wording, ~0.50 across topics, with the
decoder-unresolved mass provably containing no option-identifying content.
Experiment B converts that ceiling into **end-to-end boundary endpoints**:
if the best possible fitted reader still cannot run the channel, the
bottleneck is the writer's text, not the reader family — and write7's
under-powered `feat_addr` result is moot for this corpus.

## Method

Reuse the sealed `message_boundary_20261106/write7` pipeline verbatim:

- `fits.npz` → the three `hard_supplied_w1` controllers (shared-pointer
  convention: every channel is rolled through the *same* supplied-fitted
  controller per seed, isolating channel quality from per-channel refits).
- `test_native_hard/scores.npz` + `test_variants_hard_b100/variants.npz` →
  frozen native winner/margin tables for scoring.
- `join.npz` → donor indices and the 14 evaluation masks.
- `reports_rt_*.jsonl` → the sealed pair-level corpus.
- `rollout_rt` / `twins_w` from `selffold.mixture_boundary` — exhaustive
  38,416 four-edit strings over the 14 masks × 64 test sources; identical
  realized-table twins with intact/erased messages.

Members for the `hard` form are canonical one-hot rows; the rollout
dynamics depend only on member *indices*, never column identity, so
canonical members are exact (verified by reproducing the sealed decoder
endpoints bit-for-bit).

## Channel arms (per style, per seed 17/41/73)

| arm | report → bit | expectation |
| --- | --- | --- |
| `decoder` | sealed rowbound_v3 bits from `lm_matrix.npz` | must replicate write7: rt_directed 100%, rt_natural 13.3% |
| `oracle` | `out & 1` realized truth | supplied-convention ceiling, ~100% |
| `fitted_text` | word-TFIDF (1–2 g) + logistic regression, GroupKFold over **unique texts** (held-out exact strings) | Experiment A ceiling, familiar wording |
| `fitted_topic` | same reader, GroupKFold over **topic groups** | cross-topic generalization |
| `none` | all-zero messages | 12.5% floor |

## Pre-registered readings

- `fitted_text` ≫ `decoder` end-to-end ⇒ the reader family was the
  bottleneck; writer retains partial credit.
- `fitted_text` ≈ `none` ⇒ even the measured ceiling can't run the channel;
  the writer's spontaneous text is the binding constraint.
- Any arm *below* `none` indicates actively misleading messages (a reader
  worse than silence).
