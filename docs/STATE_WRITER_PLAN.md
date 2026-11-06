# S-WRITE: realized-state writer experiment (DESIGN_ONLY)

Follow-up to W-RITE (PR #32) addressing its main bound: the W-RITE writer read
an *idealized* assignment table constructed from (source, pole, mask), not the
controller's realized post-update state, and the decoder was handed the role
mapping. This design fixes both.

## Question

Vary **only the report channel** while holding (a) the realized boundary state
and (b) the downstream pointer family fixed. If performance differences
survive, the channel — not the state or the reader — is responsible, and the
self-verbalization claim becomes causal rather than correlational.

## Design

- **State source**: the controller's actual post-update W (member-space
  weights) → rendered to the writer as a table of *realized* rows. For hard
  boundaries rows are named actions; the writer sees exactly what the pointer
  sees.
- **Writer input**: realized table + the applied instruction (mask). Same
  prompt family as W-RITE, same styles (`directed`, `guided`, `natural`), plus
  a `prior` arm that also shows the pre-update table so "what changed" has a
  referent (fixes the W-RITE natural/guided omission).
- **Decoder**: unchanged scored-keyword decode, but restricted to the off/off
  row only — no role mapping handed in. Bit = does the report assert
  off/off → member1?
- **Arms** (hard boundary, β=1.0, 3 seeds): `state_directed`, `state_guided`,
  `state_natural`, `state_prior_natural` + ceilings `supplied`, `reward`, `none`.
  One pointer training regime shared across all channels (matched controls);
  a per-channel refit as a second block if compute allows.
- **Key comparisons**
  1. `state_directed` vs `supplied`: does text-over-realized-state still
     reach the symbol ceiling? (W-RITE predicted yes; this tests it without
     the idealized input.)
  2. `state_natural` vs `state_prior_natural`: does showing the pre-change
     table repair the "no referent" deficit?
  3. `state_*` vs W-RITE `lm_*`: idealized vs realized writer input, matched
     decoder — the first clean writer-input ablation.
- **Honesty gates**: twin and opposed-donor reversal with margin sign flip
  (corrected axes), erasure control, missing rates by topic (topic-2 verb
  answers known), message telemetry = actually-carried bits only.

## Predictions (preregistered here)

- P1: `state_directed` ≈ `supplied` (directed D is a lossless 1-bit channel
  even on realized state).
- P2: `state_natural` ≥ W-RITE `lm_natural` (prior context can only help
  referent selection) but < guided.
- P3: `state_prior_natural` > `state_natural` (referent repair).
- P4: if any `state_*` gap vs `lm_*` appears, it isolates writer-input
  sensitivity — the first real measure of "does the model describe its own
  state or a script of it".

## Non-goals

No qualitative claim about real language use; the prompts remain templated.
No qualification inputs. Evidence dir: `write3/` under the same provenance
parent.
