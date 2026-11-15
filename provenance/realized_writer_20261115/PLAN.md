# Realized-state writer test — the registered P-tests under their registered conditions (Experiment C)

**Status: DESIGN_ONLY_NOT_EXECUTED — blocked on the pinned checkpoint.
`f3503907…` (gemma-2-2b-it) lives on SpiralReality-furnace, unreachable at
design time (`spiral status`: host timeout). This document fixes the deltas
vs `write3`–`write7` so the run is executable as-written when the node
returns. No artifacts generated.**

## What is still unexecuted

After the write4→write6 repairs, the registered S-WRITE test
(`docs/STATE_WRITER_PLAN.md`) still lacks three conditions simultaneously:

1. **Writer input = the controller's own emitted W.** Reports must be
   rendered from the actual member-space weights the controller produced —
   including pointer errors on-policy — not from a lookup of the realized
   triple. write6 keyed reports by realized `(prior, mask, out)` but still
   generated them off-policy over a complete cache; the registered test
   asks whether the writer describes *its own* trajectory, including
   states reached only through its own earlier errors.
2. **Pole-free decoder.** The registered decode is
   *"does the report assert off/off → member1"* with **no role mapping
   handed in** (`slot ^ pole` is declared carried side information, which
   the registered test excludes). The decoder must identify member1 by its
   role in the prompt (default vs exception wording), not via `pole`.
3. **P1–P4 verdicts.** write3 explicitly marked them not executed under
   registered conditions; write4–7 measurements, however close in
   mechanics, do not discharge the registration.

## Execution design (fixed)

- **On-policy report generation**: run `rollout_rt`-equivalent with the
  writer in the loop — at each edit, render `realize(W)` (hard members) as
  the post-update table, plus the applied mask, plus optionally the
  realized prior table; generate the report; decode it pole-free; feed the
  bit as the next message. Coverage is on-policy only: trajectories that
  leave the feasible bound are impossible by construction (every pick
  lands on a feasible pattern).
- **Channels**: `state_directed`, `state_natural`, `state_prior_true`
  (realized prior), `supplied`, `reward`, `none` — plus optional
  `rt_directed` reuse from the sealed corpus for the off-policy vs
  on-policy contrast.
- **Decoder**: new `asserts_member1(text, off/off, member1_name)` —
  keyword/row-binding decode with no `pole` argument; misses → bit 0
  (declared fallback, same convention as rowbound_v4).
- **Controller**: shared supplied-fitted pointer per seed (the write7
  isolation convention), then a per-channel refit block.
- **Endpoints**: identical to write6/7 — all-four, per-depth, twins
  intact/erased, opposed reversal, written-vs-convention agreement, and
  now `prior_mismatch` (already instrumented).
- **Preregistered readings** (from STATE_WRITER_PLAN): P1
  `state_directed` ≈ `supplied`; P2 `state_natural` ≥ `lm_natural` <
  guided; P3 `state_prior_true` > `state_natural`; P4 = the writer-input
  sensitivity delta (`state_*` vs `rt_*`/`lm_*` at matched decoder).

## Inputs this design now has that write3 lacked

- Experiment A (`writer_parser_20261115`): a measured per-report lexical
  ceiling (~0.62–0.71 familiar / ~0.50 cross-topic) and the finding that
  ≥53% of spontaneous reports contain no option-identifying content —
  predicts P2/P3's spontaneous arms remain near floor **unless the
  realized-state prompt elicits option naming**.
- Experiment B (`fitted_channel_20261115`): fitted readers at ceiling
  still cannot run the channel (≤12.7% all-four) — the writer is the
  binding constraint, which is exactly what P4 measures.
- Experiment D (`binding_diagnostic_20261115`): fresh-wording field
  failures are single directional lamp confusions (`right-on`→`left-on`),
  one style — informs decoder-free wording sensitivity on C.

## Blocker and resume condition

- Requires `scripts/fit_lm_writer.py`-class host: pinned checkpoint
  `f3503907…`, `kv_capped_argmax` policy (parity 25/30 declared), budgets
  ~1.5M forwards. Run on SpiralReality-furnace when reachable; the
  on-policy rollout is ~4× the write6 generation surface at worst.
- Before execution: freeze source commit, write `request.json`-style
  bindings, and open a fresh `produce/` dir under
  `provenance/message_boundary_20261106/write8/` (same parent, new arm
  set) or under this dir if the provenance convention prefers a new date.
