# Realized-state writer test — EXECUTED (the registered S-WRITE conditions)

**Sealed run: `provenance/realized_writer_20261115/produce/` — REALIZED_WRITER_CLOSED
on the pinned host** (`f3503907…`, gemma-2-2b-it, `kv_capped_argmax`, hard form,
β=1.0, seeds {17,41,73}). Execution note: generation ran on SpiralReality-furnace
across three tmux-launched attempts; reports persist in `reports_rw_*.jsonl` and
are memoized per realized triple, so the completed corpus is the union of all
attempts (every record carries its generation receipt).

## What makes this the registered test (vs write3–7)

- **Writer input = the controller's own realized transition**: at each edit the
  rollout renders the actual prior table (including pointer-error states), the
  applied instruction, and the realized post-update table, and generates the
  report on-policy inside the rollout (`rollout_realized`, lazy memoized).
- **Pole-free decoder**: `asserts_member1` decides "does the report bind
  off/off → member1" using only the two action names. member1 is identified as
  the action the protocol-fixed initial table (mask 8, `[m0,m0,m0,m1]`) assigns
  to on/on — no role mapping is handed in. Misses fall back to bit 0 (the
  declared no-carrier convention).
- **`rw_directed` asks about the realized row** ("which action does the off/off
  row now select"), not the instruction's requested assignment — this is the
  realized-vs-requested distinction write6 could only approximate.
- **Single pointer regime**: every report channel rolls through the SAME
  supplied-fitted controller per seed; supplied/reward/none ceilings keep their
  own actors.

## Headline numbers (all-four trajectories, hard form)

| Arm | 17 | 41 | 73 | Twins intact | Twins erased |
|---|---|---|---|---|---|
| rw_directed (shared ptr) | 100.00% | 100.00% | 100.00% | 50176/50176 | 0/50176 |
| rw_natural (shared ptr) | 13.17% | 13.17% | 13.17% | 8064/50176 | 0/50176 |
| rw_bare (no prior table) | 22.58% | 22.58% | 22.58% | 9408/50176 | 0/50176 |
| supplied ceiling | 100% | 100% | 100% | 50176/50176 | 0/50176 |
| reward ceiling | 100% | 100% | 100% | 50176/50176 | 0/50176 |
| none floor | 12.50% | 12.50% | 12.50% | 0 | 0 |

Decoder miss rates (pole-free member1 assertion): `rw_directed` 0/4704,
`rw_natural` 67.1% (3222/4802), `rw_bare` 62.0% (2446/3948). Missing is
spread across all eight topic groups (no single-topic concentration).

## Registered predictions

- **P1 — state_directed ≈ supplied: CONFIRMED.** Directed reports on the
  realized table transmit the bit losslessly under the pole-free decoder:
  100% all-four, identical to the supplied-symbol ceiling, all three seeds.
- **P2 — state_natural ≥ lm_natural but below guided: CONFIRMED.**
  `rw_natural` 13.17% clears the W-RITE `lm_natural` 7.3% (write6) and
  matches `rt_natural` 13.3% (write7), far below the directed arm.
- **P3 — prior-referent improves natural: REJECTED — direction reversed.**
  Removing the prior table (`rw_bare`) *increases* all-four to 22.58%
  from 13.17%. The prior-referent prompt that asks "what changed" makes
  reports describe the edit diffusely; without the prior the writer more
  often names the current off/off action the decoder can bind.
- **P4 — writer-input sensitivity: measured.** At the realized input the
  directed channel is now fully lossless where write6's requested-state
  directed question left an ~11% lexical gap (89% corpus ceiling). The
  residual natural-arm gap is omission, not misdescription (67% missing).

## Interpretation bounds

- The channel now provably carries the member1 bit under realized input:
  **the earlier "writer vs parser" ambiguity is resolved toward omission** —
  when the report names the off/off action at all, the pole-free decoder
  binds it correctly; the failures are reports that assert nothing about
  off/off (consistent with Experiment A's 99.7% bit_absent annotation).
- `rw_bare > rw_natural` is a prompt-effect, not a model capability claim:
  the bare prompt removes a distractor, not an information source.
- This is a development measurement on the sealed dev split. It does NOT
  discharge the semantic-qualification challenge; the 24-group independent
  panel remains unspent.

## Provenance notes

- Two interrupted attempts (an IndexError in test-pair indexing, a silent
  mid-run kill) left no corrupted records: every report row is written and
  fsync-appended at generation time, and per-arm checkpoints (`arm_*.json`)
  record completed rollouts. The final closed run is produce_run3 on
  furnace; its reports accumulate in `reports_rw_*.jsonl` here.
- `trajectories.npz` (108 MB) and per-arm `rolled_*.npz` remain on furnace
  under `runs/realized_writer/` — unpublished, available on request.
