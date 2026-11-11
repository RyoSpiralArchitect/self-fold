# Start-address repair — saved-report refit (write8)

Prospective development protocol, fixed before controller fitting on 2026-11-10.
This follow-up preserves write6 and write7, including their first outcomes.

## Question and intervention

Write7 initializes the text channel with a zero vector and applies a bias-free
projection. Its first greedy row pointers necessarily tie and both pick row 0,
erasing the other action before a report can help. Longer fitting cannot fix
that zero-input address. Its nonzero constructed-twin scores bypass the first
step and do not validate the full trajectory.

Use the existing trainable integer-message address `address[0]` before any
report exists. Both text and discrete arms now start at that address. Later
text messages retain the original `hashbag_v1` features and `feat_addr`
projection, including when a report is erased to a zero feature vector. No
parameters, target bits, action names, future instructions or extra source
text are added to the initial message. The existing two address coefficients
can now learn from the same outcome reward.

## Fixed evidence and budget

- Base configuration is byte-identical `data/state_writer_20261109.json`:
  eight arms, seeds 17/41/73, 512 updates, batch 128, Adam learning rate 0.06,
  exploration floor 0.12, norm clip 1, final checkpoint only.
- Restore the 144 fitting and 64 evaluation source realizations from the
  archived hard-row call hashes, verifying float32 one-hot bytes. Do not
  reconstruct rows from expected policy labels. Use their saved full-vocabulary
  Gemma winners and A/B label bindings for exactly the same lookup rewards.
- Reuse all sealed write7 reports and stop receipts. Restore the same request,
  checkpoint, source, decoder and corpus hashes; preserve censored reports.
  Recompute only deterministic `hashbag_v1` features. No text regeneration,
  source recapture, new native calls or qualification data.
- The fitting masks are `[8,7,14,1,2,13,4,11]`; the other masks are
  `[6,9,10,5,12,3]`. Only fitting topics 0/2/4/6 and fitting masks enter fitting.
- First replay all 24 archived fitted controllers with their executed start
  semantics. Require exact agreement with write7's published native depth,
  complete-history, role-exchange, donor and twin counts before refitting.
- Refit all eight arms under the unchanged random schedules. Unmodified
  discrete-control fit hashes must agree exactly with write7. If they do not,
  preserve the technical stop and diagnose the environment before attributing
  an effect to the text-start repair.
- CPU execution on Furnace, one Torch thread per process, at most two
  evaluation workers, one-hour wall budget. No seed, feature, parser or
  learning-budget search. Keep every first technical stop; do not overwrite
  or silently resume an incomplete result.

## Measurements and stopping conditions

The full panel is 64 sources × 14^4 instruction strings. Four updates, four
worlds and both answer orders must all be correct for a complete history.
These are repeated decisions over a small known policy bank, not independent
generalization trials. Report every seed separately, all-four correctness,
native correctness by depth, donor/erasure controls, exact twins, and the
first-update completeness over the 64 × 14 distinct source/mask units.
Also report the 6^4 strings containing only unfitted masks and per-topic counts.

Engineering acceptance requires restored cache-free host fallback, a
mutation-sensitive actual-output lookup test, a nonzero start-address
gradient, and a perfect feature-channel fixture that completes all four edits.
Scientific interpretation uses the measured text-channel differences against
archived write7 and the unchanged supplied/reward/none controls. Restoring
the first update alone is not evidence that natural reports support full
recurrence or that the writer/reader ambiguity has been resolved.

The bounded experiment is complete after exact archived replay, the 24 fixed
refits, and all predeclared evaluations. If natural or directed text features
remain weak, close with those numbers and the remaining factor; do not expand
training until a new experiment is specified. No original six-check
qualification or independent human-review claim follows from this study.

## Durable evidence

`INPUTS.json` pins every file in write7 plus its configuration and action bank.
Write new evidence only under `write8/produce`: source/request/environment,
restored-input receipts, all fitting curves/weights, archived and refitted
summaries, and separate compact trajectory files. The four binary world rows
are losslessly packed as a four-bit pattern; the full carried report payload
has a saved hash and is reconstructible from the bound report corpus. Preserve
the original files byte-for-byte. A fresh-process reducer must regenerate the
published new summaries from the compact trajectories.

Reproduce from a clean sealed checkout:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 scripts/replay_lm_writer.py --out <new-output-directory> --workers 2
```
