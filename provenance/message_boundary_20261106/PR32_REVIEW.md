# PR #32 review and repair record

Reviewed head: `b2af0b5187befca2f730eafe41ca960332b6a412`.
This review changes implementation and interpretation; it does not rerun Gemma,
retroactively seal a protocol, or overwrite any archived experiment artifact.

## Findings and repairs

1. **Twin axes (high priority).** `twins_w` stacked four history/erasure arrays
   after the world axis and reshaped without transposing. This mixes conditions
   with worlds. Stack before the world axis instead. The reducer now explicitly
   inverts the legacy layout and marks that repair in a v2 summary. The initial
   `message_boundary.twins_boundary` is a different producer and is not affected.
2. **Actual message telemetry.** `eval_step` returned sampled controller bits
   even for the LM, supplied and no-message channels. Those are not what was
   transmitted. Record carried bits and a `carried_message_bits_v2` marker.
   Legacy summaries label the old statistic as sampled-writer agreement, not
   message agreement. LM messages/matrices now fail on missing, malformed or
   nonbinary inputs. The twin's discarded next message is explicitly supplied.
3. **Unbound and partial caches.** A style-only JSONL cache could survive a
   checkpoint, prompt or source-bank change. An interrupted cache silently left
   unobserved messages at zero. The v3 cache binds effective host identity,
   source code, each exact prompt, source key/options/polarity and requested mask;
   validates complete unique coverage and content hashes; publishes atomically;
   and preserves interrupted generations separately. Old caches are not migrated.
4. **Incomplete evidence.** The guided matrix was not serialized. All active
   styles now are. Unvisited columns contain -1, not apparently valid bits.
   Rates use only sampled masks. Generation receipts retain actual stop reason,
   censoring and token counts. `host.json` now reflects the effective 48-token
   cap rather than the pre-override EOS-only setting.
5. **Decoder matching.** Character-set stripping removed legitimate leading or
   trailing `s`, substring matches could read `east` inside `least`, and exact
   phrases were case-sensitive. New runs use token boundaries and case folding.
   This remains a lexical heuristic, not a semantic or row-aware parser. Changing
   it does not silently replace historical classifications or scores.
6. **Reducer bindings.** Multiple native-label files must agree. Hard descriptors
   must be exactly binary. Missing or conflicting data fail explicitly.
7. **Claim boundary.** Writer prompts use ideal assignments and the decoder uses
   the source role mapping; later messages are source/mask lookups. Controllers
   are fitted separately by channel. No previous table is supplied to the
   "what changed" prompts. The text cannot support intrinsic capacity ceilings,
   writer-only causal loss, a saturation claim, or autonomous self-verbalization.
   The revised narrative retains the reported end-to-end rates with that scope.

## Validation performed

- Before repair, the new boundary regression file had **10 failures and 2 passes**
  against the reviewed module, including the twin axis and actual-message errors.
- After repair, **44 focused CPU tests passed**, including an end-to-end synthetic
  native-score reduction, legacy-layout repair, exact supplied/LM convention
  agreement, message validation, cached-prompt binding, interruption, duplication,
  content corruption, all-style persistence and sampled-mask denominators.
- The reviewed `lm_writer.py`, runner, mixture module and original reducer were
  copied from the pinned connector contents and matched their Git blob hashes
  before editing. Local focused tests used the unchanged dependency definitions
  needed from `message_boundary`, `role_memory` and `role_trials`; repository-wide
  integration is delegated to the PR CI, not claimed as a local full-suite run.
- Python compilation passed for all changed runtime modules and the runner.
- No GPU execution, native-logit replay, full report reannotation, or regeneration
  of the archived all-four outcomes was performed. `write*/produce/trajectories.npz`
  was not available from the published tree. This limits numerical validation.

## Additive replay, not historical overwrite

Once the matching trajectory file is recovered and its digest is checked against
that run's `rollout_closed.json`, the repaired reducer can produce a NEW file:

```bash
python provenance/message_boundary_20261106/summarize_mixture.py \
  provenance/message_boundary_20261106/write2/produce \
  > /tmp/write2-summary-review-v2.json
```

This repairs legacy twin layout without new model calls. It labels old telemetry
as sampled-writer data. It does not reconstruct a missing guided matrix or invent
cache provenance. Compare the all-four counts separately from corrected twin
counts; do not replace `write2/summary.json` with the output.

Future executions of `scripts/fit_lm_writer.py` require a fresh output directory
and use `local/lm_reports_v3`. New code, decoder and generation records define a
new development run, not a replay of write/write2. For a writer-only comparison,
freeze the same downstream receiver and vary only the report path on an actual
shared state. Supplying before/after tables or a realized-tensor renderer is a
separately declared experiment, not a silent repair to the old prompt.

The useful remaining result is a bounded supplied-table/report/decoder/controller
comparison. Qualification and the original common-state T/L/B intervention remain
unopened by this review. The branch is left unmerged.
