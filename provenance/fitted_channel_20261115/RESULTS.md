# Fitted-lexical-reader channel: end-to-end endpoints

**Development measurement, executed locally (CPU). Sealed write7 controllers,
scores, corpus; rollout code unmodified. [Plan](PLAN.md) ·
[code](run_channels.py) · [scores](summary.json)**

## Sanity: exact replication of the sealed run

| arm (rt_natural) | all-four | twins intact |
| --- | ---: | ---: |
| decoder (stored rowbound bits) | 326,704 / 2,458,624 = **13.29%** | 8,064 / 50,176 |
| none | 307,328 = **12.50%** | 0 |
| oracle (out & 1) | 2,458,624 = **100%** | 50,176 / 50,176 |
| write7 recorded | 13.3% / 12.5% / 100% | 0.161 / 0 / 1.000 |

`rt_directed` decoder = 100% all-four, 50,176/50,176 twins — identical to the
sealed run on every endpoint and every seed. The offline harness is exact.

## New endpoints (all-four / twins-intact, identical across 3 seeds)

| channel | rt_natural | rt_directed |
| --- | --- | --- |
| oracle (perfect writer) | 100% / 1.000 | 100% / 1.000 |
| decoder (rowbound) | 13.3% / 0.161 | 100% / 1.000 |
| **fitted_text** (A's ceiling, held-out strings) | **12.1% / 0.196** | **9.9% / 0.027** |
| **fitted_topic** (cross-topic reader) | **12.7% / 0.116** | 12.5% / 0.000 |
| none | 12.5% / 0.000 | 12.5% / 0.000 |

Per-depth native accuracy shows the failure mode: `fitted_text` on rt_natural
is 1.0 / 0.531 / 0.482 / 0.487 vs decoder's 1.0 / 0.540 / 0.498 / 0.494 and
none's flat 1.0 / 0.5 / 0.5 / 0.5 — the fitted reader moves real per-edit
signal but not enough to complete trajectories.

## Readings

1. **The ceiling does not run the channel.** Experiment A's best reachable
   reader (~0.62–0.68 per report, familiar wording) yields 9.9–12.1%
   all-four — at or below the 12.5% no-message floor. Even doubling the
   decoders' extracted signal cannot carry four-edit role history, because
   ~53% of reports provably contain no bit content at all.
2. **Wrong information is worse than silence.** `fitted_text` on rt_directed
   scores *below* `none` (9.9% < 12.5%) and its twins separation collapses
   (0.027 vs decoder's 1.000): consistent, confidently-wrong reads actively
   steer the controller off the never-flipped safe path. A reader with
   abstention would be needed for any real deployment — none of the fitted
   readers abstains.
3. **Cross-topic channels carry nothing.** `fitted_topic` ≡ `none` on
   rt_directed and stays at floor-level on rt_natural (the 0.116 twins are
   topic-leaked through seen wording in the CV, not a transferable code).
   There is no wording-independent bit in this corpus, confirming A's
   group-topic chance result end-to-end.
4. **Write7's `lmtext` failure is now explained, not just observed.** The
   hashbag/`feat_addr` arm scored 0% all-four; this run shows even the
   optimal lexical reader scores ≤12.7% — the binding constraint on this
   corpus is what the writer emitted (~53% empty reports, rest
   wording-memorized), not the feature representation or the 512-update
   budget. Richer features can at most approach `fitted_text`.
5. **The remaining lever is the writer.** Oracle = 100% proves the channel
   mechanics are sufficient. To carry the bit through spontaneous text, the
   *writer* must produce a semantic code (named options, ordinals, polarity)
   — exactly what directed prompts elicit and natural prompts do not. This
   is the direct motivation for Experiment C's realized-state writer test
   and for any prompt-side elicitation arm.

## Limits

- Fitted readers are one family (word-TFIDF + LR); A's other estimators
  bound it within ~4 points. No reader can exceed the per-report ceiling,
  so the conclusion is structural, not estimator-specific.
- Canonical members are exact for `hard` form only (verified by the
  decoder/oracle replication); soft/mixture arms would need real member
  rows from the sealed bank.
- All endpoints are this bank, this interface, these prompts — no claim
  about other elicitation regimes.
