# Writer-vs-parser separation: an extractable-information ceiling

**Development measurement, executed locally (CPU, sklearn 1.8). Reads only the
sealed `message_boundary_20261106/write6` corpora; no generation, no boundary
fitting, no qualification inputs. [Plan](PLAN.md) · [code](analyze.py) ·
[scores](summary.json) · [annotation](annotation.jsonl)**

## Headline

The spontaneous reports carry the off/off bit in **wording-memorized, not
topic-general, form**: a supervised lexical reader recovers the realized slot
at 0.68–0.71 within familiar wording and 0.59–0.62 on held-out exact strings,
but **collapses to chance (~0.49–0.50) on held-out topics**. And in the mass
the shipped decoders leave entirely unresolved (53% of rt_natural), an
independent annotation of a stratified ≥80% coverage sample finds **no
option-identifying content at all in 99.7% of it**.

So both halves of the undecided comparison are now measured separately:

- **Writer failure is real and dominant on the unresolved mass.** More than
  half of spontaneous reports say only "the <topic> rules were updated" — the
  bit is absent from the text for *any* reader.
- **A parser gap also exists.** The shipped decoder family reached ~34%
  consensus-correct on rt_natural; a generic lexical reader gets 0.59–0.62 on
  unseen strings of familiar forms — roughly half the remaining signal the
  writer did emit. `state_prior_true` shows the same shape (0.64–0.66 held-text
  vs ~26% decoder-consensus).

Neither result rescues the spontaneous channel end-to-end: even the loose
upper bound (0.68–0.85 per-report) cannot produce the required role history
across four edits on this bank, and nothing transfers across topics.

## Ceiling numbers (random / group-text / group-topic splits, slot accuracy)

| corpus | target | word_lr | char_lr | word_gb | null p95 |
| --- | --- | --- | --- | --- | --- |
| rt_natural | slot_out | .684 / .619 / .494 | .667 / .611 / .492 | .708 / .593 / .502 | .510 |
| rt_natural | slot_mask | .628 / .545 / .492 | .593 / .544 / .481 | .652 / .529 / .501 | .508 |
| rt_directed | slot_out | **.891** / .394 / .500 | .891 / .438 / .500 | .890 / .323 / .500 | .509 |
| state_prior_true | slot_out=slot_mask | .823 / .656 / .452 | .742 / .584 / .476 | .854 / .649 / .473 | .508 |

Notes:

- **rt_directed slot_out = 0.891 exactly reproduces the write6 audit's 89%
  consensus-correct** — the pipeline is calibrated (sanity arm passed; on
  realized divergence directed reports name the instruction's slot, acc_diff
  0.78 vs acc_same 1.00, matching the audit's 78%).
- `state_prior_true` slot_out ≡ slot_mask because the corpus never had a
  realized/requested divergence in these records (div=0.000).
- `group-topic` at ~0.5 **includes** rt_directed: an option name is itself a
  topic-specific string, so "names the option" does not transfer by this
  lexical route. Topic-general recovery would need a semantic code (ordinal
  or polarity), which the writers essentially never produced.

## Independent annotation

58 normalized forms covering 5,612 of the 6,936 all-decoder-unresolved
rt_natural reports (≥80% of that mass + 10 tail samples). Single annotator
(the agent, sharing no code with the decoder family); verdicts recorded per
form in `annotation.jsonl`.

| verdict | forms | covered reports |
| --- | ---: | ---: |
| bit_absent | 55 | 5,598 |
| bit_ambiguous ("…were reversed") | 3 | 14 |
| bit_determinable | 0 | 0 |

Typical absent forms: "The state machine's rules for music type have been
updated." — names the topic domain only. "…rules for handling 'on/off' and
'off/on' instructions were updated" names coordinates other than off/off and
still no option. The three "reversed" forms assert a switch without binding
the referent; under a maximally charitable constant decode they could supply
one bit-value only.

## Readings

1. On this prompt/bank the spontaneous writer mostly **does not write the
   bit** (~53% of mass provably contains nothing; the residual carries
   wording-specific partial signal).
2. The shipped decoders additionally **miss roughly half the signal that is
   present** in familiar wording (0.62 vs 0.34) — a fitted lexical reader is
   a justified Experiment-B target, but its ceiling here is ~0.6–0.7
   per-report, far below what a working channel needs.
3. **No topic-general code exists in spontaneous text**: cross-topic transfer
   is chance for every corpus, including directed. If a future writer is to
   carry the bit to unseen topics, the code must be semantic (ordinals,
   polarity markers, explicit option naming), not lexical habit.
4. Stage 2 (frozen-E probe) is now *conditional on whether a semantic route
   is being tested*: for pure lexical content the n-gram ceiling already
   bounds it; E-features could only add non-lexical signal, which the
   annotation suggests is absent.

## Limits

- The ceiling is an upper bound on decoders, not writer fidelity: random-split
  accuracy partly reflects per-form majority memorization (that is why
  group-text and group-topic are reported).
- Single annotator; verdicts are per-form and not adjudicated. The annotation
  file keeps every form + count for review.
- Measurement applies to write6's generation policy and prompt bank only.
