# Reading content from eight hidden-state slots

**Question.** Two donors can have different tensor fingerprints and still generate the same words. Can a separate question recover which description supplied the slots? Seven offline GPU runs examined this question on Mistral-7B-Instruct-v0.3 and Gemma-2-2B-IT. Both used greedy BF16, a 1024-token context, eight slots and the uncalibrated identity transport. Mistral used the input to layer 16 and output of layer 31, and Gemma used layers 13 and 25 (zero-indexed). [Initial plan](PLAN.md) · [Lexical control plan](ALIGNMENT_PLAN.md) · [Replay plan](REPLAY_PLAN.md)

## 1. Re-reading the original Mistral descriptions

We restored the two source descriptions, their encodings and the L/B states from both original EOS updates. For each donor, we asked which decision priority the speaker favored, scored the two answer labels, reversed their order and averaged the two differences toward revisability. Positive margins favor revisability and negative margins favor prompt decisions. The question never entered recurrence. [Archived donor identities](trials/archive-mistral/archive_sources.json) · [Raw scores in both orders](trials/archive-mistral/scores.jsonl)

| Source supplied to the question | Revisability donor, nats | Prompt decision donor, nats |
| --- | ---: | ---: |
| Literal description | +16.766 | −16.344 |
| Initial slots E | +2.844 | +2.844 |
| L after one update | +2.438 | +2.500 |
| L after two updates | +3.031 | +3.000 |
| B after either update | +2.813 | +2.813 |

Both latent donors favored the same answer at every stage in this one previously studied pair.

## 2. Six development contrasts on two checkpoints

We wrote six topics with two opposing preferences and two phrasings per preference. They concern archive order, walking routes, meeting notes, temperature units, shelving and directions. The literal description, E, one plain update G, and B were scored on the same question within each topic. Two additional G donors reversed the slot rows or retained only each row's magnitude on a fixed direction. Uninjected placeholders and zero vectors supplied two null scores per topic. Each model contributed 156 state/question measurements. [Fixtures](../../data/slot_contrasts.dev.json) · [Mistral likelihoods](trials/contrasts-mistral/scores.jsonl) · [Gemma likelihoods](trials/contrasts-gemma/scores.jsonl)

| Route | Mistral ordered / tied pairs (of 12) | Mistral correct donor signs (of 24) | Gemma ordered / tied pairs (of 12) | Gemma correct donor signs (of 24) |
| --- | ---: | ---: | ---: | ---: |
| Literal text | 12 / 0 | 24 | 12 / 0 | 24 |
| E | 6 / 2 | 12 | 5 / 1 | 12 |
| G | 3 / 0 | 12 | 7 / 1 | 11 |
| B | 0 / 12 | 12 | 0 / 12 | 12 |
| G, rows reversed | 5 / 2 | 12 | 5 / 0 | 13 |
| G, magnitudes on fixed directions | 9 / 0 | 12 | 3 / 2 | 12 |

A pair is ordered when the plus donor has a strictly larger plus margin than its matched minus donor. Correct donor signs require the margin to favor the donor's own option. Neither E nor G placed both donors on the correct sides of zero for any of the 12 pairs on either model. Gemma G had two zero donor margins. Small rank differences include floating-point residuals. The six topic clusters, paired margins, nulls, paraphrase differences and ties are available in the [machine-readable summary](summary.json).

Each model had 24 distinct E tensors, 24 distinct G tensors and one B tensor after re-encoding. All 50 free generations per model ended at native EOS. Within each checkpoint and each E/G stage, all 24 descriptions yielded a single string. Mistral's G readout referred to contents of work areas. Gemma's G readout requested a description to summarize, exactly matching its zero-state utterance. All B donors were available for scoring. [Mistral generations](trials/contrasts-mistral/generations.jsonl) · [Gemma generations](trials/contrasts-gemma/generations.jsonl)

## 3. A lexical positive control

We placed the words of one option directly in eight native token positions, padding with the neutral period id. The phrases took 2–6 tokens in Mistral and 2–5 in Gemma. These deliberately explicit donors reuse the vocabulary of the question. We compared native presentation, the ordinary encoder on the same words, a position-matched prefix donor, and its reversed rows. Each checkpoint produced 60 state/question measurements. [Mistral conditions and scores](trials/alignment-mistral/scores.jsonl) · [Gemma conditions and scores](trials/alignment-gemma/scores.jsonl)

| Route | Mistral ordered / 6 | Mistral correct / 12 | Gemma ordered / 6 | Gemma correct / 12 |
| --- | ---: | ---: | ---: | ---: |
| Native option words | 6 | 11 | 6 | 12 |
| E on the identical words | 4 | 6 | 3 | 6 |
| Eight donor rows from a short prefix | 3 | 6 | 4 | 7 |
| Same donor, reversed rows | 3 | 6 | 4 | 7 |

Putting back a neutral donor from a **shorter forward** changed the label log probabilities by up to 0.1903 nats on Mistral and 0.1613 on Gemma. The next experiment was fixed after this observation. It reuses the same material.

## 4. Matching sequence length and changing the transplanted positions

With the donor forward padded to the recipient's full sequence length, all post-slot donor tokens replaced by a neutral id, neutral replay matched the original at every vocabulary logit for all 12 question/order combinations on each checkpoint. The maximum discrepancy with the short prefix was 0.25 logits on Mistral and 0.50 on Gemma. The full-length opaque donor's eight slots also matched the corresponding native lexical donor exactly for all 24 donor/orders per checkpoint. The earlier native label likelihoods were reproduced exactly. The underlying numerical kernel responsible for the short-prefix discrepancy was not isolated. [Mistral neutral control](trials/replay-mistral/neutral.jsonl) · [Gemma neutral control](trials/replay-gemma/neutral.jsonl)

| Positions supplied to the neutral recipient at the middle layer | Mistral correct / 12 | Gemma correct / 12 | Mistral / Gemma median signed pole gap (nats) |
| --- | ---: | ---: | ---: |
| Native words throughout (reference) | 11 | 12 | 28.219 / 19.672 |
| Eight slots only, matched length | 6 | 7 | 0.008 / 0.219 |
| Positions after the slots through the scoring position | 11 | 12 | 27.594 / 19.734 |
| Slots and all those downstream positions | 11 | 12 | 28.219 / 19.672 |

The downstream-only transplant reproduced the native **donor-level preference sign in 12/12 cases on each checkpoint**, including Mistral's original error on the author-shelving donor. Its full logit vector could still differ. Transplanting both regions reconstructed the complete native scoring-position logit vector exactly in 24/24 donor/orders per model. Downstream states come from a forward that already saw the real question and options and span more positions. This demonstrates sufficiency for the specified single-forward intervention. It does not establish unique information storage or a portable semantic state. [Mistral interventions](trials/replay-mistral/interventions.jsonl) · [Gemma interventions](trials/replay-gemma/interventions.jsonl)

## Reproduce and inspect

The archive and first content probes used `a714567a1e5d4c7931ae99a10d01633a57123491`, the lexical controls `eac5155eaa64fc36eb0219a0bda56a8e538c3103`, and the replay controls `1d36e662df170359b36504051f836677df0e7245`. All seven actual worktrees were clean. All used the same RTX 5090 and passed native hook parity. Exact source and checkpoint hashes, environment information, label scores, and terminal counts are in [the trial directories](trials/). The control data are development-only. Human semantic judgments, long trajectories and held-out reasoning gains remain open.

To check the published files and reproduce the tables from label likelihoods, run the following from this directory. [Verification record](verification.json) · [File digests](SHA256SUMS)

~~~bash
shasum -a 256 -c SHA256SUMS
python3 summarize.py | diff - summary.json
~~~
