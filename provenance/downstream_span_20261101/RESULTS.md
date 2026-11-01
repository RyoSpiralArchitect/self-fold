# Localizing the downstream lexical readout

Which part of the question-conditioned span can reproduce a lexical donor's preference? The earlier replay moved every residual after the eight slots, including the scoring position. Here the recipient always sees the real question and its two labeled alternatives. The native donor supplies the words of one alternative at the eight input positions. At the input to layer 16 of Mistral-7B-Instruct-v0.3 or layer 13 of Gemma-2-2B-IT, we change which donor positions enter one neutral forward.

The study used the original six topics and six newly written topics about calendars, paint, photographs, packing, recipes and music. The primary interventions and both panels were fixed before collecting localization scores. The clause follow-up was fixed after seeing the original Mistral localization and before inspecting the new-panel outcomes. All material uses the same explicit lexical task. [Primary plan](PLAN.md) · [Clause follow-up plan](CLAUSE_PLAN.md) · [New topics](../../data/span_topics.prospective.json)

## How to read the numbers

For each donor, score both one-token labels, reverse the option assignments and average the two signed log-probability margins toward the same option. For a topic, subtract the minus donor's margin from the plus donor's margin. Tables below give **100 × the sum of these six pole gaps divided by the corresponding sum under whole-downstream transfer**. Each panel and checkpoint has its own denominator. A negative entry means the aggregate gap reversed. Large reference gaps contribute more to the ratio. Entries are rounded to one decimal place and the underlying values retain their signs and precision. Accuracy, native agreement, ties, individual gaps and ranges obtained by omitting one topic are provided separately in the [full summary](summary.json).

**Only** inserts the named region into a neutral recipient. **Omit** restores that region to neutral within the otherwise complete downstream transplant. **Native knockout** replaces it with neutral residuals while the other positions and source slots remain native. The percentages of different regions can overlap or interact.

## Span size, position and scoring token

| Panel · checkpoint | Options only | Options omitted | Score only | Score omitted | First 8 only | Last 8 only |
| --- | --- | --- | --- | --- | --- | --- |
| Original · Mistral | 70.4 | 20.4 | 20.7 | 77.3 | 0.0 | 90.6 |
| Original · Gemma | 84.8 | 4.8 | 1.6 | 100.3 | -0.1 | 5.7 |
| New · Mistral | 49.5 | 22.9 | 26.6 | 67.6 | 0.0 | 90.8 |
| New · Gemma | 79.3 | 8.4 | 0.6 | 100.0 | 0.2 | 8.7 |

The options block alone retained 49.5–84.8% across the four model/panel combinations. Omitting it left 4.8–22.9%. At the single scoring position the corresponding only values varied much more by model. The first eight downstream positions had little donor separation. The last eight behaved very differently in the two tokenizers. Mistral has six tokens from the start of the closing instruction through scoring, so its last-eight window also contains the last two option tokens. Gemma has ten closing tokens, so its last eight fall entirely after the alternatives. The whole downstream reference ordered all six matched pole pairs in all four cells.

## Clause order and the two periods

First and second designate the printed option order. Either pole could occupy either position. In all 12 prompts per model/panel the two periods had the same native token id, and each was a separate token from the preceding option phrase. Every transplant therefore uses a recorded absolute index and actual token id.

| Panel · checkpoint | First clause only | Second clause only | First period only | Second period only | Second period omitted | Second period native knockout |
| --- | --- | --- | --- | --- | --- | --- |
| Original · Mistral | 21.9 | 54.1 | 14.4 | 37.8 | 76.6 | 78.2 |
| Original · Gemma | 23.1 | 65.8 | 1.4 | 38.8 | 82.2 | 83.8 |
| New · Mistral | 13.6 | 37.6 | 9.2 | 21.8 | 87.9 | 88.0 |
| New · Gemma | 31.8 | 45.8 | 11.9 | 27.3 | 88.8 | 88.6 |

The second period carried a larger donor contrast than the first on both panels and both checkpoints. On the new topics it carried 21.8% in Mistral and 27.3% in Gemma. Omitting just this period still retained 87.9% and 88.8%, respectively. Other positions can therefore sustain a substantial contrast. For all 13 follow-up windows, including isolated phrase tokens and matched two-position tails, see the [region-level results](summary.json).

## What the donor had seen

The following transfers all use the second period in a neutral, correctly ordered recipient. Only the donor's context changes. Stem masking hides the topic question. Phrase masking hides both alternatives' words while preserving their printed label scaffolding and separator tokens. The reversed donor reads grammatical alternatives in the opposite order. The rebased route adds the source lexical-minus-neutral residual to the target neutral background, with FP32 arithmetic and the original BF16 injection dtype.

| Panel · checkpoint | Matched | Stem masked | Phrases masked | Stem + phrases masked | Reversed raw | Reversed rebased |
| --- | --- | --- | --- | --- | --- | --- |
| Original · Mistral | 37.8 | 44.5 | 0.7 | 0.2 | -32.9 | -31.9 |
| Original · Gemma | 38.8 | 31.5 | 0.2 | 0.0 | -33.3 | -32.5 |
| New · Mistral | 21.8 | 16.4 | 0.0 | 0.1 | -22.3 | -18.4 |
| New · Gemma | 27.3 | 20.7 | 0.0 | 0.0 | -26.4 | -20.8 |

Masking the phrase tokens reduced the absolute raw gap at this period to less than 0.7% of the whole-suffix reference on every panel/model. Stem masking alone retained a positive contrast in all four. Reversing the mapping reversed the aggregate direction on both panels in both models, including after rebasing. The behavioral count below restricts the sample to individual donor/orders on which the source-native and recipient-native preferred letters actually disagreed.

| Panel · checkpoint | Follows source letter | Follows recipient letter | Tie | Second period only, leave-one-topic-out (%) | Compatible topics for reversed **first** period |
| --- | --- | --- | --- | --- | --- |
| Original · Mistral | 18 | 4 | 0 | 32.1 to 45.2 | 1/6 |
| Original · Gemma | 15 | 7 | 0 | 34.9 to 45.6 | 4/6 |
| New · Mistral | 19 | 3 | 0 | 18.5 to 23.9 | 4/6 |
| New · Gemma | 19 | 5 | 0 | 25.6 to 28.4 | 5/6 |

The second period was compatible in 24/24 donor/orders in all four cells. First-period transfers under reversal had fewer eligible topics because the first alternative can change token length. Missing positions were never shifted. At the closing instruction and score token, reversed raw transfers generally gave smaller aggregate gaps and less consistent letter following. All positions, orders, raw/rebased contrasts and eligibility counts are retained. [Example prompt IDs and offsets](trials/clauses-prospective-mistral/prompts.jsonl) · [Example donor interventions](trials/clauses-prospective-mistral/interventions.jsonl)

For a concrete trial, take the first new topic and its week-view donor. The Mistral recipient displayed A for a week and B for a month. Its native lexical A-minus-B margin was +15.969 nats. Transplanting only the second period produced +5.469 when the donor saw that order and −3.500 when only the donor saw the opposite order. The recipient's input stayed identical. These are single-order margins for one source. The tables above instead contrast both sources and both orders. [Calendar trial and its counterparts](trials/clauses-prospective-mistral/interventions.jsonl)

## Scope and checks

These interventions identify a mapping-sensitive residual at the end of the second displayed alternative under this prompt and these cuts. The second period occurs later and has access to both alternatives. Punctuation identity, order, causal history and proximity to scoring have not been independently varied. Repeated neutral-token masks also change the donor input distribution. This single-forward choice task leaves portability before a question, independent semantic updating and transfer to other tasks open.

Eight offline GPU jobs used the same two pinned snapshots. The primary source was `8e8cb072ab4f288f764b2264b449f44aaabcc93e` and the follow-up source was `d47dd4667127839e7372ea34ed644da37aa97244`. Their GPU preflights passed 105 and 107 tests with zero skips. All 192 trial donor/order records exactly reconstructed their own neutral and combined slot-plus-suffix references across the entire scoring-position vocabulary. Original-panel native and full-suffix scores reproduced the earlier study exactly. Follow-up closing and score interventions reproduced the primary measurements exactly on both panels. The two stages repeated the same donor/orders under different interventions, with 13,848 metered forwards in total and no free generation. [Verification](verification.json) · [All raw trials](trials/) · [File checksums](SHA256SUMS)

~~~bash
shasum -a 256 -c SHA256SUMS
python3 summarize.py | diff - summary.json
~~~
