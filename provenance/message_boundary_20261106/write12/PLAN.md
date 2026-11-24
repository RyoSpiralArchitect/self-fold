# Can Gemma read a named row when supplied with the correct record?

Prospective development study, 22 November 2026. The preceding record-cue
study found that even when three off/off entries agreed, only 112/128
instruction answers and 124/128 observation answers matched them. This
study removes competing records while keeping all four rows of the requested
table visible. The two choices remain in the prompt. The model must return
the action at the named coordinate.

## Stimuli and matched contexts

Use the four familiar action menus (0, 2, 4, 6). Express the table in the
listed option order (slot 0 or 1). Use every four-row assignment with two
occurrences of each action, the six patterns 3, 5, 6, 9, 10 and 12. This
balances the answer distribution at each coordinate and removes a majority
action. Ask both the instruction question and the observation question from
write11, substituting each of the four coordinates for off/off. Display each
table in four cyclic rotations so each coordinate appears exactly once in
each printed position. Prompt language and actions are English.

In the **single-record** condition the program supplies the requested record
type and its full table. Each (menu, target pattern, question, rotation,
coordinate) produces one unique prompt, for 768 generations. In the
**full-record** comparison, add the previous table and the other active
record. The active records either agree in all four rows or disagree in all
four. Show each of those conditions in both active-record orders. This gives
four comparison panels of 768, for **3,840 unique prompts** in total.
Single-record answers are generated once and referenced by their matching
full-record comparisons. Within a matched set the target table, record
heading, list of action names, wording, requested coordinate, explicit record
cue and output instruction are identical. Only the additional tables and
their order vary. The program supplies the record identity in every arm.

The previous patterns for menus 0, 2, 4, 6 are 1, 14, 1, 14 respectively.
The full transitions are feasible under the project's row-copy operator.
These priors make the 48 full transition keys distinct from write10 and
write11. Check both key and prompt disjointness before generation. The words,
menus and table task are familiar. Each prompt is an independent greedy call
with a fresh KV cache. Generate in deterministic hash order across conditions.

## Outcomes and decision rule

The primary answer is correct when its entire normalized text equals the
canonical action at the requested coordinate. Use write10's case folding
and punctuation/whitespace normalization. Count other answers as incorrect.
Also report valid canonical-name rate, literal string correctness, stops and
censoring. Score a capped answer by its content. Keep each raw response.

Apply the following conjunction separately to the **768 unique single-record
prompts** and to each of the two **768-prompt conflicting full-record
orders**. Publish the same measurements on both agreeing panels for diagnosis.

| Check | Denominator | Minimum |
| --- | ---: | ---: |
| Accuracy for each question type | 384 per question | 95% |
| Each question × named coordinate, including off/off | 96 per cell | 90% |
| Each menu and each printed position, separately | 192 per cell | 90% |
| Every coordinate correct for one menu, question, table and rotation | 192 blocks | 90% |
| Both answers correct when two action assignments trade places, including the requested coordinate | 768 unordered pairs | 90% |

For the last check, the two patterns differ in exactly two bits, one of them
the queried bit. Both actions keep the same frequency. In single-record
prompts only the two row assignments change. In the full panels the other
active table follows the registered relation. The pairs and rotations reuse
the same finite menus. Report exact coverage without treating them as
independent draws.

For each conflicting transition, rotation and coordinate, ask whether both
questions named their correct action. Compare the same two answers in the
single-record presentations with the two in each full-record order. Preserve
both-correct, single-only, full-only and neither for all 384 units per order.
The net gain is (single-only minus full-only) divided by 384. Also count
complete full-record units across both record orders.

1. If the single-record conjunction fails, prioritize reading the named
   row and choosing its action. Use the row, position and question breakdown
   to locate the failure.
2. If single-record and both conflicting full-record conjunctions pass,
   close this lookup question on the registered panel.
3. If single-record passes, both conflicting orders fail, and single-record
   improves the matched two-question rate by at least 10 percentage points
   in each order, investigate selection among competing records.
4. Otherwise investigate the order or context discrepancy before fitting a
   selector.

Removing two tables also changes sequence length and the position of the
requested record. Differences measure the effect of this context change.
The procedure addresses lookup over these rendered menus. Learned selection,
state updates and the independent six-check challenge are separate experiments.

## Execution and closure

Use the pinned Gemma-2-2B-IT checkpoint with CUDA bfloat16, eager attention,
one CPU thread, context 1,024 and cap 48 new tokens. Stop earlier at EOS.
Verify the checkpoint and native hook parity. Check resource availability.
Allow 200,000 metered forwards, six million processed tokens, a 2,400-second
host wall budget and an external 2,700-second limit. No fitting, sampling
seeds, trained controllers or downstream native-choice scoring enter this
experiment. Make a single scientific pass with the sealed code and stimuli.
Preserve a technical stop with completed rows in a distinct directory.

Save requests, raw generations, model identity, stop receipts, per-answer
scores, matched counts and decision. A fresh process regenerates the prompts
and recomputes all scoring and decision outputs from saved text. The analysis
also has literal table-lookup and wrong-record controls implemented as
fixtures. Close on the registered result without tuning on these answers.
