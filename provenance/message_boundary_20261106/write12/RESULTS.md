# Reading a named row from a single correct record

Gemma-2-2B-IT returned the requested action on **602/768 (78.4%)** prompts
when the program supplied the correct four-row record. The registered
criterion required at least 95% on each question type. Instruction questions
scored **298/384** and observation questions **304/384**. On the same table
presentation, four separate questions about its four rows were all correct
in **59/192 (30.7%)** cases. The result sets the next priority to reading
the named row in a single record (`row_readout_first`). [Registered plan](PLAN.md)

## Which rows were read correctly?

Each coordinate appeared equally often in each of the four printed
positions. Each action was correct equally often at every coordinate. The
four familiar menus each supplied two English action names. The six table
assignments per menu put each action in exactly two of the four rows.

| Requested coordinate | Instruction question | Observation question |
| --- | ---: | ---: |
| off/off | 79/96 | 87/96 |
| off/on | 69/96 | 71/96 |
| on/off | 56/96 | 55/96 |
| on/on | 94/96 | 91/96 |

In particular, the two mixed coordinates gave different scores even with
printed position balanced. Pooling across coordinates, positions 1 through
4 yielded 132, 139, 163 and 168 correct out of 192 each. Always returning
the first listed action, the first printed row's action or the last printed
row's action would each score 384/768. The menu-level counts were 130, 145,
165 and 162 out of 192, in menu order 0, 2, 4, 6.

On **482/768 (62.8%)** paired comparisons, both answers were correct after
the queried row traded actions with another row. The required proportion
was 90%. For example, the following single-record prompts used the same
question about the first printed entry, `on/off`. Only its action and the
action at `on/on` traded places.

| Saved table ID | Action at on/off | Action at on/on | Gemma's answer to on/off |
| --- | --- | --- | --- |
| `single/single/6/instruction/6/2/2` | sorting mail by date | sorting mail by sender | sorting mail by sender |
| `single/single/6/instruction/10/2/2` | sorting mail by sender | sorting mail by date | sorting mail by sender |

Of the 166 single-record errors, **133** returned the other listed action
verbatim. The remaining 33 returned a coordinate (19) or shortened
“the earliest reports” to “earliest reports” (14). The scoring rule required
the full action name. Counting literal matches gave the same 602 correct
as normalization. [Raw answers](produce/reports.jsonl) · [Per-answer scores](produce/scores.json)

## What changed when the other records were present?

Each full prompt added the previous table and the other active record.
Agreement and conflict refer to the **instruction and observation**,
which agreed everywhere or disagreed everywhere. The single-record answer
was generated once and paired with the same target table, row, cue and
question in each comparison.

| Visible records, in order after the previous table | Instruction | Observation | Total | All four rows correct | Both sides of a swap correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Requested record alone, no previous table | 298/384 | 304/384 | 602/768 | 59/192 | 482/768 |
| Instruction then observation, agreeing | 331/384 | 329/384 | 660/768 | 110/192 | 571/768 |
| Observation then instruction, agreeing | 344/384 | 334/384 | 678/768 | 120/192 | 602/768 |
| Instruction then observation, conflicting | 196/384 | 303/384 | 499/768 | 62/192 | 321/768 |
| Observation then instruction, conflicting | 280/384 | 268/384 | 548/768 | 55/192 | 391/768 |

There were 384 matched units with opposite instruction and observation
answers. A unit succeeds when the model gives **both** correct actions in
separate calls. With each record presented alone, 261/384 units succeeded.
The same two questions succeeded on 128/384 units with instruction first
and 175/384 with observation first. Both full orders succeeded on 73/384.

| Conflicting full-record order | Both presentations succeed | Single only | Full only | Neither | Net gain for single |
| --- | ---: | ---: | ---: | ---: | ---: |
| Instruction then observation | 86 | 175 | 42 | 81 | +34.6 percentage points |
| Observation then instruction | 127 | 134 | 48 | 75 | +22.4 percentage points |

Removing conflicting records improved joint correctness in both orders.
Adding an agreeing record and the previous table improved per-answer
accuracy over the single record in both orders. These comparisons change
sequence length and the target record's placement along with its context.
The single-record errors directly establish that mistakes can occur after
the experimenter has identified and supplied the right record. The repeated
coordinates, menus and table assignments define a finite panel. The counts
do not estimate performance on new vocabularies or other renderings.

A follow-up can distinguish locating an entry from returning its action by
showing just the requested row on a fresh panel and comparing it with the
intact four-row table. The same questions, action names and paired swaps
would preserve the distinction surfaced here. The present study closes
at the registered single-record criterion. [Full gate and matched counts](produce/summary.json)

## Execution and reproduction

The plan, stimuli and scoring code were committed before generation. A single
frozen Gemma run generated one answer for each of **3,840 distinct prompts**.
Every answer ended at EOS after 3–10 output tokens. It used one RTX 5090, greedy decoding and a
fresh cache for each prompt, with a limit of 48 new tokens. The 48 transition
keys and all prompts were disjoint from the two preceding studies. This was
a text lookup experiment on known menus with no fitting. [Source commit](https://github.com/RyoSpiralArchitect/self-fold/commit/c5b2d3e79f927b191b445574fe153b54a0f7da4e) · [Run receipt](produce/terminal.json)

A fresh process reproduced all 3,840 scores, the five panels, paired counts
and decision from saved outputs. A separate standard-library parser read the
actual displayed headings, row labels and action strings. It agreed with
every per-answer correctness label, all four-row counts and all 768
single-record swap comparisons. [Reproduction commands](RUNS.md) · [Replay](reduction.json) · [Visible-table verification](visible_rows.json)
