# Literal-first conditional curriculum

The fixed 384-update literal warmup improved native fitting accuracy from 3,276/3,840 (85.3%) to 3,687/3,840 (96.0%), but failed the predeclared stage and strict-unit gate. The frozen-E and joint-E slot branches remained closed. No source adapter or G was trained, and independent qualification content remained unopened.

The [sealed plan](PLAN.md) retained all eight known action topics, three fitting flag-name pairs and two fitting wording families. Its separate fresh bank used two new flag-name pairs, a new wording family and a new question scaffold. The 480 fitting and 160 fresh descriptions were independently authored as source, exhaustive cases or reintegrated rule. Every description was scored on all four Boolean worlds and both A/B label mappings. Each strict unit fixes topic, flag pair, wording and stage and requires every policy, world, genuine source order and label mapping to be correct.

| Literal panel | Initial native | Final native | Final strict units | Final forced A/B | Final A/B ties |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fit source | 1,271/1,536 | 1,482/1,536 | 17/48 | 1,480/1,536 | 4 |
| Fit cases | 1,277/1,536 | 1,443/1,536 | 13/48 | 1,440/1,536 | 4 |
| Fit rule | 728/768 | 762/768 | 44/48 | 761/768 | 1 |
| Fresh source | — | 429/512 | 1/16 | 429/512 | 0 |
| Fresh cases | — | 485/512 | 4/16 | 485/512 | 1 |
| Fresh rule | — | 203/256 | 0/16 | 201/256 | 4 |

Native correctness uses the full-vocabulary greedy answer token. Forced A/B correctness uses the strict sign of the A/B logit difference and counts ties as incorrect. All final decisions emitted A or B. The complete counts, per-world breakdowns and coverage are in [summary.json](summary.json), reduced from the unfiltered [native decision records](run/decisions.jsonl).

The fitting admission required at least 95% native accuracy and 90% strict units at every stage. Rule passed both requirements with 762/768 native and 44/48 strict. Source passed native accuracy but failed strict coverage with 17/48 units. Cases failed both requirements with 1,443/1,536 native and 13/48 strict. The total native threshold passed, while the conjunction remained false. The post-endpoint fresh literal diagnostic was 1,117/1,280 (87.3%). It did not select a checkpoint or reopen slot training.

This is a bounded failure of the declared literal-recipient curriculum. Its high aggregate fitting accuracy does not establish complete conditional reading, a transferable slot interface, functional continuity or semantic recurrence. The first failed gate and fixed endpoint are retained. Any revised loss, schedule or additional training belongs to a separately sealed experiment with fresh diagnostic material.

The task-only rank-16 q/v adapter used the last four layers of the frozen Gemma-2-2B-IT backbone. The [training trace](run/training.jsonl) contains all 384 prescribed updates. The run used 5,315 forwards and 3,100,582 processed tokens in 184.5 seconds, within its bounds. It generated no continuation tokens. The backbone hashes before and after match, and saving and reloading the final task adapter reproduced the checked native logits exactly. These receipts are in [terminal.json](run/terminal.json), [fits.jsonl](run/fits.jsonl) and [warm_gate.json](run/warm_gate.json).

The exact clean execution source was `94473d4c775a01c9e60cf83ccccdf615bf073121`, recorded in [source.json](run/source.json). The returned run archive SHA-256 was `49b7196152620f5e147e7a3c7c0cdd53b23dc1af6df7046668410c9b2ce6eb55`. The saved adapter is [warm_task.npz](run/warm_task.npz), SHA-256 `4fb28b1c00ccebf89387f3dfb1793bef3c9ea8bedf782cb0d359e23943cd7d5c`. These Gemma-derived factors are distributed under the [artifact conditions](ARTIFACT_LICENSE.md), including the accompanying Gemma terms, use policy and modification notice. Original code and synthetic materials retain Apache-2.0.

Recompute the score reduction with `python3 -S provenance/conditional_curriculum_20261103/summarize.py` from the repository root. Verify this study's files by running `shasum -a 256 -c SHA256SUMS` in this directory.
