# Opposed-donor training at a structured language boundary

On the fresh assignments, the writer trained on the worse donor correctly switched the default action name for **83/96 pairs** and reversed all subsequent decisions for **81/96**. Requiring the four reported rows to be correct on both donors as well gave **50/96**. The starting writer completed one joint pair. With identical starting weights, exposures and updates, averaging the two donors' role losses gave **64/96** joint pairs. The registered row-retention and paired-specific advantage criteria both closed negative. [Sealed design](PLAN.md) · [Full paired scores](produce/summary.json)

## The held-out assignment

Training exposed each of the eight action menus only with one-against-three or three-against-one row assignments. The primary endpoint uses the two-against-two assignments for four menus previously outside the parent writer study: river/ridge, Celsius/Fahrenheit, 24/12-hour clock, and instrumental/vocal music. Both continuations learned these names. The fresh test thus probes unseen assignments of known names. Each panel crosses 24 physical tables, two opposite role donors and four display orders. Each report is written before the later edit and reused for 14 edits, four worlds and both native answer orders. Both donors must satisfy the criteria on the same presentation. Display orders are repeated measurements.

Both fits started from the same 245,760-parameter Gemma writer adapter. Each received the same four opposed pairs per batch, shuffled in precisely the same order for 256 steps. Both trained on the five report fields and a signed default-choice margin. The individual objective averaged the two role penalties within each pair. The paired objective applied the larger penalty. Neither fit conditioned a test report on its twin's response. G1, the role marker, parser, G2 and final Gemma recipient used the previous structured interface. One training seed was evaluated. [Training configuration](design.json) · [Weights at the two fixed endpoints](produce/fits_closed.json)

## Fresh panel

| Condition | Four exact rows in one report /192 | Both donors' rows exact /96 | Both default names correct /96 | All later choices reversed /96 | All requirements together /96 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Starting writer | 133 | 60 | 4 | 2 | 1 |
| Average donor penalties | 155 | 69 | 87 | 87 | 64 |
| Larger donor penalty | 138 | 52 | 83 | 81 | 50 |
| Direct state (L) | 192 | 96 | 96 | 96 | 96 |
| Exact report through the fixed parser | 192 | 96 | 96 | 96 | 96 |

Both continuations substantially improved use of the supplied role. For the larger-penalty fit, the number of pairs reporting the same default on both sides fell from 78 to 10. It recovered 49 joint pairs over the parent with zero losses. Relative to the average-penalty control it recovered eight and lost 22. On **31 pairs** it named both defaults correctly and succeeded at every later choice despite a row error on at least one donor. The copied-row editor can therefore conceal report errors if measured on downstream decisions alone. Among the 83 pairs with correct default names, two still failed downstream.

The larger-penalty fit exceeded the parent in exact four-row reports (138 versus 133), while exact rows on both donors fell from 60 to 52 pairs. The absolute row-preservation floor was 164/192; the matched control reached 155/192. Consequently the registered transfer gate failed despite the larger-penalty fit gaining joint pairs in all four fresh menus. Its joint score beat the control only on music. [Pair-by-pair outcomes](produce/outcomes.json)

| Fresh menu | Parent joint /24 | Average penalty /24 | Larger penalty /24 |
| --- | ---: | ---: | ---: |
| River or ridge | 0 | 16 | 10 |
| Celsius or Fahrenheit | 0 | 20 | 11 |
| Clock format | 0 | 16 | 14 |
| Music preference | 1 | 12 | 15 |

As a descriptive stricter count, requiring joint success under **all four printed orders** for a physical assignment gave 0/24 for the parent, 4/24 with the average penalty, and 1/24 with the larger penalty. The results concern a supplied role in a structured four-row state. The original raw-slot recurrence and independent qualification remain open.

## Earlier panel and verification

On the four previously evaluated menus, the newly generated parent scored 30/96 joint, the average-penalty fit 75/96, and the larger-penalty fit 40/96. Their exact four-row counts were 163, 159 and 136 of 192 reports. Their downstream-only paired scores were 33, 91 and 62 of 96. For the parent, 189/192 token sequences on this panel matched the earlier archive. This run used a new mixed-panel batch schedule. The earlier downstream score remains 31/96; comparisons above use contemporaneous generations.

The RTX 5090 run completed both fits, generated 1,152 free reports and observed native EOS on every report. The final recipient freshly evaluated 256 distinct row/menu/order inputs. CPU replay reproduced all 1,920 route presentations including the two anchors. A second implementation read the report text and reconstructed the 129,024 downstream judgments for the three writer arms. Its lexical lookup uses the same case folding for the dictionary and output, including Celsius and Fahrenheit. [Raw reports](produce/reports.jsonl) · [Independent check](verification.json) · [Reproduction](RUNS.md)

A next experiment can hold the average role loss fixed and separately train agreement on the four output rows across the opposed donors. Testing row preservation and role reversal jointly on fresh assignments would show whether that constraint helps.
