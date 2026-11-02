# Policy form and native trainability

A native Gemma reader trained on an explicit four-case list answered 58/64 training decisions correctly when the list was presented in training order. Reversing those same four sentences reduced correctness to 47/64. Under both orders, the reader selected the correct action for both opposed policies in 13/32 paired units. The earlier rule-trained literal reader also achieved 13/32 on this joint test. Neither new reader met the registered trainability and order-robustness screens. [Prospective design](PLAN.md) · [All decisions](run/decisions.jsonl)

## Paired comparison

We started from the four exposed groups in the preceding trainability control. Its final literal reader made 16 errors across 128 logical training decisions. The two source-order tags had identical teacher text. After collapsing those duplicates, all eight distinct errors occurred on disjunctions containing multiple satisfying worlds, including four errors in the exclusive-or family. This observation motivated the new comparison.

We compared the original disjunctive rule with an exhaustive four-case list in forward order and with the same sentences in reverse order. The source-to-action assignment was checked against the four-world ledger in every form. The new literal and eight-slot arms trained on the forward list only. The exact initial parameters, 64 group draws, questions, gold answers, optimizer and number of updates matched the previous rule fits. All 768 archived rule decisions replayed exactly. The 72 source records matched previously recorded hashes, representing 36 distinct donor forwards.

| Input and training source | Rule | Forward cases | Reversed cases | Both policies correct in both case orders |
| --- | ---: | ---: | ---: | ---: |
| Literal, initial | 43/64 | 47/64 | 52/64 | 10/32 |
| Literal, trained on rules | 56/64 | 48/64 | 52/64 | 13/32 |
| Literal, trained on cases | 47/64 | 58/64 | 47/64 | 13/32 |
| Eight slots, initial | 24/64 | 24/64 | 24/64 | 0/32 |
| Eight slots, trained on rules | 29/64 | 31/64 | 29/64 | 0/32 |
| Eight slots, trained on cases | 29/64 | 35/64 | 31/64 | 0/32 |

All entries use full-vocabulary greedy output on the training groups. At fixed forward wording, case training added ten correct literal decisions relative to rule training. At fixed reversed wording, it lost five. On the forward list, the case-trained literal reader met 26/32 opposed pairs, the slot reader 4/32. Their 95% native and 90% opposed-pair training screens both failed. The slot reader tied the two forced-choice scores on 56/64 forward decisions. The reported native score includes the model's deterministic argmax at ties. [Full crossover, forced ties, coverage and family counts](summary.json)

In the two exposed development groups, the literal case-trained reader obtained 27/32 forward and 28/32 reversed, compared with 28/32 and 27/32 at initialization. The corresponding slot scores were 13/32 and 14/32 after case training. These two groups supply a descriptive check. The wording change also alters length, repetition and lookup cues. The literal case fit processed 309,248 training tokens versus 245,760 for the archived literal rule fit, while the slot fits each processed 163,840. Each fit had 256 training forwards.

This single-seed cycle ended at its cap. The local improvement is confined to the trained literal rendering on this panel. A next design can replace the duplicate forward-list training tag with a genuine order variation at the same budget and reserve other case permutations and new groups for evaluation. The two-step G contribution and qualification remain open.

## Reproduction

The run used the pinned Gemma-2-2B-IT, BF16, the existing eight-slot workspace and source c807c638fc412d153a3b9948f5310fff6ac39e0a. Both newly trained adapters round-tripped through their saved files with identical checked logits. Backbone hashes matched before and after. The two fits completed 128 optimizer updates in total. No G or independent qualification data entered the run. [Execution receipt](run/terminal.json) · [Learned arrays and conditions](run/ADAPTER_LICENSE.md)

Run the experiment from the repository root at the source commit above.

~~~bash
PYTHONPATH=src python scripts/probe_policy_form.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/policy-form
python3 provenance/policy_form_20261103/summarize.py runs/policy-form
~~~

From the published study directory, run the following to verify its records.

~~~bash
shasum -a 256 -c SHA256SUMS
python3 summarize.py | diff - summary.json
~~~
