# Choice clauses and what the donor has already read

On original-panel Mistral, the last eight positions alone retained a whole-suffix-normalized pole gap of 0.906. The closing span retained 0.218 and the scoring position 0.207. Its last eight tokens extend into the end of option B. This follow-up was specified from that original Mistral observation before inspecting any prospective-panel outcome. The same two six-topic panels and two checkpoints will be reported separately. The new panel shares the task template with the development panel.

## Fixed regions

Separate option A's clause from option B's at the actual token containing the start of the second label. In each clause test the tokens of the option phrase, its final word token, the token containing the following period, and the final two clause positions. The word token is the one covering the final character of the phrase, which can be a word fragment. Report whether the period also belongs to an option-phrase token, and whether the two periods share an ID. Repeat the full closing span and exact score token. Test the final two positions of B plus the full closing span as one contiguous intervention. Save each tokenizer's exact indices, offsets and IDs. Apply the same only/omit/native-knockout operations and counters as the primary study. B is later in the causal sequence and its state can already depend on A.

## Question versus option content

At four anchors (A's period, B's period, the closing span and the scoring position), test raw and rebased transfers from donors with (i) normal text, (ii) reversed option assignments, (iii) only the question stem neutralized, (iv) only the two option phrases neutralized, or (v) both the stem and phrases neutralized. For phrase masking retain the label, equals sign, separators and chat template wherever the tokenizer assigns them separate tokens. Replace the selected IDs by the same number of neutral period IDs. Forward length and the selected recipient token IDs must match. Report missing or fused positions without realignment. Keep the simple joint mask from the first study as a different condition, since it also masked the A/B scaffolding.

Use the primary study's oriented margins, pole gaps, normalization, leave-one-topic-out ranges, native-sign agreement and source/recipient native-letter conflict counts. Compare A versus B clause tails of identical width and the four masking cells on the same target. Repeat the primary run's closing and score interventions and report their maximum A/B score discrepancy. No new material is used to choose an intervention after seeing this study's results.

~~~bash
for panel in original prospective; do
  python scripts/probe_slot_content.py clauses --model mistral7b --panel "$panel" --checkpoint "$MISTRAL_LOCAL_CHECKPOINT" --out "runs/clauses-$panel-mistral-001"
  python scripts/probe_slot_content.py clauses --model gemma2b --panel "$panel" --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out "runs/clauses-$panel-gemma-001"
done
~~~
