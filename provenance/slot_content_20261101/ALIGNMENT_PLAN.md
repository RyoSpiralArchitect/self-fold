# Lexical positive control at the injection boundary

This follow-up was specified after reading the original Mistral archive probe and before running either alignment experiment. The archived literal descriptions changed the sign of the option preference. The archived initial slots yielded equal margins for the two donors. Test whether the same scoring prompt can read deliberately explicit lexical donors supplied at the same eight positions.

Reuse the six development topics, their fixed option strings, both label orders and both pinned checkpoints. Each donor is just **one option phrase**, without a letter or a question. This deliberately reuses the vocabulary of the question as a lexical positive control. Tokenize without added special tokens and pad on the right with the experiment's period token to eight positions. An overlength or empty donor stops the run rather than changing the phrase.

For every topic and pole, compare four routes.

| Route | Input to the scoring prompt |
| --- | --- |
| Native words | Insert the eight token ids in place of the eight neutral placeholders and run the native model at all layers. |
| E(words) | Encode the same unpadded phrase with the original SELF-FOLD encoder and inject those eight slots into the neutral recipient. |
| Aligned | Run the native lexical prefix only up to the last slot, capture the eight residuals at the input to the cut layer, then inject those residuals at the same positions in a neutral recipient. |
| Aligned, reversed | Reverse the eight captured rows before injection into the neutral recipient. |

Both label orders must have the identical donor prefix and slot positions. The donor prefix ends before the question and labels. In the aligned recipient, the earlier layers outside the injected slots still process neutral placeholders. This transfer can therefore differ from exposing native words throughout the full model. The comparison with E(words) changes donor context, positions and which residuals carry the content together.

Per checkpoint, run 12 donors × 4 routes and 6 topics × 2 controls (uninjected neutral and replay of its own neutral prefix) for **60 measurements, 120 option orders, 240 label likelihoods**. Report balanced-sign correctness out of 12 donors, directional ordering out of six pairs, ties, median signed pole difference and maximum absolute neutral-replay discrepancy across the raw label scores. Save every donor token sequence and captured state. Use the same BF16, EOS settings and CUDA limits as the first probe. There is no free generation in this study. Neither checkpoint is trained or selected on the outcome.

~~~bash
python scripts/probe_slot_content.py alignment --model mistral7b --checkpoint "$MISTRAL_LOCAL_CHECKPOINT" --out runs/aligned-mistral-001
python scripts/probe_slot_content.py alignment --model gemma2b --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/aligned-gemma-001
~~~
