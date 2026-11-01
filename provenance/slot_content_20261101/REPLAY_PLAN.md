# Sequence-length and intervention-span controls

The alignment runs showed a maximum absolute neutral-replay discrepancy of 0.1903 nats on Mistral and 0.1613 on Gemma. Freeze this diagnostic before executing it. Use the same six option pairs, both label orders, both pinned checkpoints, BF16 and the existing cut. The question is whether matching forward geometry restores neutral replay, and where the native lexical preference can be recovered at this cut.

Append the same neutral dummy id after the complete scoring prompt in every forward. Both labels are single tokens. Read both log probabilities from the last prompt position, before the dummy. Compare these native scores to the already frozen alignment study, which appended the two alternative labels in separate teacher-forced forwards.

1. At each of 12 question/order combinations, record the native neutral logits. Capture neutral slot residuals from (a) a prefix ending at the last slot and (b) an equally long sequence with every post-slot token replaced by the neutral id. Inject each set of eight residuals into the original neutral recipient. Compare the slot bytes/numerics to a natural full-length capture, and report maximum differences across both labels and the entire final-position vocabulary.
2. For each of 24 donor/order combinations, capture residuals at the cut from a full native lexical donor. Compare the original native scores with three interventions into an otherwise neutral recipient: only the eight slots from a same-length opaque-tail donor, only the natural donor positions after the slots through the scoring position, and slots plus those downstream positions. Exclude the trailing dummy from the transplanted span. Check native versus opaque-tail donor slots, and all-vocabulary reconstruction for the combined transplant.
3. The downstream-span donor has already processed the actual question and alternatives at earlier layers. This is a single-forward localization control. It cannot qualify a portable latent state. Summarize the six paired donor orderings, twelve correct/incorrect donor signs, median margin difference and exact ties for each route. Keep all residual discrepancies. This study reuses development material and has no gradient, free generation or semantic gate.

~~~bash
python scripts/probe_slot_content.py replay --model mistral7b --checkpoint "$MISTRAL_LOCAL_CHECKPOINT" --out runs/replay-mistral-001
python scripts/probe_slot_content.py replay --model gemma2b --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/replay-gemma-001
~~~
