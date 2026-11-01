# Which downstream residuals reproduce the native lexical choice?

The previous lexical replay transferred every middle-layer position from the end of eight source slots through the scoring position. It reproduced the native choice in 12/12 donors on both checkpoints. Two explanations remain entangled. The donor provided many positions and it had already read the actual question and option-to-letter assignment. Freeze the following design and analysis before any localization scores are collected.

## Panels, models and geometry

Run the two previously pinned checkpoints on the [original six topics](../../data/slot_contrasts.dev.json) and on [six new topics](../../data/span_topics.prospective.json). Keep the panels separate. The second panel is an authored prospective diagnostic using the same explicit lexical task. In each of four trials use two donor poles, both option orders, BF16 eager execution, the original midpoint cut, eight padded lexical slots and one trailing neutral dummy. Each trial comprises 6 topics × 2 poles × 2 orders = 24 donor/orders. Score both single-token labels at the last prompt position before the dummy. Keep donor and recipient forward lengths identical for every transplant.

Partition the *actual* downstream tokens, with zero-based half-open indices, into the fixed instruction before the topic question, the question, both labeled options, the rest of the closing instruction/template before the scoring position, and the scoring position alone. Locate text boundaries with the native fast tokenizer's offsets and verify that its IDs exactly match the original rendered suffix. A boundary-crossing token belongs to the block on its right. Store offsets, actual IDs, prompt, slot indices and windows. Complement the disjoint partition with the whole closing span (through scoring), everything before the options, the first 8 downstream positions, the last 8 option positions, and the last 8 or 4 prompt positions. Windows can overlap. The last eight may include option text depending on the model template.

## Three interventions on each window

1. **Only.** Insert just the window from the natural lexical donor into a neutral recipient.
2. **Omit.** Transfer the entire downstream span into a neutral recipient except this window. Restore the neutral residuals there. This asks what is lost relative to the successful whole-suffix intervention.
3. **Native knockout.** Start from the natural lexical input and overwrite just the chosen window with neutral residuals. The donor's input slots and all other positions stay native.

The first and second interventions distinguish sufficiency from conditional removal. The third has a different background. Subsets can interact, so the values need not add up. The natural reference, neutral recipient, whole downstream transplant and combined slot-plus-downstream reconstruction are retained. Require exact vocabulary-logit equality for neutral self-replay and combined reconstruction. On the original panel compare native and whole-suffix A/B scores to the archived run.

## Does the transferred signal follow the donor's answer mapping?

Keep the recipient's real question and mapping. For the whole closing span, the last four positions and the score position, compare three donor contexts. The matched donor saw the same prompt. A natural reversed-order donor saw the same question but the opposite assignment of options to A/B. A masked donor saw neutral period ids instead of the topic question and both options, at their original positions, followed by the same closing instruction. The masked donor keeps the lexical source and the generic task instruction.

Transfer both raw residuals and a rebased lexical difference. For the latter, compute the lexical-minus-neutral residual under the **same donor context**, add it to the target neutral residual in FP32, and cast back to BF16 before injection. Preserve the raw result. For every cross-context window require identical forward length, absolute positions and native token ids in the transplanted window. If incompatible, report the window as missing and its denominator. No padding or manual realignment. Record the maximal pre-question discrepancy and the masked prompts. Neutral masking changes the input distribution. Reversal provides a grammatical control.

## Endpoints

For each donor average the plus-oriented label log-probability differences over both option orders. The *pole gap* for a topic is the plus-donor margin minus the minus-donor margin. The primary continuous measure is the sum of pole gaps for an intervention divided by the sum for whole-suffix transfer, separately for each checkpoint and panel. Equal weight goes to each complete topic before summing; large reference gaps can carry more weight in the ratio. Report numerator, denominator, per-topic gaps and leave-one-topic-out range. A nonpositive denominator gives no fraction. Fractions can be negative or exceed one. For omission report the retained fraction (loss is one minus that fraction).

Also report correct donor signs and agreement with native preference signs (12 donors), positive and tied paired gaps (6 topics), exact ties and numerical replay errors. On reversed-context raw and rebased reads count following the source versus target native letter specifically on donor/orders where those letters disagree and neither is tied. Read the reversal and masking results alongside the natural control. Keep the original and prospective panels separate and report every planned region. These measurements concern one forward at the stated cut. No token is generated, no weights are fitted and no semantic gate is changed.

~~~bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for panel in original prospective; do
  python scripts/probe_slot_content.py localize --model mistral7b --panel "$panel" --checkpoint "$MISTRAL_LOCAL_CHECKPOINT" --out "runs/span-$panel-mistral-001"
  python scripts/probe_slot_content.py localize --model gemma2b --panel "$panel" --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out "runs/span-$panel-gemma-001"
done
~~~
