# Can a fixed question distinguish sources that share a greedy readout?

This is a development diagnostic. Before scoring, freeze the [descriptions and questions](../../data/slot_contrasts.dev.json), [runner](../../scripts/probe_slot_content.py), model snapshots and comparisons in one Git commit. No threshold is selected from these outputs. No adapter is fitted.

## Run sequence

1. **Archived Mistral donors.** Restore the two literal descriptions, initial encodings (E), L after steps one and two, and B after steps one and two from the [EOS trials](../mistral_eos_20261101/trials/). Verify the exact host, checkpoint and state fingerprints. Ask whether the speaker prioritizes revisability or speed. The probe is frozen before scoring, but its contrast comes from an already studied pair. These twelve donor measurements plus two nulls use one question in two label orders.
2. **Mistral content controls.** Use six new preference contrasts, two poles and two phrasings per pole (24 descriptions). The question and option strings are identical for all donors within a contrast. Measure literal text, the initial encoding E, one plain update G, its language roundtrip B, reversed G slot order, and G with only its per-slot magnitudes preserved. A native uninjected placeholder prompt and all-zero slots provide two nulls per group.
3. **Gemma 2 replication.** Apply the same fixtures, prompts, labels, controls, decoding and analysis to Gemma-2-2B-IT. Only the checkpoint and its native tokenizer/template, layer widths, boundaries and EOS ids differ. Fix this comparison before viewing Mistral scores.

Use the exact Mistral and Gemma snapshots identified in the runner, one GPU process at a time, BF16, eager attention, eight slots, midpoint cut, final writer, identity transport, no KV reuse. Stop greedy free readouts at native EOS or the 1024-token context boundary. A donor's question and labels never enter encoding, updating or free verbalization. Both controls and scoring operate on clones. A magnitude-only sham replaces each slot's direction with a fixed CPU Gaussian unit vector for that position (seed 4711), rescales it to the donor's norm, then casts back to BF16. Record rounding error. Each checkpoint has its own fixed directions at its own width.

## Endpoints and denominators

- For a fixed donor and question, score both answer labels using full teacher-forced log likelihood. Swap the options attached to A and B and score again. Orient both log-probability differences toward the plus option and take their arithmetic mean, called the plus margin (nats). Store both raw orders and label lengths. This cancels an additive label bias.
- The primary directional endpoint counts matched pairs with a larger margin under the plus donor than under the minus donor. There are 12 pole pairs per route, clustered in six groups, and one pair per route in the archive. Report exact ties separately and show how many groups favor plus in both phrasings. A stricter endpoint counts correct donor signs over 24 descriptions, with ties reported separately. Ordered pairs can still lie on the same side of zero.
- For each group report the mean absolute opposite-pole difference across matched phrasings and the mean absolute same-pole difference across phrasings. Report the two null margins separately. They receive no donor-specific signal. The controls test sensitivity to input substitutions in this readout.
- Generate D(E) and D(G) once per description, plus one zero and one uninjected free readout per model (50 generations). Reuse the exact D(G) for B. Report distinct strings, cross-pole collisions in the 12 matched pairs, and stop reasons. An empty D(G) has no B state: retain its generation, count its missing pair and keep the other routes. A nonempty censored output remains eligible.
- Each fresh checkpoint has 156 planned state/question measurements, or 312 option orders and 624 label likelihoods if all B states exist. Raw magnitudes across different models are descriptive. No human semantic review, held-out task evaluation, training or new semantic gate is part of this design.

## Reproduce

From the experiment source commit, install the HF extras and set the two environment variables below to existing snapshot directories. Use fresh output paths.

~~~bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/probe_slot_content.py archive --model mistral7b --checkpoint "$MISTRAL_LOCAL_CHECKPOINT" --out runs/content-archive-001
python scripts/probe_slot_content.py contrasts --model mistral7b --checkpoint "$MISTRAL_LOCAL_CHECKPOINT" --out runs/content-mistral-001
python scripts/probe_slot_content.py contrasts --model gemma2b --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/content-gemma-001
~~~
