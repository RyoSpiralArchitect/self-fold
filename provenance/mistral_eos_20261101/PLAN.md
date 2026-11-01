# First pretrained experiment

Compare greedy native EOS stopping with an eight-token ceiling on the same cached Mistral-7B-Instruct-v0.3 checkpoint, revision `c170c708c41dac9275d15a8fff4eca08d52bab71`. The earlier tiny decoder explicitly lowered the logits of its three reserved ids, including EOS. The pretrained checkpoint uses its native logits and stop ids.

Run the two development descriptions `caution_plus_p0_self` and `caution_minus_p0_self`, separately in each length condition. Use the plain operator, two updates in each of T/L/B, eight slots, the default midpoint cut/final writer, uncalibrated identity transport, eager BF16 on one CUDA device, greedy decoding, no KV reuse, clone observers and one terminal L branch. Score only the first two development tasks, after the trajectories, at steps zero and two.

Both length conditions use a 1024-token context and a 600-second forward-boundary time limit per trial. EOS-first has the full remaining context for each generation. The fixed condition also stops after at most eight generated tokens. Prompt lengths can differ by arm. Stop at native EOS even on the last allowed draw. Retain nonempty censored text, stop reason, effective budget, actual EOS id, and content-token count. Keep incomplete trials under their original identities. Run each trial once, in a fresh process, one checkpoint at a time.

Report main T generations, main B roundtrips, L/B clone observers and terminal branch readouts separately. Check identical L/B starting states and first shared updates, post-update state hashes, forward counts and development scores. Inspect sample text with its recorded stop reason. Differences are descriptive for two starts in one seed group on one checkpoint. The latent route has no reviewed semantic gate.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for seed in caution_plus_p0_self caution_minus_p0_self; do
  for policy in eos cap8; do
    if [[ "$policy" == eos ]]; then length=(--until-eos); else length=(--max-new-tokens 8); fi
    python -m selffold smoke --backend hf --profile configs/models/mistral7b.yaml \
      --checkpoint "$MISTRAL_LOCAL_CHECKPOINT" --device cuda:0 --dtype bfloat16 \
      --allow-uncalibrated --seed-id "$seed" --regime plain --steps 2 --slots 8 \
      "${length[@]}" --max-context 1024 --wall-seconds 600 --tasks data/tasks.dev.jsonl \
      --task-limit 2 --branch --out "runs/mistral-eos/$seed-$policy"
  done
done
```
