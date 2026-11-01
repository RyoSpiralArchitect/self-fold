# First public CPU experiment

The public source starts from `self-fold-starter.zip`, SHA-256 `162898dbae26d899c84efc22a8c46b901bc77cd4436f805c79f9950b11caaae2`. The archive contains a source snapshot and historical receipts. Its Git history was not included. The original [starter receipts](../RECEIPTS.md) identify the source used for those earlier runs.

## Question

Do L and B start from the same encoded state and produce the same first update before B decodes and re-encodes it? Record whether their stored states then separate and how much main-path compute the language roundtrip adds. T provides the text-only reference. These are measurements of the harness using random weights.

## Fixed conditions

Use `caution_plus_p0_self` (“I prioritize keeping decisions revisable.”) and `caution_minus_p0_self` (“I prioritize making decisions promptly.”). Run each in both the plain and expansion regimes for three steps. Expansion uses decompose, boundary, then reintegrate. Keep the tiny decoder at weight seed 47, uncalibrated identity transport, CPU FP32, one thread, eight slots and an eight-token generation cap. Keep clone observers and the single branch from the final L state enabled.

Each of the four runs has three arms (T/L/B), nine transitions and twelve task rows. Score the first two development tasks at steps zero and three. They are the two cost variants of `decision-0`. Record native parity, L/B initialization and first pre-roundtrip hashes, main-path forward counts, and length-censored generations. Keep stop reasons and any failed trial. The design contains no held-out evaluation or semantic qualification.

## Reproduce

From the source commit identified by each run's `source.json`, install the project and use a fresh output directory.

```bash
python -m pip install -e '.[dev,hf]'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m pytest -q
for seed in caution_plus_p0_self caution_minus_p0_self; do
  for regime in plain expand; do
    python -m selffold smoke --backend tiny --device cpu --dtype float32 \
      --cpu-threads 1 --slots 8 --max-new-tokens 8 --allow-uncalibrated \
      --seed-id "$seed" --regime "$regime" --steps 3 --tasks data/tasks.dev.jsonl \
      --task-limit 2 --branch --out "runs/launch-001/$seed-$regime" || exit 1
  done
done
```
