#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
profile="${1:?Usage: furnace_smoke.sh PROFILE EXISTING_CHECKPOINT NEW_OUTPUT}"
checkpoint="${2:?Existing local HF checkpoint path required}"
out="${3:?New output path required}"
# No download, retry, automatic deletion, training, or parallel model loading.
python -m selffold smoke --backend hf --profile "$profile" --checkpoint "$checkpoint"   --device cuda:0 --dtype bfloat16 --allow-uncalibrated --steps 2 --max-new-tokens 32   --max-context 768 --wall-seconds 600 --out "$out"   --tasks data/tasks.dev.jsonl --task-limit 2 --branch
