#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m selffold smoke --backend tiny --allow-uncalibrated --steps 2   --out "${1:?Supply a new output directory}" --tasks data/tasks.dev.jsonl --task-limit 2 --branch
