# Furnace runbook

## 1. Hardware is a prior, not a live measurement

The working hardware assumption is one RTX 5090 with 32 GiB-class VRAM. 128 GiB system RAM is a **planning assumption, not independently verified in this session**. CPU, OS, driver, installed wheels, free disk and checkpoint paths are unknown. `configs/furnace.expected.yaml` must not substitute for inventory.

The earlier LTWS V23 receipts document a high-memory CUDA workstation and a source-sealed, one-role-at-a-time procedure. They are historical context, not authorization to relaunch those completed jobs. No LTWS code or weights are copied into SELF-FOLD.

Historical source: https://github.com/RyoSpiralArchitect/Latent_Workspace/blob/main/provenance/engineering/v23_preservation_ft01_20261022/EXECUTION_HANDOFF.md

## 2. Use the existing working CUDA environment

Do not replace the system PyTorch installation. PyTorch's 2.7 release announced Blackwell support and CUDA 12.8 wheels; that is background compatibility information, not an instruction to downgrade or choose that exact version. Record what actually works on Furnace. The current package's HF range (`transformers>=4.57.1,<5`) is a candidate API line, **not a tested lock**. Adjust in a dedicated environment only after inspecting the actual checkpoint configuration, then freeze and test the resulting versions.

Official compatibility source: https://pytorch.org/blog/pytorch-2-7/

```bash
cd /path/to/self-fold
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m selffold inventory --out /path/to/new-inventory.json
python -m pip freeze > /path/to/new-environment-lock.txt
python -m pytest -q
python -m pytest -q -m hf
```

The last command must execute four tests, not skip them. It uses small random-config versions of the four real HF classes, requires no model download and should precede large weight loading. Then check one small CUDA matrix multiplication in the same environment and record the device capability and result. Do not infer checkpoint compatibility from a random-config class test alone.

## 3. Candidate checkpoints

| Profile | Official repository | Approx. total parameters | Active parameters where relevant |
|---|---|---:|---:|
| mistral7b | https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 | 7.3B | Dense |
| llama3b | https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct | 3.2B | Dense |
| gemma2b | https://huggingface.co/google/gemma-2-2b-it | 2.6B | Dense |
| phi_mini_moe | https://huggingface.co/microsoft/Phi-mini-MoE-instruct | 7.6B | 2.4B |
| phi_tiny_moe | https://huggingface.co/microsoft/Phi-tiny-MoE-instruct | 3.8B | 1.1B |

Sources: official model cards and SlimMoE, `paper/references.bib`. Counts are rounded. MoE resident weight memory follows **total**, not active, parameters. FP16/BF16 raw weights are about `2 * parameters` bytes, before runtime buffers and activations. The config admission floors are conservative proposed thresholds, not measured peak memory.

Phi-mini is the primary MoE candidate; Phi-tiny is a separately reported alternative, not a silent replacement. Phi-3.5-MoE's larger total weight count is deliberately not the initial default. Standard Phi dense models are not relabeled as MoE. Gemma-2-2B is a deliberately small text-only candidate, not a claim that it is Ryō's currently installed Gemma checkpoint.

Local path and revision fields are null until inspected. The user must already have the relevant model access/license and weights. No credentials or model files are included. `load_local` hashes local config/tokenizer/safetensors files and uses local-only loading with `trust_remote_code=False`. Phi cards have custom-code tags; a mismatch with native Phimoe requires explicit review rather than enabling remote code automatically.

## 4. One model, one bounded canary

```bash
bash scripts/furnace_smoke.sh configs/models/mistral7b.yaml \
  "$MISTRAL_LOCAL_CHECKPOINT" runs/mistral-canary-001
```

The script performs two recurrence steps in each arm with greedy native EOS stopping, a shared 1024-token context limit, baseline/final development task scoring and a one-time branch. It is an engineering canary. Each generation receipt records the prompt size, available output budget, emitted text-token count, stop reason and observed EOS id. EOS is excluded from the text-token count and included in compute. A nonempty output stopped by the context limit or an explicit token ceiling continues in the experiment as a censored observation. An empty generation cannot initialize the next text state. Use fresh output directories for other checkpoints.

Admission checks inspect available host RAM, CUDA free memory and disk. Forward-boundary guards cap time/compute and stop when less than 2 GiB CUDA memory remains. Those checks are not a hard process supervisor and cannot interrupt a stalled GPU kernel. Batch size is one, eager attention, no quantization, no compilation, no parallel loading. Model identity hashing occurs before loading and may be I/O-heavy.

Inspect `terminal.json`, `native_parity.json`, `host_identity.json`, `transitions.jsonl`, censored readouts and memory snapshots. Keep raw failed outputs. Fixes require a new source commit and a new run directory; do not edit a completed source seal.

## 5. After canaries

First return compact evidence to Ryō. Next develop and review a semantic gate using separate calibration examples. Only after it passes run a short L/B pilot with the same exact checkpoint, host settings and transport. `run --arm L/B` is blocked without a matching gate. T can be explored separately, but does not establish L/B validity.

There is no automatic matrix launcher. The proposed development matrix is 48 seeds × 3 arms × 2 regimes = 288 trajectories per model, each 12 steps. This is a **plan**, not a run authorization or a completed result. Start with a selected pair and inspect all intermediate outputs before separately deciding whether to scale.

## 6. No-go operations

No old weight deletion, no killing unrelated GPU processes, no rewriting completed LTWS trials, no automatic retry, no implicit download, no provider API call, no training hidden in smoke, no remote model upload. Keep model/tokenizer access credentials local. Private or proprietary data do not belong in this repository.

## 7. SlimMoE compatibility finding and explicit local-code route

The official Phi-tiny-MoE config checked on 2026-11-01 declares hidden_size=4096, 16 attention heads and head_dim=128, while Transformers v4.57.1 native Phimoe computes head_dim as hidden_size/num_heads (256). Therefore this checkpoint is not interchangeable with that native class. The loader now rejects this geometry before model weight loading. Phi-mini must be checked from its actual local config; its geometry is not assumed from the tiny variant.

Sources:
- https://huggingface.co/microsoft/Phi-tiny-MoE-instruct/raw/main/config.json
- https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/phimoe/modeling_phimoe.py

A custom official model implementation can be used only after a separate review. The following creates a **pending** inventory without importing model Python:

```bash
PYTHONPATH=src python scripts/review_local_model_code.py "$PHI_LOCAL_CHECKPOINT" local-code-review.pending.json
```

Inspect every recorded Python file and its imports, source revision, dependencies and initialization. This is executable local code and the review manifest is not a security sandbox. After real review, create a separate approval with `LOCAL_CODE_REVIEWED`, reviewer and notes, retaining the exact hashes. The HF loader accepts `--reviewed-local-code /path/to/reviewed-receipt.json` only when those hashes and checkpoint identity match. It then permits the reviewed local code with local-only loading; there is no automatic fallback and no download. The existing simple Furnace script does not enable this route. Use the explicit CLI in a dedicated compatible environment, with the same smoke budgets and a new output directory.

`output_loading_info=True` is checked strictly: missing, unexpected or mismatched weights fail, rather than silently retaining newly initialized parameters. API compatibility of custom SlimMoE code with the candidate HF environment is still untested and may need a separately qualified environment. The native small-Phimoe class test cannot stand in for this check.
