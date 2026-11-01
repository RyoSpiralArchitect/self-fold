# Executed verification receipts

Date: 2026-11-01. These are completed local CPU checks, not promised remote work.

## Frozen source

Source commit: `98992ec9bedfaf2d96680d508dc7d017ffb3bdec`. Both sealed smoke runs recorded a clean worktree and the same source manifest. The later receipt commit adds evidence only and does not change the tested source files.

## Executed

| Check | Actual result |
|---|---|
| Pytest | **81 passed, 4 skipped**, 85 collected, zero failures/errors |
| HF family checks | All four **skipped** because Transformers is absent; not success |
| Plain smoke | T/L/B, two updates each: **6 transitions**, 12 task scoring rows, one shared-state branch |
| Expansion smoke | T/L/B, three updates each: **9 transitions**, 24 task scoring rows, one shared-state branch |
| Native observation-hook parity | Exact in both random tiny decoder runs |
| Safe state restoration | Tested for FP16, BF16, FP32 and FP64; tampering rejected |
| Fresh-process continuation replay | Exact tiny-state update SHA equality |
| Bibliography/source checks | 13 entries; LaTeX citation keys resolve; source audit present |
| Python / shell syntax | compileall and bash -n passed before the source seal |

Both smoke terminal statuses are `ENGINEERING_PASS_NOT_SEMANTICS`. These runs use an actually executed, randomly initialized causal decoder, **not a pretrained LLM or a semantic simulator**. Generated text and toy scores are not evidence of useful semantics or a reasoning gain. Numeric traces are preserved only to audit the plumbing.

## Reproduce from repository root

```bash
PYTHONPATH=src python -m pytest -q --junitxml=runs/new-unit-tests.xml
PYTHONPATH=src python -m selffold smoke --backend tiny --allow-uncalibrated \
  --steps 2 --out runs/new-plain --tasks data/tasks.dev.jsonl --task-limit 2 --branch
PYTHONPATH=src python -m selffold smoke --backend tiny --allow-uncalibrated \
  --steps 3 --regime expand --out runs/new-expand --tasks data/tasks.dev.jsonl --task-limit 4 --branch
```

[Machine-readable summary](local_cpu_98992ec/summary.json), [unit-test XML](local_cpu_98992ec/checks/unit-tests.xml), [file hashes](local_cpu_98992ec/manifest.sha256.json). `plain/` and `expand/` contain raw state/activation NPZ, JSON outputs, environment and source identities. Safe state files can be inspected with `selffold.records.load_state`; no model weights are included.

## Explicitly not executed

Furnace access, CUDA testing, Mistral/Llama/Gemma/Phi pretrained inference, code-reviewed custom SlimMoE loading, semantic gate qualification, confirmation evaluation, GitHub remote creation/push, GitHub CI and LaTeX compilation. The four-family HF tests remain an actual next check, not a completed validation.

`provenance/unit-tests.xml` preserves the earlier unsealed construction check (66 passed and 4 skipped) and is not the final receipt. Differences in that count reflect added tests, not suppressed failures.
