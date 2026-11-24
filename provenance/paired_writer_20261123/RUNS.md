# Reproducing the paired report-writer study

The scientific source was sealed at [f2a5667](https://github.com/RyoSpiralArchitect/self-fold/tree/f2a566700a0d0905bc233245cf1516881769bc6d). The parent adapter, G1, G2, text parser, report instructions and Gemma recipient come from the preceding [structured report experiment](../structured_report_20261122/RESULTS.md).

To reproduce the stored outcomes from the results checkout:

```sh
study=provenance/paired_writer_20261123
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/fit_paired_role_writer.py --reduce "$study/produce"
python "$study/verify.py" "$study/produce"
(cd "$study" && sha256sum -c SHA256SUMS)
```

For a new fit, use a clean checkout of the scientific source, the local Gemma-2-2B-IT snapshot in `configs/models/gemma2b.yaml`, and the original native recipient environment (Torch 2.13.0+cu132 and Transformers 4.57.1). Set NEW_OUTPUT to an unused directory outside the checkout. The run scores the frozen recipient first, fits both adapters, saves their weights and then generates all reports.

```sh
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/fit_paired_role_writer.py --check
python scripts/fit_paired_role_writer.py --run "$NEW_OUTPUT" --checkpoint "$GEMMA_CHECKPOINT" --seal f2a566700a0d0905bc233245cf1516881769bc6d
```

Each line of `produce/reports.jsonl` records an arm, a held-out presentation, its prompt fingerprint, the response tokens, text and stop condition. Use `prepare()` from the runner to regenerate all prompts. The per-arm per-step traces retain every example index. Each weights archive contains only the writer adapters. The two native score banks contain the full-vocabulary winning token and the A/B logits for each scored row, question menu and label order. The saved outcomes include actual G2 output rows and all three parts of the joint score.
