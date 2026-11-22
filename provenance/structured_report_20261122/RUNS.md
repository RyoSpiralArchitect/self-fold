# Reproduction

The writer fit and held-out reports ran at [f4b151a](https://github.com/RyoSpiralArchitect/self-fold/tree/f4b151a6d95dc4398b182d32adf69e39e77c1485). The saved reports were scored at [700e7ca](https://github.com/RyoSpiralArchitect/self-fold/tree/700e7ca2db4c5f5b88ed39e9aaa9b8e4b7cea8be). The fixed plan, initial input checksums, trained weights, schedule, reports and native margins accompany the result. Gemma-derived artifacts are subject to the [included terms](ARTIFACT_LICENSE.md).

From the results checkout, replay all measurements on CPU and compare with the independent calculation:

```sh
study=provenance/structured_report_20261122
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/fit_structured_report.py --reduce "$study/produce"
python "$study/verify.py" "$study/produce"
(cd "$study" && sha256sum -c SHA256SUMS)
```

To repeat the fit with a new output directory, use a clean checkout of the scoring commit. Provide the local Gemma-2-2B-IT snapshot specified in `configs/models/gemma2b.yaml` and the pinned historical reader environment (Torch 2.13.0+cu132 and Transformers 4.57.1). The primary generation policy is greedy, up to native EOS or 160 new tokens. The recipient uses its separately registered historical host identity. The measurement requires one CUDA GPU.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/fit_structured_report.py --check
python scripts/fit_structured_report.py --run "$NEW_OUTPUT" --checkpoint "$GEMMA_CHECKPOINT" --seal 700e7ca2db4c5f5b88ed39e9aaa9b8e4b7cea8be
```

For a recovery of the first fit, the same runner also accepts `--from-stopped "$ORIGINAL_STOPPED_RUN"` with a fresh `--run` directory. It verifies `INITIAL.json`, the original stop, and all reports before executing only the frozen recipient and reducer. The original terminal receipt is included in `INITIAL.json`; the other nine original file contents are byte-identical in `produce/`.
