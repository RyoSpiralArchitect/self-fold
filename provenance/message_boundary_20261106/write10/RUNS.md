# Execution and replay

Source seal: `0d9f4c103c933615aa9af409dd46f961a3f83660`, a clean isolated
Furnace checkout. The parent is write9 at
`44b106a918ea7d963626b6cdcd6a897f752b1aae`. Both scientific jobs started only
after the plan, implementation, fixtures and input hashes were committed.
The source seal's push and pull-request CI receipts are in `source-ci/`.

Use the repository root as the working directory and choose **new**, absent
output directories. The original runs used Python 3.14.4, PyTorch 2.13.0+cu132,
NumPy 2.5.4 and Transformers 4.57.1 on Linux. All computation was on CPU;
another workload occupied the GPU. `nice -n 10` was applied to both jobs.
The 176 files in `INPUTS.json` bind the previous reports, models and native
choice scores. No independent qualification material is loaded.

```bash
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# CHECKPOINT is the local Gemma-2-2B-IT snapshot named in host.json.
# OUT is a new experiment directory outside the source checkout.
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 nice -n 10 timeout 3700 \
  python scripts/measure_actual_report.py \
  --out "$OUT/reports" --checkpoint "$CHECKPOINT"

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 timeout 3600 \
  python scripts/fit_reference_retention.py \
  --out "$OUT/retention" --workers 2

# Run only after the two terminal receipts say that their panels are closed.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 \
  python scripts/compose_actual_report.py \
  --reports "$OUT/reports" --retention "$OUT/retention" \
  --out "$OUT/composition"
```

The report run caps model-host wall time at 3,600 seconds internally. Both
question arms use CPU float32, greedy KV decoding, 1,024 context tokens and
at most 48 new tokens, ending earlier at EOS. Censored records are retained
and content-scored. The two neutral timing generations in `cpu-preflight.json`
are excluded from the 448 scientific report records.

Fresh processes independently reduce the saved artifacts. These commands
generate no language and make no new native scoring calls:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/measure_actual_report.py \
  --reduce "$OUT/reports"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/fit_reference_retention.py \
  --reduce "$OUT/retention"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/compose_actual_report.py \
  --reports "$OUT/reports" --retention "$OUT/retention" \
  --reduce "$OUT/composition"
```

The composition reducer reruns every frozen trajectory and compares all saved
arrays. The retention reducer checks file and logical-array hashes and
recomputes audits, summaries and intervention comparisons. The report reducer
checks request fingerprints and regenerates the report summaries and frozen
continuation judgments from the saved text. These are computational checks,
not independent semantic annotation or a replication on new tasks.
