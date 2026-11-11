# Execution and independent reduction

The scientific source seal is `2d835bf933db6b9e32c015cada23a9f20d9fb75d`,
based on write10 at `1205bc4675d9d5ec9edd7dc515d12c2642ff8ee7`. The source
worktree must be clean; write outputs into a new directory outside it.

The two neutral GPU timing generations are in `gpu-preflight.json`. They
verify resource operation and native hook parity, not report quality. No
policy-bank question was used to choose between prompt variants.

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 timeout 3900 \
python scripts/measure_source_selection.py \
  --out /path/to/new-run --checkpoint /path/to/pinned/gemma-2-2b-it
```

The JSON design pins CUDA bfloat16, one CPU thread, context 1,024 and at most
48 new tokens, ending earlier at EOS. The internal model-host wall budget is
3,600 seconds. Both conditions are generated under the same KV argmax policy.
The run releases its model/GPU allocation before computing the frozen probes
and compositions on CPU. Another active workload must not be displaced.

The receipt includes the exact requests, raw outputs, token/stop receipts,
model identity, complete summaries and every frozen trajectory. The reference
cells and readers are loaded from the 311 pinned prior files, with no fitting.
Missing report-bank coverage is a protocol error; capped or imperfect text is
retained and scored normally. A technical stop is preserved, never silently
replaced by a successful rerun.

Replay in a fresh process using only the saved reports and bound native scores:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/measure_source_selection.py --reduce /path/to/completed-run
```

The reducer regenerates the full request panel, validates request fingerprints,
recomputes report gates and 21,504 continuation judgments, reruns all 216 frozen
compositions, and checks all arrays against file hashes and logical-array
hashes. It makes zero language-generation or native-scoring calls. This is a
computational replay, not independent semantic annotation or new-task evidence.

The separately labeled post-result agreement diagnostic is reproduced with:

```bash
python scripts/audit_source_selection.py /path/to/completed-run \
  --out /path/to/new-agreement-audit.json
```

Pass `--verify` with an existing audit path to check it in a fresh process.
This audit does not alter the preregistered source, model outputs or gates.
