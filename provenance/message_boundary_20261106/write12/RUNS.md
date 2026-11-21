# Reproducing the row-readout comparison

The scientific runner used source commit
`c5b2d3e79f927b191b445574fe153b54a0f7da4e`. For a new generation run,
check out that commit in a clean worktree and set `GEMMA_CHECKPOINT` to the
local Gemma-2-2B-IT snapshot specified by the model profile. Set
`NEW_OUTPUT` to an unused directory outside the worktree. From its root:

```sh
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/measure_oracle_row_readout.py --check-design
nice -n 10 timeout --kill-after=5s 2700s python -u scripts/measure_oracle_row_readout.py \
  --run "$NEW_OUTPUT" --checkpoint "$GEMMA_CHECKPOINT" \
  --seal c5b2d3e79f927b191b445574fe153b54a0f7da4e
```

The design check verifies the input hashes and regenerates all prompts. The
scientific run checks local checkpoint identity and native hook parity, then
logs each generation in execution order. The output directory is created
exclusively. A terminal technical stop preserves any completed rows.

To reproduce the published scoring from the repository root on CPU, use the
results checkout and its saved run. The first command regenerates the exact
instrument and all registered scores, panels, paired comparisons and the
decision. The second parses the text Gemma actually saw and verifies the
coordinate, action, position, complete tables and single-record swap counts
independently of the bit-mask scorer.

```sh
study=provenance/message_boundary_20261106/write12
PYTHONPATH=src python scripts/measure_oracle_row_readout.py --reduce "$study/produce"
python3 -S "$study/verify_visible_rows.py" "$study/produce"
(cd "$study" && sha256sum -c SHA256SUMS)
```

`produce/requests.jsonl.gz` contains one exact prompt per line.
`produce/reports.jsonl` contains the prompt fingerprint, response and stop
receipt. `produce/scores.json` uses the same question IDs.
`produce/summary.json` records each gate and decision. The recorded replay
and literal verification are `reduction.json` and `visible_rows.json`.
