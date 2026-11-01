# Reproducing the experiments

Start from the Git commit identified in the trial's `source.json`. Install the project in its own Python environment. The pretrained experiments record their actual PyTorch, Transformers and CUDA versions. The random-configuration HF tests exercise four decoder layouts without fetching weights.

~~~bash
python -m pip install -e '.[dev,hf]'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python -m pytest -q
python -m selffold inventory --out runs/inventory-001.json
~~~

## Pretrained recurrence

The Mistral comparison used an existing Mistral-7B-Instruct-v0.3 snapshot on one RTX 5090 in BF16 with eager attention. Use a local directory containing its configuration, tokenizer and safetensors shards. The loader hashes the checkpoint and checks for missing, unexpected and mismatched parameters. The exact revision and a command for each experimental condition are in the [Mistral plan](../provenance/mistral_eos_20261101/PLAN.md).

~~~bash
bash scripts/furnace_smoke.sh configs/models/mistral7b.yaml \
  "$MISTRAL_LOCAL_CHECKPOINT" runs/mistral-eos-001
~~~

This CUDA command runs two plain updates in T/L/B. Greedy generation stops at native EOS or the configured context boundary. An explicit `--max-new-tokens N` adds a shared ceiling. Receipts record the effective budget, content length, actual EOS id and stop reason. Nonempty censored outputs remain in the trial.

The [profiles](../configs/models/) describe admission floors and model layouts. MoE memory use depends on resident parameters. Some Phi checkpoints need reviewed local Python because their attention geometry differs from the native Phimoe implementation. A hash-bound review can be prepared with `scripts/review_local_model_code.py`. The default loader uses local files with remote code disabled.

## Inspect a saved run

Read `request.json`, `host_identity.json` and `terminal.json` together. Each transition separates main-path generation from readout on a clone. Tasks are scored after recurrence. JSON metadata and NPZ arrays retain state dtype and fingerprints. The original trial directories are immutable. Run a reproduction in a fresh directory and record any failure there.

~~~bash
(cd provenance/mistral_eos_20261101 && shasum -a 256 -c SHA256SUMS)
(cd provenance/mistral_eos_20261101 && python summarize.py | diff - summary.json)
~~~

The exposed development tasks, numeric state differences and random-weight checks have separate roles. Interpreting a latent trajectory as a semantic update requires the [qualification checks](QUALIFICATION.md).
