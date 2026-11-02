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

## Description-only carriers and codec learning

Use the source commit in each record. `scripts/probe_carriers.py` fixes two carrier routes on the twelve exposed topics and verifies every native tokenizer layout before loading weights. `scripts/fit_codec.py` performs the fixed Gemma codec-only fit on public train/dev files. The qualification commitment is public, while qualification content is neither required nor opened for this fit. Public group JSON also contains proposed G targets: the parser reads those fields, but the codec iterator supplies only source-copy examples, and G targets do not enter model inputs or the loss.

~~~bash
PYTHONPATH=src python scripts/probe_carriers.py --model gemma2b \
  --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/carriers-fresh
PYTHONPATH=src python scripts/fit_codec.py --capture-route workspace \
  --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/codec-fresh
~~~

`provenance/carriers_20261102/audit_codec.py` independently replays the fitted NLLs and the fixed opposed/zero-donor controls. Learned arrays use safe NPZ files and bind their base-host identity, geometry, source and data. A frozen artifact can be installed with `--encoder-bridge` only under the exact saved host identity, including context length and runtime versions; semantic admission is still required for a latent pilot. [Carrier plan](../provenance/carriers_20261102/PLAN.md) · [Learning contract](../provenance/carriers_20261102/LEARNING_CONTRACT.md) · [Results and integrity](../provenance/carriers_20261102/RESULTS.md)

## Saved evidence

For the archived-state and two-checkpoint probes, start from the commit recorded for each study. The probe commands use the same frozen fixtures and local snapshots across the label, position and replay controls. [Content probe commands](../provenance/slot_content_20261101/PLAN.md) · [Lexical donor commands](../provenance/slot_content_20261101/ALIGNMENT_PLAN.md) · [Replay commands](../provenance/slot_content_20261101/REPLAY_PLAN.md)

The downstream localization study uses two more source revisions. Select the original or prospective six-topic panel with `--panel`. The `localize` runs partition the full span and the `clauses` runs isolate the displayed option words and periods. Their supplement regenerates the reported fractions directly from raw A/B likelihoods. [Localization commands](../provenance/downstream_span_20261101/PLAN.md) · [Clause and masking commands](../provenance/downstream_span_20261101/CLAUSE_PLAN.md)

Read `request.json`, `host_identity.json` and `terminal.json` together. Each transition separates main-path generation from readout on a clone. Tasks are scored after recurrence. JSON metadata and NPZ arrays retain state dtype and fingerprints. The original trial directories are immutable. Run a reproduction in a fresh directory and record any failure there.

~~~bash
(cd provenance/mistral_eos_20261101 && shasum -a 256 -c SHA256SUMS)
(cd provenance/mistral_eos_20261101 && python summarize.py | diff - summary.json)
~~~

The exposed development tasks, numeric state differences and random-weight checks have separate roles. Interpreting a latent trajectory as a semantic update requires the [qualification checks](QUALIFICATION.md).
