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

## Replaying the prospective separator design

The separator preflight binds the complete historical baseline tree and the test bytes at its study seal. To replay it from a later checkout, use a clone with full Git history.

~~~bash
git worktree add --detach ../self-fold-separator-seal 66e499f30317b97b98527078f1a435ddbe36067c
(cd ../self-fold-separator-seal && python3 provenance/separator_factorization_20261101/verify.py --check)
~~~

## Paired Gemma fitting and clause-order geometry

The two fits share the same train-only schedule and use the fixed step-64 endpoints. The geometry run uses the frozen base and the earlier capture hashes. Run each at its own source seal in a clean checkout with the pinned local Gemma snapshot and recorded dependencies. Each command needs a fresh ignored output directory.

~~~bash
# Source 5e4bd86f9ca45fd0e3f0192fb165cc3eac76c7e5
PYTHONPATH=src python scripts/fit_policy_binding.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/paired-dev

# Source 65457166eb22d450802c2cc8f18ba09560ddb44a
PYTHONPATH=src python scripts/probe_policy_geometry.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/geometry-dev
~~~

The [study record](../provenance/policy_binding_20261103/RESULTS.md) includes both reducers, raw decisions, the generation rule for the crossed panel and the registered endpoints. These runs require no qualification inputs.

## Integrated continuity development phases

The corpus construction and native receiver learning are sealed at separate source commits. All commands use the pinned Gemma checkpoint and existing environment versions. Use a clean checkout at the corresponding source and fresh ignored run directories. The later phases bind their original completed receipts.

~~~bash
# Source 77c4ab6623cd843b730bc2b0635b62f998aef1c5
PYTHONPATH=src python scripts/fit_continuity_recipient.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/recipient

# Source 35c71a1aada6e2503e02e98e64f45ddba9b60706
PYTHONPATH=src python scripts/audit_continuity_recipient.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --fit runs/recipient --out runs/numerical-audit

# Source 1dc37cdaee70212bd2f449931e1bfde2adecaf2e
PYTHONPATH=src python scripts/fit_continuity_recipient.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --objective native_binary --baseline-fit runs/recipient --audit runs/numerical-audit --out runs/recipient-binary

# Source 7f284c18ec4f5f330df86f8190c081a60422d497
PYTHONPATH=src python scripts/fit_joint_continuity.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --baseline-fit runs/recipient --out runs/joint

# Source d316bce3f35d8de3aa025683d022e2b84fe17218
PYTHONPATH=src python scripts/probe_recipient_trainability.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --baseline-fit runs/recipient --out runs/trainability
~~~

The [integrated study](../provenance/continuity_20261103/RESULTS.md) preserves source hashes, all decisions, intermediate numerical observations, learned arrays and Gemma terms. Each of the four candidate fits ended at its cap without admitting the recipient. The final paired control measures small-set trainability. None trains G or opens the sealed qualification material.

## Literal-first conditional curriculum and margin continuation

These two studies have separate source seals and preserve the first closed gate. The margin continuation starts from the archived warm task adapter, with its exact hash enforced by the entrypoint. Each run needs the pinned Gemma snapshot and runtime from its `run/host.json`, a clean checkout at its source seal and a fresh output directory. No qualification inputs are required.

~~~bash
# First curriculum source 94473d4c775a01c9e60cf83ccccdf615bf073121
PYTHONPATH=src python scripts/fit_conditional_curriculum.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/conditional-curriculum

# Margin continuation source d5d152f4dfae21f0fccf60fe864a5e0e90cf84f7
# Uses the published first run's warm_task.npz as the fixed parent.
PYTHONPATH=src python scripts/fit_conditional_margin.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/conditional-margin
~~~

The second source includes the first run's arrays and complete receipts. A reproduction of the first warm fit is a separate check; the continuation binds the published parent rather than silently replacing it with a newly fitted endpoint. Recompute every study's scores and verify its public file manifest:

~~~bash
(cd provenance/conditional_curriculum_20261103 && shasum -a 256 -c SHA256SUMS && python3 -S summarize.py | diff - summary.json)
(cd provenance/conditional_margin_20261103 && shasum -a 256 -c SHA256SUMS && python3 -S summarize.py | diff - summary.json)
~~~

The [first plan](../provenance/conditional_curriculum_20261103/PLAN.md) and [continuation plan](../provenance/conditional_margin_20261103/PLAN.md) specify literal fitting gates, conditional source-learning arms, full-vocabulary native answers and all-world strict units. Their result records retain any unattempted branch. Source-only adaptation follows the ordinary encoder's exact native prompt length and uses mutually exclusive source and task scopes. It is a separately learned interface and does not extend the earlier frozen-role claim.

## From two flags to an action

Use clean execution source `9913c3ffd2c827e04faaa32076435f10522cebda` and the pinned Gemma runtime recorded in the [study](../provenance/two_flags_20261104/RESULTS.md). The two arms load the same published margin-continuation adapter, use a single training schedule, and close their fixed endpoints before fresh scoring. The reducers run without model inference.

~~~bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src python scripts/fit_two_flags.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/two-flags
python3 -S provenance/two_flags_20261104/summarize.py runs/two-flags
(cd provenance/two_flags_20261104 && shasum -a 256 -c SHA256SUMS && python3 -S summarize.py | diff - summary.json)
~~~

## Change a policy after discarding its words

Check out the prospective source `effaaef64af873f8e32342049470558bfeb8c0a9` to reproduce the fit. The second command runs in a fresh process and takes only the published numeric packet directory plus the original Gemma checkpoint. The grader joins the predictions to their labels after inference. The released study directory can also be graded directly without a GPU. [Design and cutoffs](../provenance/state_only_20261104/PLAN.md)

```bash
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/fit_state_only.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/state-only/produce
python scripts/decide_state_only.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --packet runs/state-only/produce/packet --out runs/state-only/decide
python provenance/state_only_20261104/summarize.py runs/state-only
```

For the separate, zero-training reuse diagnostic, use source `0405d2bce6bb9e60432ed7d1b1618558e4808bee` and the released one-edit run bytes. Prepare a new packet from the complete outputs of the learned first update, then use the identical blind decision program. [Composition protocol](../provenance/state_only_20261104/COMPOSITION_PLAN.md)

```bash
python provenance/state_only_20261104/compose.py prepare provenance/state_only_20261104/run runs/policy-chain
python scripts/decide_state_only.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --packet runs/policy-chain/packet --out runs/policy-chain/decide
python provenance/state_only_20261104/compose.py grade provenance/state_only_20261104/run runs/policy-chain
```

The source seal predates the public result files. For a new checkout at the seal, supply the immutable published one-edit run at the first path in the example or pass its external path to the preparation and grading commands. To verify the published numbers directly, run each grader on `provenance/state_only_20261104/run` and, for the follow-up, `provenance/state_only_20261104/chain`.

The fixed role-recovery diagnostic uses source `4b8b54bcf1970e11c7f95337d6fa3bd6702ba525` and the unmodified one-edit and chain artifacts. The restored and deliberately swapped layouts go through the same blind native consumer. [Recovery design](../provenance/state_only_20261104/ROLE_RECOVERY_PLAN.md)

```bash
study=provenance/state_only_20261104
python "$study/role_recovery.py" prepare "$study/run" "$study/chain" runs/policy-recovery
for mode in restore_roles exchange_roles; do
  python scripts/decide_state_only.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --packet "runs/policy-recovery/$mode/packet" --out "runs/policy-recovery/$mode/decide"
done
python "$study/role_recovery.py" grade "$study/run" "$study/chain" runs/policy-recovery
```

## Learn a role bit on two-edit trajectories

The frozen design and execution code are at `00104f747316d77f91e4aa143eed0231272e63b9`. The first command fits all nine controllers before compiling the new evaluation descriptions. The second starts a fresh Gemma process with only numerical rows, menus and bindings. A lossless row index for each actual continuous output joins it to its native decision after scoring. The reducer can also run directly on the published study directory. [Prospective protocol](../provenance/role_memory_20261104/PLAN.md)

```bash
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/fit_role_memory.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/role-memory/produce
python scripts/role_memory_native.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --packet runs/role-memory/produce/packet --out runs/role-memory/decide
python provenance/role_memory_20261104/summarize.py runs/role-memory
```

The parity follow-up uses source `cf946256111981945732b40ab4106344e8a93bb3` and the exact published original run. It verifies a fresh native replay before simulating every four-edit string using the original soft rows and endpoint weights.

```bash
study=provenance/role_memory_20261104
python scripts/role_memory_native.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --packet "$study/run/produce/packet" --out runs/role-memory-replay
python "$study/all_orders.py" "$study/run" runs/role-memory-replay runs/role-memory-all-orders
```

## Vary the grammar and train two frozen-reader transitions

The literal continuations execute at `5fac1bd10591b6227ec1c6fb6cdfcef989a7cb30`. All three start from the published component reader and complete their endpoints before native scoring. The state-map execution source is `b8c4511b9334b3e4b5dd69235f2e757e4a595b06`. It binds the focus reader from the first run and captures only source text before later questions. Set the same pinned local Gemma checkpoint and offline environment as above. Use clean checkouts of the respective seals.

```bash
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/fit_stage_grammar.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --out runs/stage-grammar/literal
python provenance/stage_grammar_20261104/summarize.py runs/stage-grammar/literal
python scripts/fit_stage_transitions.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --literal runs/stage-grammar/literal --out runs/stage-grammar/maps
python provenance/stage_grammar_20261104/score_chain.py runs/stage-grammar/maps
```

The published reductions can also be rerun on CPU by replacing the two run directories in the grading commands with `provenance/stage_grammar_20261104/literal` and `provenance/stage_grammar_20261104/maps`.

A separately sealed fixed continuation backpropagates the native losses at both stage outputs through the unchanged focus reader. Use execution source `beb453d9c6990ab59db9f00c7fd6c3554493286e`. It takes the completed feature-only map and literal runs at their original bindings. The readback uses the saved BF16 states. [Native-gradient objective and cap](../provenance/stage_grammar_20261104/NATIVE_GRADIENT_PLAN.md)

```bash
python scripts/guide_stage_transitions.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --literal runs/stage-grammar/literal --maps runs/stage-grammar/maps --out runs/stage-grammar/guided
python provenance/stage_grammar_20261104/score_chain.py runs/stage-grammar/guided
python scripts/replay_stage_transitions.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --literal runs/stage-grammar/literal --maps runs/stage-grammar/guided --out runs/stage-grammar/replay-guided
```

The post-result wording cross and historical batch-geometry audit use source `c6a710a8074e60b6197dd5ac69ed619df63de445` and the fixed focus endpoint. They perform zero training. [Crossed conditions and comparisons](../provenance/stage_grammar_20261104/WORDING_AUDIT.md)

```bash
python scripts/audit_stage_wording.py --checkpoint "$GEMMA_LOCAL_CHECKPOINT" --literal runs/stage-grammar/literal --out runs/stage-grammar/wording
```
