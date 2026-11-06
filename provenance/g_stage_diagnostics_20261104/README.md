# Stage audit of the supported G experiment

Post-result, development-only reanalysis of PR #27 at
`ba20b3125ac498ef1f2fa21730eef3557e3dd953`. No model forwards or training were
performed. The original qualification groups remain unopened and the gate
remains `NOT_QUALIFIED`.

[Detailed Japanese report](RESULTS.ja.md) ·
[Prospective development specification](../../docs/G_STAGE_DEVELOPMENT_PREREGISTRATION.md)

The structured updater exactly reproduces formal interpretation of the same
inferred premises. Remaining errors separate into source-field errors,
disagreement between soft-row behavior and hard-rendered action content, and
literal reading failures. On fresh wording, 224 native-correct decisions have an
incorrect global-argmax action row; 1,436 literal failures occur even on correctly
rendered rows. Equal G/re-encoded choices do not imply equal states or scores.

All 1,024 native identity, each native G, and encoded-teacher predictions select
A. The boundary increase from 503 to 507 consists entirely of previously tied
A/B scores becoming non-ties, with no answer changes or repairs on affected
worlds. All 128 E/boundary state hashes differ. Thus, numerical updating exists,
while semantic updating cannot be separated from the failed carrier/reader
interface using the archived scores alone.

The exported primary overlap joins 128 identical authored policies and 1,024
semantic questions, while explicitly recording different physical questions,
readers and injection sites. It does not estimate a G-only causal difference.
Native D/ED cells outside the original 16-policy subset remain missing.

## Reproduce

Use a Git checkout containing the pinned base commit and NumPy. No Torch,
Transformers, model download, qualification loader or GPU is needed.

```bash
python provenance/g_stage_diagnostics_20261104/audit.py --out /tmp/self-fold-g-stage-replay
python tests/test_g_stage_diagnostics.py -v
```

`--out` must not exist. The audit verifies the original checksum manifest,
reproduces all three original summary reductions and both original field
diagnostics, and checks declared inputs against the pinned Git blobs before
writing a new output directory. It emits deterministic JSON, gzip JSONL and CSV.

The new experiment document is a design, not an executed experiment or a
retroactive registration. Its execution manifest must be frozen before any
additional measurements. It schedules only missing soft/hard, native
intermediate-state and common-question comparisons, retaining the original
results and preprint v1 claims.

## Interpretation contract

- Full-vocabulary winner and finite nonzero A/B margin reproduce the original
  scoring. Tie-excluding totals are not silently replaced by token-only totals.
- Source-policy correctness and fidelity to the emitted text are separate.
  Off-menu emitted actions have no correct literal-fidelity A/B answer.
- Literal reading and re-encoding branch from the same emitted text. Re-encoding
  does not consume the literal reader's A/B response.
- Native state hashes prove numerical changes, not semantic correctness or its
  absence. The archive contains no raw writer/transport activation arrays.
- Shared integer case keys are insufficient: source wording, question hash,
  reader, geometry and scoring must be checked.
- Counts are descriptive observations of known menus, with repeated worlds,
  orders and styles; they are not independent population samples.

See [comparison_contract.json](comparison_contract.json),
[verification.json](verification.json), and
[input_manifest.json](input_manifest.json) for machine-readable provenance.
Re-exported model records retain the [upstream artifact conditions](ARTIFACT_LICENSE.md).
