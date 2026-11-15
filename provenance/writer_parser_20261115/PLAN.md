# Writer-vs-parser separation: an extractable-information ceiling on the saved report corpora

**Status: development measurement. Not a qualification run. No new generation, no model fitting inside the boundary. Reads only sealed artifacts.**

## Question

`message_boundary_20261106` left one comparison undecided: spontaneous
(`rt_natural`, `state_prior_true`) reports carried the off/off bit at ~7–14%
end-to-end while directed reports reached the ceiling. The recorded audit
categories are *decoder-consensus* measurements on a shared lexical scoring
family; all-decoder missingness does not prove the writer omitted the bit, and
write7's raw-text reader was too weak to arbitrate (near-flat reward curves,
512 updates, 32-dim hashbag).

This experiment separates the two explanations on text alone:

- **Writer failure**: no representation of the off/off-role-violation bit is
  recoverable from the report text by *any* procedure.
- **Parser/family failure**: the bit is recoverable by some procedure but not
  by the shipped decoders.

The decision procedure is an *ideal-parser ceiling*: a supervised classifier
with unrestricted lexical access to the text is itself a parser. Whatever it
cannot extract, no lexical decoder extracts either.

## Data (read-only)

`provenance/message_boundary_20261106/write6/produce/reports_*.jsonl`
(sealed; `lm_matrix_rt.json` already corrected post-review).

| corpus | n | ground truth |
| --- | ---: | --- |
| `rt_natural` | 12,992 | `out & 1` (realized) and `mask & 1` (instructed) |
| `rt_directed` | 12,992 | same — sanity arm, expected ceiling |
| `state_prior_true` | 14,336 | `mask & 1` |

The text itself names one of the two source **options**, and every shipped
decoder recovers the bit as `named_slot ^ pole`. The classifier target is
therefore the *named slot* — `slot_out = (out&1) ^ pole` and
`slot_mask = (mask&1) ^ pole`. Classifier accuracy on the slot is exactly the
parser ceiling on the bit; the option↔pole binding is declared side
information, identical to what `decode_state_slot` receives.

## Estimators

For each corpus and each target bit:

1. **Word n-gram TF-IDF + logistic regression** (1–2 grams).
2. **Char n-gram TF-IDF + logistic regression** (3–5 grams).
3. **Word n-grams + gradient-boosted trees** (nonlinear interactions).

Cross-validation schemes, strictest to loosest:

- `group-topic`: GroupKFold by source group index — wording must transfer
  across topics (the boundary use case).
- `group-text`: GroupKFold by unique raw text — held-out exact strings.
- `random`: stratified KFold — loose upper bound, memorization allowed.

Targets: `out&1` and `mask&1` separately, plus accuracy conditioned on
`out == mask` vs `out != mask` (the realized-vs-instructed divergence the
write6 audit exposed).

Null calibration: 200 label permutations on the random split; report the
permutation max distribution vs the real score.

Sanity: `rt_directed` must reach ≥99% under every estimator/split or the
pipeline is declared broken and no other number is interpreted.

## Independent annotation

A stratified sample of unique normalized forms (option names → `<A>`/`<B>`,
coordinates → `<ROW>`, same normalization as `audit_roundtrip.form_of`)
covering the `unresolved_all_missing` mass is annotated by an annotator that
shares no code with the lexical decoder family (the agent, recording each
form and verdict in `annotation.jsonl`). Verdicts: `bit_determinable`,
`bit_ambiguous`, `bit_absent`. Coverage-weighted frequencies are reported.

This annotation is descriptive: it bounds what *any reader who can see the
options* could extract, and cross-checks the classifier ceiling.

## Interpretation pre-commitment

- If every estimator stays at the null band on `group-topic` → the bit is
  not lexically present with topic-independent wording; consistent with
  writer failure *at this prompt and bank*. It does not prove unwritability
  (a different prompt may elicit it) and says nothing about non-lexical
  signal (stage 2).
- If the ceiling is high on `random`/`group-text` but collapses on
  `group-topic` → the information exists but is wording-memorized; a fitted
  parser family could in principle recover it on familiar topics.
- If the ceiling is high on `group-topic` → parser/family failure is
  demonstrated; Experiment B (richer frozen features / fitted reader on raw
  text) is justified and gets a concrete target.

## Stage 2 (conditional, declared)

If the lexical ceiling is low on a corpus, re-encode each report with the
frozen Gemma-2-2B-IT host (`checkpoint f3503907…`, furnace) and fit the same
three estimators on the E representation. A positive E-probe over a negative
lexical ceiling would locate non-lexical signal and directly design
Experiment B's feature set. A negative E-probe strengthens writer failure.
This stage needs the pinned checkpoint and is executed separately; its run
gets its own entry in this RESULTS.

## Falsification / honesty notes

- Classifier accuracy is an *upper bound on decoders*, not a writer-fidelity
  measure: the classifier may exploit confounds (e.g., reports whose wording
  correlates with `mask` when `out == mask`). Targets are therefore reported
  on both `out&1` and `mask&1` and stratified by realized divergence.
- A high ceiling with a confound path does not rescue the boundary channel
  by itself; it only shows the information crossed into the text.
- Agent annotation is a single annotator; disagreement sources are reported
  per form, not adjudicated.
