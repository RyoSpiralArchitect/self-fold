# Description-only carriers and the first codec fit

The bounded native-interface diagnostic closed without a passing route. We then completed the first **codec-only learning run** under the negative-closure route in the [six-package plan](../../docs/SEMANTIC_GATE_PLAN.md). Source-reconstruction loss improved, while functional content identification remained near 50%. No semantic gate or latent pilot was admitted.

## Four carrier cells

All 48 source descriptions per route and checkpoint were encoded before constructing questions. Each saved state was reused unchanged for a topical question and a generic matching question, in both option orders. Both carriers have eight rows and a 256-position donor forward. Workspace captures the original E slots; native tail captures the last eight source-token positions in a description-only prompt and is a new interface.

Each accuracy averages within each of 12 meaning groups, then across groups. There are 192 dependent readouts per cell. The strongest fixed null is the maximum group accuracy among uninjected, zero, donor-independent random and other-topic states. Reversed rows and norm-only states are reported separately in the [full summary](summary.json).

| Checkpoint | Carrier | Accuracy | Strongest null | Literal source | Distinct D strings / 48 |
| --- | --- | ---: | ---: | ---: | ---: |
| Mistral-7B | Workspace | 50.0% | 50.0% | 100.0% | 1 |
| Mistral-7B | Native tail | 50.5% | 49.5% | 100.0% | 45 |
| Gemma-2-2B | Workspace | 50.0% | 50.0% | 96.9% | 1 |
| Gemma-2-2B | Native tail | 59.9% | 50.0% | 96.9% | 5 |

No cell reached the predeclared 80% accuracy plus 10-percentage-point null advantage. All 192 completed free readouts reached native EOS. Mistral's high native-tail string diversity did not establish source identification. All 96 full-vocabulary neutral replays and both native hook-parity checks were exact.

The initial Gemma attempt stopped after 82 donor captures because one source occupied only seven native tokens. It scored no questions. Its failure and tensors are preserved. Both pole-zero photo phrasings were lengthened together, and actual native-tokenizer shape validation was moved before model construction. No score informed this repair. [Plan and amendment](PLAN.md) · [Initial failure](trials/gemma-layout-failure/terminal.json)

## Data and narrowed learning scope

The [public data](../../data/codec_preparation_20261102/manifest.json) contain 48 training groups in four construction families and 24 dev groups in two other families. Two opposed policies and two phrasings give 192 train and 96 dev descriptions. Another 24 groups in two other families were committed by SHA256 before fitting and stored outside the training checkout. Qualification outcomes have not been observed. Reference authorship is disclosed, and independent human reference/transition review is pending.

These are finite synthetic policies with explicit truth ledgers. The first codec teacher is an exact copy of the source description. Proposed case-expansion targets remain pending human usefulness review. The grouped public JSON contains those G target fields and is parsed, but the codec iterator returns only source-copy examples. G targets do not enter model inputs or the loss. Qualification content is outside that data and was not opened. Ridge fitting on G targets and operator adaptation were deferred under the codec-only scope. [Learning contract](LEARNING_CONTRACT.md)

The immutable fit receipt uses the legacy field `G_teacher_loaded: false` for absence of G supervision. It does not mean those public JSON fields were never parsed. Later fit receipts use `G_teacher_used_by_loss` to state this boundary precisely. The original receipt and source remain unchanged.

Gemma's native screen failed, so the declared fallback retained workspace. We fitted `z + (z A) B` at rank 16 with 73,728 trainable parameters and a frozen backbone. B began at zero. Exactly 64 AdamW updates accumulated eight examples each, using source-text-plus-native-EOS NLL, seed 47 and learning rate `3e-4`. All 96 dev descriptions were evaluated at five fixed checkpoints. Minimum dev NLL selected step 64, with identity initialization eligible. Every checkpoint is retained in [the learning record](learning/gemma-codec).

## Learning and functional outcomes

| Endpoint | Before | Selected step 64 |
| --- | ---: | ---: |
| Mean dev source NLL, nats/token | 4.309862 | 1.857181 |
| Functional dev group-macro accuracy | 49.9% | 50.0% |
| Distinct greedy D strings / 48 | 1 | 4 |

NLL averages tokens, including native turn-ending token 107, within each description, then averages 96 descriptions. Functional accuracy averages within each of 24 dev groups, then across groups. Groups have different numbers of context cases, so macro accuracy differs from pooled accuracy. Raw functional counts are 287/576 before, 288/576 after and 483/576 for literal text. Literal group-macro accuracy is 87.9%. Both label orders and all attempted readouts remain in the denominators.

All 96 pre/post free readouts reached EOS. The four post-fit strings described signal policies, while dev references concerned visitor mode or request/override conditions. No blind human support judgment has been collected. Lower teacher-forced loss and more strings did not demonstrate recovery of these individual policies.

Every update had a positive finite bridge gradient norm. Full backbone parameter hashes matched before and after fitting. At initialization and after loading the selected artifact, the active differentiable and inference paths gave exactly the same checked NLL. The artifact, capture geometry, data manifest and source are hash-bound. The encoder remains unqualified.

## Read-only replay and opposed donors

After the fit, a fixed [read-only audit](AUDIT_PLAN.md) reproduced all 480 per-description NLLs at five checkpoints through the inference host and retained each value. Original source-state hashes, artifact digests and host identity matched. The largest discrepancy between a replayed and recorded mean was `3.48e-8` nats. This replay used zero optimizer updates.

At the initial and final checkpoints, we also kept each target description fixed and replaced its donor with the opposite policy in the same group/phrasing, or zero rows before the bridge. These post-hoc development controls did not alter selection.

| Checkpoint | Matched donor NLL | Opposite donor NLL | Zero donor NLL | Opposite minus matched |
| --- | ---: | ---: | ---: | ---: |
| Initial | 4.309862 | 4.309670 | 4.327749 | -0.000192 |
| Step 64 | 1.857181 | 1.858054 | 4.327749 | +0.000873 |

The opposed donor preserved almost all aggregate NLL reduction. At step 64 the matched donor had lower loss in 49/96 descriptions, with no exact ties. The zero donor did not receive the reduction. The observed improvement uses nonzero carrier states but has very little aggregate sensitivity to the opposed policy in this test. It does not show that all content is absent from the tensors or identify a unique mechanism.

## Where this leaves the anchor

- Package 1 is closed by its bounded negative result. No extra layers, row counts or lexical positions were searched.
- Data and implementation prerequisites for a **codec-only** fit were made concrete, and the first fit is complete. Broader G teaching and human semantic review remain pending.
- The codec has not met a content-preservation screen. G adaptation, the independent qualification challenge and long latent pilots remain unstarted. The 24 qualification groups remain sealed.
- A next comparison can test an explicitly content-sensitive training signal against this NLL-only objective at a matched parameter/update budget. It needs a separate fixed contract. This result does not identify a universally inadequate loss, capacity or learning budget.

## Sources and reproduction

The layout failure used `54a9d1649a63aace227df15bc7d5612a435ab9af`, completed carrier cells used `1c07475`, and codec fitting used `b6ba167`. Full source commits/manifests are in the records. All jobs used the earlier pinned snapshots, one RTX 5090, eager BF16 and native EOS. No pretrained checkpoint weights were added. The learned bridge arrays are included.

From this directory, check file integrity and rebuild the summary without loading a model.

~~~bash
shasum -a 256 -c SHA256SUMS
python3 summarize.py | diff - summary.json
~~~

[Verification](verification.json) · [Carrier command](../../scripts/probe_carriers.py) · [Learning command](../../scripts/fit_codec.py) · [Independent replay](audit_codec.py)
