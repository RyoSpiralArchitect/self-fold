# Functional continuity under semantic recurrence

**Development protocol and status, 2026-11-03.** This document integrates the experiment around the original SELF-FOLD question. Execution is sealed by phase. The original recipient contracts below retain their fixed architectures, budgets and decision rules. All four tested configurations are now closed with negative development outcomes. G learning and the subsequent language intervention require separate phase seals. The intended two-stage contribution remains prospective, and no semantic qualification pass has been issued.

## Dated development status

On 2026-11-03, four configurations completed their fixed 256-update endpoints. The first two trained only the native recipient with full-vocabulary cross entropy, using raw or standardized frozen E states. A separately sealed conditional-label fit used raw frozen E. The last configuration jointly trained a source-only E adapter and a task-only recipient under its own contract. The zero-update numerical audit remained diagnostic. [Contracts, exact counts and raw evidence](../provenance/continuity_20261103/RESULTS.md)

| Configuration | Stage2 native correct / 4,608 | Stage2 compound correct / 1,152 |
| --- | ---: | ---: |
| Raw frozen E, full-vocabulary recipient | 2,294 (49.78%) | 52 (4.51%) |
| Standardized frozen E, full-vocabulary recipient | 2,308 (50.09%) | 17 (1.48%) |
| Raw frozen E, conditional-label recipient | 2,304 (50.00%) | 0 (0.00%) |
| Joint source E and task recipient | 2,336 (50.69%) | 57 (4.95%) |

Every row missed the same recipient screen of 90% native accuracy, a ten-point advantage over the strongest fixed null and 75% compound correctness. All three representation stages in these tests were encoded independently from the source text or its formally equivalent teacher text. The stage2 text is canonical, so its two source-order tags have identical bytes. Its compound count crosses two genuine policy polarities with duplicated order inputs. Source and explicit-case panels retain their own wording/order results.

No learned G was trained or evaluated. These outcomes therefore close the tested input/readout configurations and leave actual two-stage state reconstruction, operation omissions, intermediate-state exchange and language insertion pending. The earlier sealed qualification groups retain their original commitment. New two-stage confirmation content and independent human review remain pending.

## The question

When the meaning of a current self-description is interpreted again and folded into the next current state, can the resulting distinction support another meaning update and a judgment in a new situation without consulting the original description or a transcript? How does verbalizing that intermediate state alter its later use?

The first target is a finite, auditable reconstruction of conditional policies. A supported reconstruction can make an existing condition or exception explicit without inventing an additional fact. The later judgment retains the same truth criterion across update and no-update controls. A separate open-ended study can examine defensible reinterpretations with several acceptable successors. That study needs independent human support judgments.

The intended positive contribution is a constructed route in which a source-dependent distinction is reconstructed, passed through a second update, and used in a later judgment, with interventions identifying the role of the intermediate state. This is a prospective claim. Each result will state the subset actually established.

## Why the existing evidence leads here

Two fresh Gemma-2-2B-IT encoder fits used the same initial parameters, sample schedule, 512 description forwards and 44,804 training tokens per arm. Copy training and an added opposed-token objective obtained 49.35% and 50.00% functional group accuracy. Each obtained zero correct switches in 288 comparisons requiring correct answers for both opposed policies. A separate source-minus-neutral readout identified the first-mentioned tray in 93/96 clause-reordered descriptions, while the prescribed action was correct under both orders for 6/48 policies. These results motivate direct training on condition–action use and explicit clause-order controls. [Frozen development evidence](../provenance/policy_binding_20261103/RESULTS.md)

The neutral reference in that readout is constructed from the original source text and requires another encoding. It supplies an initialization diagnostic. The next route must specify the information available to every later update when that source text has been removed.

## Interfaces and information available at each operation

Let `Z0 = E(s)`, `Z1 = G1(Z0)` and `Z2 = G2(Z1)`. The native recipient `A(Z, q)` receives a clone of the tested state and a question introduced after the state has been constructed. A task-trained adapter is active only in this recipient call unless its broader scope is explicitly part of the sealed architecture.

| Operation | Available input | Output and binding |
| --- | --- | --- |
| Initial E | One current description | Question-independent slots, capture geometry and source-state hash |
| G1 and G2 | Current slots and a content-independent operation | New slots in the frozen interface, with operation position recorded |
| D | A clone of the current slots | One current description, generation receipt and censoring flag |
| E after D | The current D output | Slots derived solely from that output |
| A | A cloned state and one later question | Native LLM answer scores under both label mappings |

The experimenter's archive contains the original source, references and history. Those files supply evaluation and audit records. The recurrent input is the current state and the specified operation. Gold next descriptions, expected actions, evaluated question text and previous verbalizations enter training or scoring only through their declared interfaces. All trainable modules, hook locations, prompts, checkpoint snapshots and data manifests are part of the binding.

E is the frozen Gemma-2-2B-IT raw workspace interface, with eight slots at pre-cut 13 captured from a fixed 256-token source-only forward. The donor is computed without a task adapter. Training a recipient on its own training questions is permitted. Every evaluation donor is captured before its evaluation questions and reused unchanged across those questions. A small external probe can diagnose accessibility, while the functional endpoint uses the intended LLM output path.

The recipient adds rank-16 LoRA modules to the query and value projections in the last four decoder layers, with LoRA scale one. These modules are active only during task forwards. During recipient fitting, E, G and D use the frozen base. The objective is full-vocabulary native A/B cross entropy. Both displayed label mappings occur in training and evaluation.

Candidate 1 receives raw slots. If it misses the development recipient screen, candidate 2 uses the same LoRA architecture and budget with fixed coordinate standardization fitted only on train slots before injection. Its exact centering, scaling and numerical floor belong to the recipient implementation seal. There are at most two candidates. A passing first candidate ends the fallback search. This conditional search does not supply a randomized comparison between the two input geometries.

Each candidate uses seed 47 and exactly 256 AdamW updates at learning rate 0.001, zero weight decay and gradient-norm clip one. The effective batch is 32 examples, accumulated as four microbatches of eight. The endpoint is the final update. No checkpoint is selected by development scores.

## Two stages and the identity controls

A finite semantic specification `Fc` defines permitted reconstructions independently of D. For a held-out source and later question, the intended action follows the truth conditions of `F2(F1(s))`. If both transformations preserve the same policy, identity can also satisfy that action equality. Final correctness and an opposed-state swap therefore need an additional test of what each requested reconstruction contributed.

The minimum comparison includes these paths under the same downstream answer budget.

| Path | Role |
| --- | --- |
| `Z0 -> G1 -> G2 -> A` | Full two-stage route |
| `Z0 -> G2 -> A` | Omit the first reconstruction |
| `Z0 -> G1 -> A` | Omit the second reconstruction |
| `Z0 -> A` | Initial-state identity baseline |
| `Z0 -> sham1 -> sham2 -> A` | Predeclared generic computation or perturbation control |

Each stage also receives an independently specified criterion for the supported distinction it should expose. An operator-target string match alone cannot establish semantic support. The formal domain can use a separately executable interpretation and contrastive cases. Free reinterpretation requires independent transition review. Numeric state change, stage labels and fluency do not supply the criterion.

On cases designated as change-required before fitting, omission tests measure the change in the specified distinction and its use. On already explicit cases, preserved correct meaning is a justified no-change outcome. Every stage need not increase accuracy. If the full route preserves policy use while an omitted stage yields the same tested distinctions and utility, preservation can be reported and that stage's incremental contribution remains unresolved.

Depth two is the primary composition endpoint. Depths zero and one diagnose the first failing connection. Depth four is a fixed secondary endpoint if admitted by the frozen execution contract. No outcome selects a different primary depth.

## Intermediate-state exchange

At the first stage boundary, exchange only `Z1` while holding the second operation, recipient, question, label mapping and readout budget fixed.

* An opposed-policy donor should induce the later answer specified by that donor's reconstructed policy.
* A same-meaning donor with a different wording or clause order should preserve the specified answer.
* The identical-state replay checks the mechanics.
* An initial-state replacement tests whether the first reconstruction is needed at this boundary.
* Fixed absent, unrelated and geometry-matched controls bound dependence on the intended source distinction.

The exchange occurs at the same operation position. The continuation starts with the second operation. Reinitializing an engine schedule at operation zero would test a different composition and is an invalid continuation for this contrast. Source-state hashes, donor IDs, operation identities and all attempted exchanges are retained.

## A single language insertion at a shared state

The primary language contrast branches at the same saved `Z1`.

`L1 = G2(Z1)`

`B1 = G2(E(D(Z1)))`

Both branches share G2 and the final recipient. D observes a clone on the L trajectory when a measurement is needed. D followed by E is a causal operation on the B branch. Native EOS is the default, and the full generation, stopping reason, token count and nonempty censored output are retained. Censoring alone does not assign semantic failure.

The single insertion estimates the effect of this E/D roundtrip on the subsequent update and judgment. Report D/E compute, generation length, slot positions and state norms. Fixed continuous bottleneck or perturbation controls assess damage that could arise from this particular codec. The all-steps B trajectory is a separate secondary condition when specified.

L exceeding B is one possible outcome. B exceeding L can show useful organization by the roundtrip. Practical equivalence is declared only when the full 95% paired interval for the registered accuracy difference lies inside minus five to plus five percentage points. An interval extending outside those bounds leaves equivalence unresolved. The effect direction is determined by the data.

## Material and independent units

The current authoring manifest has 96 training and 24 development meaning groups, with 1,152 and 288 source variants respectively, each represented at three stages. Shared Boolean policy classes have explicit formal truth conditions. The training data cross policy polarity, clause order, action names, question wording and answer labels. Matched self, other-agent and impersonal frames are bound in the data manifest. Condition–action assignments stay balanced across lexical order so a first-mention shortcut cannot solve the panel.

A meaning group owns all its paraphrases, opposite policy, frames and evaluation situations. Splitting occurs by whole group and declared construction family. The same source state is scored on multiple later situations, with the two label orders retained. Their repeated answers are dependent observations within that group.

The previous 24-group sealed qualification commitment remains associated with its earlier contract. The two-stage continuity study needs a new independently sealed panel covering composition, operator omission and intermediate-state exchange. The authoring process, reference checks and human-review status are disclosed. The new panel is kept outside training and development paths until candidate and scorer bindings are frozen.

## Compound endpoint and decisions

The recipient screen is evaluated on stage-2 source descriptions through the native LLM output path. It requires at least 90% group-macro label accuracy, at least ten percentage points above the strongest fixed null, and at least 75% group-macro joint correctness across both policy poles and both clause orders for a world. The executable scorer fixes the additional frame, question and label-order nesting before scoring. Literal-source performance is a benchmark. A group on which literal text fails remains in all registered denominators.

The future primary continuity target requires at least 70% joint correctness after actual G1 and actual G2 across both policy poles and both clause orders, together with at least 90% actual post-G2 accuracy. Each compound unit holds group, world, attribution frame, question wording and displayed answer mapping fixed. The G phase seal binds all further scorer and uncertainty details before execution. Each required source-dependent judgment and formal transition-support check is retained. Opposed and same-meaning exchanges have their own specified joint units. Report the component outcomes as well as the joint count.

### Prospective G endpoint clarification, 2026-11-03

Before any G training or G evaluation, the planned primary conjunction was clarified to require correctness after G1 and G2. The earlier proposal also required the initial Z0 answer to be correct. That requirement could reject the intended outcome in which reconstruction makes an initially inaccessible distinction usable. Z0 accuracy, the paired gain over Z0 and the original strict Z0/G1/G2 conjunction are retained as secondary diagnostics. The original recipient screen and all closed recipient results keep their definitions.

Every original group remains in the G evaluation regardless of its Z0 answer. Report change-required and justified-no-change strata. A pass on the post-G1/post-G2 conjunction establishes the tested continued use after updating. Attribution of a useful distinction to G additionally requires the independently specified intermediate support checks, operation omissions and intermediate-state swaps. Any claim of improved task utility uses the preregistered comparison on the same questions and truth conditions. G architecture and its training budget remain unsealed.

G omission controls use independently specified intermediate format and truth-support checks. They assess which requested reconstructions are realized and used. Each G need not improve accuracy. If an omission preserves the same evaluated distinction and utility, the corresponding incremental contribution remains unresolved.

The old carrier screen of 80% accuracy and a ten-point null advantage applied to its earlier unit. The numerical targets above belong to this new experiment. The scorer, uncertainty method and denominators must be bound at their phase seals before evaluation. Score ties, missing answers, failed captures and censored generations under those rules. Preserve all attempted groups and also describe completed-output coverage.

| Stage | Positive closure | Bounded negative closure |
| --- | --- | --- |
| Recipient | Stage-2 native-label accuracy reaches 90%, the strongest fixed null is exceeded by ten points, and joint pole/order correctness reaches 75% | At 256 updates retain candidate 1. If it fails, run only the declared standardized-input candidate under the same cap. If both fail, close recipient learning for this cycle |
| Two-stage reconstruction | Joint correctness after actual G1 and G2 reaches 70% and post-G2 accuracy reaches 90%, with formal support, omissions and swaps supporting the stated contribution | G learning begins only after the recipient screen and its own architecture/budget seal. At that fixed cap record failed support, failed reuse or unresolved stage contribution |
| Language insertion | The admitted route supports the frozen L/B comparison with usable codec controls and declared uncertainty | Record codec failure or an inconclusive L/B contrast, including the first failing connection |
| Independent challenge | Candidate, data and scorer bindings are frozen and every required scientific check has been reviewed | Retain failed, pending or invalid checks and close this challenge. Further selection uses a fresh holdout |

Parity failure, leaked evaluation input, state mutation, changed artifact, broken reference truth or an invalid operation position stops the affected comparison as invalid. Missing independent review is pending. Valid sub-threshold evidence is a scientific failure of the tested route. Budget exhaustion closes a candidate at its registered endpoint.

## Phase-specific execution seals

The recipient source commit, exact manifest hashes and artifact identity complete its execution seal. Its phase is fixed as follows. A later G contract is committed only after a recipient passes its development screen. G architecture, training budget and target representation remain deferred at this point. A closed-world formal development result can proceed before the independent human review needed for semantic qualification.

| Field | Recipient phase and later-phase status |
| --- | --- |
| E route and geometry | Frozen Gemma-2-2B-IT raw workspace, K=8, pre-cut 13, 256-token source-only capture |
| Recipient modules | Query/value rank-16 LoRA in the last four decoder layers, scale one, active only on task forwards |
| Objective and optimizer | Full-vocabulary native A/B CE, AdamW, learning rate 0.001, zero weight decay, global gradient clip one |
| Candidate cap | Raw slots first. One standardized-input fallback only if the raw candidate misses the recipient screen. Standardization moments use train data only and its formula is part of the implementation seal |
| Seed and update budget | Seed 47, 256 updates per candidate, effective batch 32 in four microbatches of eight |
| Endpoint selection | Fixed final update, no development checkpoint selection |
| Current public data | 96 train / 24 dev groups, 1,152 / 288 source variants and three stages per variant. Exact bytes and hashes are in the committed data manifest and run request |
| Recipient screen | Stage-2 accuracy at least 90%, ten-point advantage over strongest fixed null, joint both-poles/both-orders correctness at least 75% per world, with group-macro aggregation |
| G1/G2 architecture and budgets | Deferred to a separate G learning seal after a recipient passes |
| Primary continuity target | At least 70% joint actual post-G1/post-G2 correctness across poles and clause orders at fixed world, frame, wording and answer mapping, plus at least 90% post-G2 accuracy. Z0 and the original all-three-stage conjunction are secondary diagnostics. Full scorer details are bound at the G seal |
| Language comparison | One insertion at Z1, common G2 and answer budget. Equivalence margin five points, full 95% paired interval inside the margin |
| Independent confirmation and human review | New two-step confirmation panel and independent reviewers pending. The earlier sealed 24-group commitment retains its own contract |

The later phase seals specify the G stopping rule, compound scorer nesting, exchange denominators, resampling unit, codec controls and applicable wall/forward/token limits. A recipient pass admits only the next declared development phase. It does not issue a semantic qualification approval.

## Contribution in relation to nearby work

The Quine Test studies whether model-written self-descriptions reproduce preference profiles in other models or fresh instances, and includes successful transfer under explicitly assigned policies. SELF-FOLD's proposed comparison follows a current description through successive supported reconstructions and tests whether each resulting state remains usable by later computation. [The Quine Test](https://arxiv.org/abs/2610.38726)

Natural Language Autoencoders already learn an activation-to-language-to-activation route and use edited explanations to construct interventions affecting subsequent output. The additional target here is the reuse of an intermediate reconstructed self-description by a later meaning update and the effect of a common-state language insertion on that reuse. [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html)

A successful constructed route would establish the specified continuity under its learned interface and task family. New useful distinctions, performance gains on the same unseen questions, self-reference-specific effects and cross-model replication are additional claims requiring their corresponding evidence. The initial experiment starts on Gemma. A frozen successful procedure can then be examined on Mistral under a new binding.


### Recipient implementation seal

Both candidate endpoints are step 256, without intermediate dev selection. Candidate raw injects the stored native workspace unchanged. Only if it misses its screen, candidate standardized initializes a fresh identical adapter and uses the same schedule.

For standardized inputs, fit coordinate means `mu[K,d]` and population standard deviations `sigma[K,d]` across all **train** source, explicit-case and reintegrated teacher captures. Let `tau[K,1]` be each row's mean coordinate deviation. Transform each task input as `mu + tau * (z-mu) / max(sigma, 0.01*tau)`, using FP32 arithmetic and BF16 at injection. The same transform applies to matched, opposed, zero, random and unrelated states. Empty input supplies native placeholder slots without an injection. The transform belongs to the recipient and supplies no original-text reference. Save these train-only arrays and their digest with every model binding.

The fixed nulls are empty input, zero slots, a donor-independent seeded random state at the train mean row norm, and the first training policy's plus/self/order-zero donor. The unrelated donor is selected separately for each representation stage. A source swap uses the opposite policy in the same group, frame, stage and clause order. A fresh zero-initialized recipient gives a literal-text base control before fitting.

Every candidate is scored on all three teacher-encoded representation stages. The recipient entry screen uses stage2, whose texts are independently authored equivalent rules encoded directly by E. This is an interface admission test. Actual learned `G2(G1(Z0))` receives a separate source seal and evidence. The compound recipient unit holds group, frame, world, question wording and displayed label order fixed and requires all four combinations of policy polarity and source clause order to answer correctly through native greedy A/B output. Average within group and then across 24 groups. A native output outside A/B is unsuccessful. Forced-choice ties are unsuccessful in the forced-choice score, while native deterministic argmax is retained as its own outcome. Publish forced-choice accuracy and native output coverage alongside this endpoint.

There are 4,608 logical decisions per stage per route, with six routes per candidate. Shared exact question/state computations may be reused while keeping every logical decision in the record. The fixed formal source/teacher corpus contains 96 training groups and 24 development groups, with 1,152 and 288 variants at each of three stages. The source and targets share Boolean truth functions across splits. Conditional source grammars, flag/action bindings and sites are held out. Canonical teacher renderers are shared, and explicit-case no-op sources also share a canonical grammar across splits. New confirmation content and independent human transition review remain pending.
