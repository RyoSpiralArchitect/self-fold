# Licensing and provenance

The project's original code, documentation and synthetic development data are licensed under the Apache License, Version 2.0 (`SPDX-License-Identifier: Apache-2.0`). See the [license text](../LICENSE). External model weights and tokenizers retain their upstream licenses and access conditions. The repository ships no pretrained checkpoints or external tokenizers.

The five `encoder_step_*.npz` arrays in `provenance/carriers_20261102/learning/gemma-codec/` are distributed as Gemma-derived model artifacts under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and its incorporated [Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). Their directory contains the required notice, a copy of the agreement and policy, modification details, and an [artifact-specific license statement](../provenance/carriers_20261102/learning/gemma-codec/ADAPTER_LICENSE.md). The Apache license for the original software and synthetic data remains unchanged.

SELF-FOLD is an independent implementation. LTWS documents informed the architecture and recording of experiments. External model implementations retain their upstream notices.

The synthetic descriptions and questions were written for these experiments. Bibliographic metadata and source links accompany the paper.

The paired policy-binding study also distributes two Gemma-derived encoder arrays, `copy.npz` and `opposition.npz`. Their directory includes the terms, policy, modification notice and training provenance. [Conditions for these two arrays](../provenance/policy_binding_20261103/run/ADAPTER_LICENSE.md)

The continuity study's task adapters, source adapter and activation-statistic arrays have artifact-specific Gemma notices and accompanying terms in each experiment folder. [Original recipients](../provenance/continuity_20261103/recipient/ADAPTER_LICENSE.md) · [Conditional-label follow-up](../provenance/continuity_20261103/binary/ADAPTER_LICENSE.md) · [Joint source/recipient](../provenance/continuity_20261103/joint/ADAPTER_LICENSE.md)

The paired small-set control also provides its two recipient adapters with the same source-specific Gemma terms and notice. [Trainability-control artifact conditions](../provenance/continuity_20261103/trainability/ADAPTER_LICENSE.md)

The two recipient arrays in the policy-form study carry their Gemma conditions, modification notice and copied terms alongside the fit records. [Policy-form artifact conditions](../provenance/policy_form_20261103/run/ADAPTER_LICENSE.md)

The bounded readout cycle provides four learned arrays with source-specific Gemma conditions and copied terms. [Decoder, lexical transformation and mixed-order recipient conditions](../provenance/slot_readout_20261103/run/ADAPTER_LICENSE.md)

The E-to-lexical source maps contain training-only activation-derived ridge buffers and learned residual factors with their applicable Gemma conditions. [Source-map artifact conditions](../provenance/source_lexical_20261103/run/ADAPTER_LICENSE.md)

The stopped initialization and the four within-topic policy mixture artifacts contain Gemma-derived teacher rows and statistics. [Within-topic artifact conditions](../provenance/within_topic_20261103/ARTIFACT_LICENSE.md)

The calibrated teacher renderer and source-direction arrays use the same Gemma terms. [Teacher and source readout artifact conditions](../provenance/teacher_polarity_20261103/ARTIFACT_LICENSE.md)

The role-exchange readouts contain activation-derived directions, standardization and the prior topic router. [Role-exchange artifact conditions](../provenance/role_exchange_20261103/ARTIFACT_LICENSE.md)

The action-swap direction arrays and their paraphrase-penalized counterpart have Gemma-derived artifact conditions. [Action and operation artifact conditions](../provenance/action_operation_20261103/ARTIFACT_LICENSE.md)

The scoped native-recipient and source adapters from the conditional curriculum retain their Gemma-derived terms. [Conditional curriculum artifact conditions](../provenance/conditional_curriculum_20261103/ARTIFACT_LICENSE.md)

The margin continuation provides its fixed native-recipient and any conditionally admitted source adapters under the same Gemma-derived conditions. [Margin continuation artifact conditions](../provenance/conditional_margin_20261103/ARTIFACT_LICENSE.md)

The two final-action and component-trained task adapters have [Gemma-specific artifact conditions](../provenance/two_flags_20261104/ARTIFACT_LICENSE.md).

The study of updates after discarding the source includes the fitted compiler, update, native reader, and numeric state and score records. [Study-specific artifact conditions](../provenance/state_only_20261104/ARTIFACT_LICENSE.md)

The nine-parameter role controllers, captured rows and native decisions have [study-specific distribution conditions](../provenance/role_memory_20261104/ARTIFACT_LICENSE.md).

The stage-grammar study's three continued task readers, state-to-state maps, rule decoder and numerical records have [study-specific artifact conditions](../provenance/stage_grammar_20261104/ARTIFACT_LICENSE.md).

The version-one preprint PDF uses Apache-2.0. The associated experimental arrays and native scores carry the [study-specific Gemma terms and notice](../provenance/preprint_v1_20261104/ARTIFACT_LICENSE.md).

The post-v1 supported-boundary development study distributes its learned arrays, captures, native scores and generations with [artifact-specific Gemma terms and notice](../provenance/supported_g_20261104/ARTIFACT_LICENSE.md).
