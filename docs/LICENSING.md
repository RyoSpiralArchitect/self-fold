# Licensing and provenance

The project's original code, documentation and synthetic development data are licensed under the Apache License, Version 2.0 (`SPDX-License-Identifier: Apache-2.0`). See the [license text](../LICENSE). External model weights and tokenizers retain their upstream licenses and access conditions. The repository ships no pretrained checkpoints or external tokenizers.

The five `encoder_step_*.npz` arrays in `provenance/carriers_20261102/learning/gemma-codec/` are distributed as Gemma-derived model artifacts under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and its incorporated [Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). Their directory contains the required notice, a copy of the agreement and policy, modification details, and an [artifact-specific license statement](../provenance/carriers_20261102/learning/gemma-codec/ADAPTER_LICENSE.md). The Apache license for the original software and synthetic data remains unchanged.

SELF-FOLD is an independent implementation. LTWS documents informed the architecture and recording of experiments. External model implementations retain their upstream notices.

The synthetic descriptions and questions were written for these experiments. Bibliographic metadata and source links accompany the paper.

The paired policy-binding study also distributes two Gemma-derived encoder arrays, `copy.npz` and `opposition.npz`. Their directory includes the terms, policy, modification notice and training provenance. [Conditions for these two arrays](../provenance/policy_binding_20261103/run/ADAPTER_LICENSE.md)

The continuity study's task adapters, source adapter and activation-statistic arrays have artifact-specific Gemma notices and accompanying terms in each experiment folder. [Original recipients](../provenance/continuity_20261103/recipient/ADAPTER_LICENSE.md) · [Conditional-label follow-up](../provenance/continuity_20261103/binary/ADAPTER_LICENSE.md) · [Joint source/recipient](../provenance/continuity_20261103/joint/ADAPTER_LICENSE.md)
