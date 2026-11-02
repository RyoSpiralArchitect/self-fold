# Conditions for the Gemma encoder artifacts

This statement covers `encoder_step_000.npz`, `encoder_step_016.npz`, `encoder_step_032.npz`, `encoder_step_048.npz` and `encoder_step_064.npz` in this directory. SELF-FOLD distributes these Gemma-derived model artifacts subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms), including the use restrictions in Section 3.2 and the incorporated [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). Use, reproduction, modification and distribution of these files must comply with those conditions.

The arrays are newly created low-rank encoder parameters for a frozen Gemma-2-2B-IT host. Step 0 is identity initialization. Later steps record SELF-FOLD's source-copy training, with the backbone unchanged. The artifact metadata identifies their exact base host, data, source and update count. [Modification and distribution notice](NOTICE.txt)

Copies of the agreement and incorporated policy accompany the files. Their source URLs, retrieval date and file digests are in [the copy provenance](terms-provenance.json).

- [Gemma Terms of Use, copied 2026-11-02](GEMMA_TERMS.html)
- [Gemma Prohibited Use Policy, copied 2026-11-02](GEMMA_PROHIBITED_USE_POLICY.html)

The project's original code, documentation and synthetic fixtures use the [Apache License 2.0](../../../../LICENSE). These five model artifacts have the Gemma conditions stated above. The agreement copies retain the source-page attribution and content-license notice.
