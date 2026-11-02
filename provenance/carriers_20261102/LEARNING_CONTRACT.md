# First learning contract

This narrows package 3 to the codec-only route permitted by the [qualification plan](../../docs/SEMANTIC_GATE_PLAN.md). G targets have formal ledgers but human usefulness review is pending, so ridge fitting on G targets and operator learning are deferred. The first learned object is a description encoder bridge, not the semantic judge.

## Material

The public preparation data contain 48 training groups from four construction families and 24 development groups from two other families, with opposed policies and two source phrasings per pole. They yield 192 train and 96 dev source variants. A further 24 groups from two other families are committed by SHA256 in the public manifest and stored outside the training checkout. The qualification content is not loaded. Its reference author is the current study author, and independent human reference/transition review is pending. This is synthetic explicit-policy preparation, not held-out open-ended reasoning evidence.

For this fit the teacher is an exact copy of the current source description. The target does not rely on a model-generated semantic judgment. Finite truth ledgers and proposed case-expansion texts are retained for later review; the codec loader returns only source/target strings and identifiers. Its interface rejects a qualification split and checks data file hashes. Data separation is a documented loading contract, not an operating-system access-control claim.

The public group parser reads the complete group record, including its proposed G target fields. Those fields are not passed to the model or optimizer loss. The original receipt's `G_teacher_loaded` flag denotes this lack of supervision, not absence of JSON parsing; later receipts use `G_teacher_used_by_loss`. Qualification content is in a separate, unopened file.

## Candidate and selection

Use Gemma-2-2B-IT, the existing midpoint, eight rows and a 256-position description-only donor. If a Gemma carrier passes package 1, choose the passing route with the highest group accuracy, then the largest margin over null, with workspace breaking exact ties. If neither passes, retain workspace as the existing interface and declare a negative native-interface screen. Native self-replay must be valid before fitting.

Train `z + (z A) B` with rank 16, zero-initialized B, seed 47 and 73,728 trainable parameters. Source capture and the backbone are frozen. D is the existing native read prompt. Teacher forcing targets only the source text and native turn-ending token 107. The loss is mean token NLL, without a semantic-task loss. The differentiable path is separate from `SlotHost.forward` and `State`.

Run exactly 64 optimizer updates with eight accumulated examples per update. AdamW uses learning rate `3e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, no weight decay and gradient clipping at 1. Shuffle source examples with seed 47. Evaluate all 96 dev descriptions at steps 0, 16, 32, 48 and 64. Choose the lowest mean dev NLL, breaking ties toward the earliest step. The initial identity adapter is an eligible candidate. Preserve every checkpoint and training loss. One job has 1,800 seconds, 20,000 forwards and 20 million processed positions, with no automatic retry or extension.

Only after selection, ask functional dev questions from the independent truth ledgers, in both option orders, for raw slots, selected bridged slots and literal source text. Read each pole's first phrasing with D before and after selection, using native EOS and recorded censoring. Group the dependent observations by meaning group. These outcomes diagnose the selected codec; they do not select another checkpoint or grant a semantic gate.

Hash every backbone parameter before and after fitting. Verify that only bridge parameters received gradients. Bind the serialized adapter, exact base-host identity, capture geometry, source and data manifests. Changing any learned artifact invalidates the old host binding. No latent trajectory pilot, G learning or qualification challenge is part of this job.

At identity initialization and again after loading the selected serialized artifact, compare the differentiable teacher-forcing NLL with the inference host on the first development source at identical shape, positions, dtype and target EOS. Require an absolute difference no larger than `1e-5` and retain both values. A failure stops the fit or invalidates its evaluation.
