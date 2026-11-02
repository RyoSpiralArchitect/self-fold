# One conditional native-logit contrast fit

The original raw and standardized full-vocabulary recipient fits are closed. This is a separately bounded development experiment using the exposed material. It is admitted only if the frozen numerical audit finds that at least 10% of the trained raw recipient's audited rows have distinct pre-cap A/B logits but exactly equal native capped A/B values. Otherwise this phase stays unexecuted.

The working hypothesis is that the full-vocabulary objective learned to raise both labels together while BF16 softcapping erased their differences. A/B conditional cross entropy removes the reward for increasing their combined mass. This is a hypothesis about a tested optimization route.

Use exactly one fresh raw rank16 q/v recipient, same frozen Gemma E, source data, seed47, initialization, sequence of group/frame/stage/wording batches, AdamW learning rate0.001, zero weight decay, clipping1,256 updates and32-question effective batch as the original raw candidate. The loss is cross entropy on the two A/B logits returned by the unchanged native Gemma forward, including its original BF16 final softcap. Gold labels select index0 or1 in these two columns. No pre-cap ranking or analytic precision diagnostic supplies the native prediction. Existing trained weights are not resumed.

Replay all source-state hashes from the completed fit and require identical initial adapter parameters, ordered task schedule, training forward count and processed-token count before evaluating the endpoint. Reuse the previous literal base result by its immutable terminal digest. Record the actual new elapsed time and compute.

The final endpoint remains full-vocabulary native greedy output, with the same stage2 90% accuracy,10-point advantage over fixed nulls and75% compound screen. Conditional-label success alone is insufficient. Native outputs outside A/B and exact forced-score ties stay in their respective denominators. Source/case stage and held-out frame/construction outcomes remain visible. There is no dev checkpoint selection and no second candidate.

If this fit fails, close the recipient architecture under these three tested configurations. A new carrier or broader model requires a separate design. If it passes, freeze the receiver and separately specify and test G reconstruction, stage omission, intermediate swaps and language insertion. This fit alone does not establish the intended two-step contribution. The original24 qualification groups and independent transition review remain pending.
