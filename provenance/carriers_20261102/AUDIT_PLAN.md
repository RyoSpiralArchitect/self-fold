# Fixed read-only codec replay

This analysis was specified after the 64-update fit completed and its functional dev accuracy remained near 50%. It is a post-hoc development diagnostic, with no model selection, training or qualification access.

Recompute all 480 development-example NLLs at the five frozen checkpoints using the inference host rather than the differentiable training function. Require the same source state hashes, exact fitted-host identity, matching artifact digests and a mean-NLL discrepancy no larger than `1e-5` at each checkpoint. Retain every per-example result.

At the initial and final checkpoints only, additionally replace each source state with the opposed pole of the same group and phrasing, or zero rows before the bridge. Keep the target source description unchanged. Evaluate all 96 development sources under these three controls. Positive opposite-minus-matched NLL means the correct donor better predicts its own source text. Report the paired gap, group means and sign counts. This is a token-likelihood diagnostic and does not grant a semantic gate or infer unique information storage.

The replay has one pass, 600 seconds, no generated tokens and no optimizer updates. Any parity or digest failure invalidates that replay. Source and audit-script hashes accompany its receipts. No result changes the selected checkpoint or the first fit's budget.
