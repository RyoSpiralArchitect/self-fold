# Can a role-paired writer keep the four rows and reverse its next edit?

Continue the exact Gemma writer from the [previous study](../structured_report_20261122/RESULTS.md). A frozen first edit produces a four-row table and one role bit. The writer sees the resulting rows and the role of off/off. It writes one English report before seeing any downstream question. Only the report reaches the fixed parser. The same frozen second edit and native Gemma recipient judge the two histories. Direct continuation and the exact report anchor the measurement.

## What is held out

Fit on all eight canonical action menus, with one or three of four coordinates assigned to the second action. The primary test contains the six previously unused balanced assignments for each of the four additional menus (routes, temperature, clock, music). Both action names occur during fitting. No primary physical table appeared in the previous writer fit, its evaluation, or the two new fits. Keep both role donors and all four cyclic row orders together when splitting. The four earlier menus with balanced assignments form a separate historical panel. Each panel contains 24 physical tables, 48 role-bearing states, 192 reports per writer and 96 twin presentations. The four display rotations share an underlying assignment.

## Training contrast

Both continuations restore the identical parent adapter, use the same batch of four twin pairs at every step, and have the same supervision, optimizer, learning rate, compute schedule and seed. Inputs in each twin have identical rows and opposite role markers. Training targets share four lines and switch the default on the fifth. In addition to the original token and five lexical-choice cross entropies, use softplus(2 minus the signed default-choice margin). The individual objective averages the penalties for both donors. The paired objective uses the larger penalty within each pair. Only this aggregation differs. With teacher forcing both default predictions see the same correct preceding rows. Test uses independent greedy free generation. Retain EOS or the text at the 160-token cap.

Four shuffled passes through 512 prompts, eight prompts per step, 256 AdamW steps per fit, learning rate 0.00015, no decay, unit gradient clipping. Train the original 245,760 adapter parameters. Close both endpoints before generating evaluation reports. Compare parent, individual continuation and paired continuation at one endpoint with one seed. Preserve all missing fields and failed cases in denominators.

## Decisions fixed before training

The primary count requires both donors on the same presentation to report all four correct rows, name the correct opposite defaults, and produce strictly signed correct full-vocabulary choices after each of 14 next edits across four worlds and both A/B orders. Also count each component and paired gains/losses. Score the 256 native row/menu/order combinations before fitting; the training loss has no access to them.

Require at least 92/96 joint successes for both L and the oracle. Row preservation requires at least 164/192 exact four-row reports with at most nine fewer than the parent. Transfer requires these anchors and preservation plus at least ten additional joint pairs over the parent and a gain in three menus. A paired-specific advantage additionally requires at least five more joint pairs than the individual objective and a gain in three menus. Apply the primary decision on the fresh panel. Report the old panel separately. Close the run at the fixed endpoints even if the joint result, row retention or marginal advantage fails. These are finite-panel development criteria for a structured state with a visible role bit.
