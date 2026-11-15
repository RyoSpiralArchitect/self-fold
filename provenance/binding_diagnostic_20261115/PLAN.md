# Field-binding diagnostic on the sealed supported_g panels (Experiment D, offline half)

**Status: development measurement, fully offline. Reads only sealed
`supported_g_20261104/{structured,readout}` artifacts; no model, no
generation, no refits.**

## Question

`supported_g_20261104` ended its fresh-wording continuation below the 90%
justified-stability floor (396/512) and showed, by field substitution, that
supplying the true veto alone recovered 10,240/10,240 on the primary panel —
so the learned G is faithful and the failure enters through the *field
readers*. The RESULTS closed with the suggested next step: hold the updater
fixed and test **left/right binding and action recognition across fresh
source forms, including stable cases**.

This entry is the offline half of that bounded comparison: instead of new
source forms, it decomposes the *already-sealed* two panels' field-reader
predictions (`states.npz`: `field_base`, `guard`, `field_actions` vs the
true `keys`) into exact confusion structure — which lamp binding breaks, in
which wording family, on which topics, and whether stable (no-veto) cases
are involved.

The online half — the same measurement on *newly generated* fresh source
forms through the frozen host — requires the pinned Gemma checkpoint
(`f3503907…`, furnace) and is declared but not executed here.

## Method

`analyze_fields.py` computes, per panel:

- Trigger field: accuracy + full confusion matrix (`field_base` vs
  `keys[:,1]`), crossed with style × clause order.
- Veto field: same for `guard` vs `keys[:,2]+1` (0 = no veto).
- Action fields: order-independent pair accuracy of `field_actions` vs the
  menu's two actions, per style and per topic, plus overlap with lamp-field
  errors (independent error sources).
- Stable subset (`veto == -1`): every field's error count — the proposed
  "including stable cases" check.

## Readings sought

- If errors are diffuse across fields/styles → the field reader generally
  fails on fresh wording; next spend should retrain readers.
- If errors localize to one binding/one style → the fresh miss is a single
  lexical-binding failure; the correct next experiment is a targeted
  probe (or patch) of that binding, not a panel rerun.
