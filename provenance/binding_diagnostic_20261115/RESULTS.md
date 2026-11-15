# Field-binding diagnostic: the fresh-wording miss is one lamp binding

**Development measurement, fully offline on sealed artifacts.
[Plan](PLAN.md) · [code](analyze_fields.py) · [scores](summary.json)**

## Result

Every field error on both sealed panels is **one directional lamp confusion
inside exactly one wording family** — not a diffuse readout failure.

| panel | field | errors | confusion | where |
| --- | --- | ---: | --- | --- |
| primary (structured) | trigger | 0/1,280 | — | — |
| primary | veto | 128/1,280 | `left-off` → `right-off` | style0 only, all 8 topics, both orders |
| primary | actions | 0/1,280 | — | — |
| fresh (readout) | trigger | 160/1,280 | `right-on` → `left-on` | style1 only, all 8 topics, both orders |
| fresh | veto | 128/1,280 | `right-on` → `left-on` | style1 only |
| fresh | actions | 120/1,280 | one action collapsed onto its partner (e.g. true {6,7} → pred {6,6}) | style1 only, topics 3/4/6 (all 40 cases each) |

Field identities: `trigger`/`veto` index = side×2+value over
{(left,right)×(off,on)}; `veto=0` in the table means "no veto"
(keys`[:,2]+1`). Style indexes the panel's two wording families. On the
fresh panel style1 is the family
`"I use the exception if the {s} signal reads {v}" / "I cancel that
exception if the {s} signal reads {v}"` — the `"signal reads {v}"`
construction binds the wrong **side** for value `on` in both the trigger
and veto slots, while the sibling style (`"shows {v}"`-family) reads every
field perfectly. On the primary panel the same pattern inverts direction:
style0 misreads `left-off` as `right-off` in the veto slot only.

## Stable cases

No-veto policies (256/panel): zero guard errors on both panels — the
no-veto field is read perfectly. The fresh panel does carry 32 trigger
errors and 24 action errors on stable sources — all inside the same style1
family — so stability degradation is a *consequence* of the same single
binding fault, not a separate failure mode.

## Error-source independence

The lamp-binding errors and the action-field collapse overlap only
partially (48 sources both; 72 action-only) — two distinct faults, both
confined to style1. The action collapse (second menu action mapped onto
the first on topics 3/4/6) is consistent with the residual 320 incorrect
decisions that remained after the both-fields substitution recovered
9,920/10,240 on the fresh panel (up to 960 candidate decision-slots over
120 action-errored sources).

## Readings

1. The fresh-wording miss decomposes into **two concrete bugs in one
   wording family**: (a) `right-on` read as `left-on` in "signal reads
   {v}" clauses — a side-binding fault; (b) action-field collapse on 3 of
   8 menus. Together they account for the whole gap; fixing fields
   recovers the panel (substitution evidence already showed 9,920/10,240).
2. The fault is **directional and lexical**, not semantic drift: the
   sibling wording in the same panel is error-free, and the primary
   panel's only fault was the mirrored `left-off→right-off` confusion in
   its own style0. Ridge field heads bind lamp side to wording-specific
   surface form.
3. The proposed online experiment now has a precise target: probe
   `reads/shows {v}` variants and near-neighbor paraphrases of the two
   failing clauses on fresh source forms, with the updater held fixed —
   a small bounded generation, not a panel rerun.
4. For the stability-floor miss specifically: the no-veto field itself
   never fails, so stability loss is mediated by trigger/action errors
   on the same style1 sources.

## Limits

- Field predictions come from the sealed ridge heads; this measures *these*
  readers' bindings, not the model's latent knowledge — an E-state probe
  could still hold the correct binding.
- Both panels share the eight menus; "style1" is a family of four
  templates, so the generalization claim is at clause level.
- The online half (new source forms through the frozen host) is declared
  but not executed — the pinned checkpoint was unreachable at run time.
