# Research reading — the self-reference question after two experiments

Status: development synthesis. Bounded to the structured interface (16
policies, frozen Gemma-2-2b reader, idealized argmax verbalization). No
qualification inputs used; `NOT_QUALIFIED` preserved.

## The question

Does an intermediate state that refers to its own history change when it is
verbalized — T/B (through words) vs. L (through slots)?

## What the three experiments established

**1-bit role memory** (`role_memory_20261104`): a 9-parameter controller
discovered, from outcome reward alone, that it must forward "the off/off world
selects the switched action" as a bit, and solved every opposed-history pair.

**Message boundary** (`message_boundary_20261106/run`): soft vs hard boundary,
learned vs supervised vs supplied vs no message. Result: identical for all
message arms (100%), identical failure for no-message arms (exactly the
never-flipped-role subset, 12.5%). The learned bit is a pure forward of the
current instruction's role coordinate, not history accumulation.

**Mixture sweep** (`message_boundary_20261106/mixture`): genuinely mixed rows
(β = 1.0/0.8/0.6) and a `traced` channel that writes the role into mixture
weights where only a soft boundary can see it. Result: the trace was present
but unreadable; soft never exceeded hard; at β=0.6 soft *fell below* hard due
to reader fidelity, not memory.

## The reading

1. **The memory channel of this task is effectively discrete.** In every
   configuration where continuous information was demonstrably written into
   the state (β<1 mixtures, ε-trace marks), it either carried nothing the task
   needed or was unreadable by the fitted reader. The load-bearing residue is
   exactly one bit, and it is already symbolic.

2. **"Verbalization loss" is the wrong frame at this interface.** The loss
   lives upstream: what the *writer* puts into the state, not what the
   boundary strips out. A soft handoff cannot rescue a bit the writer never
   wrote; a hard handoff loses nothing the writer did write — provided the
   writer's output is itself symbolic.

3. **The real open question sharpens to a writer question:** does the model's
   *own* verbalization D emit the self-referential symbol it will need? Our
   learned writer discovered it. A frozen LM writer — emitting natural text —
   may or may not. That is the self-reference effect worth measuring, and it
   is unmeasurable until the native reader and carrier are separated
   (PR #28's S/N/C diagnostics: the native pipeline currently cannot even
   read the answer E*).

4. **A falsifiable nuance remains on the reader side.** "Unreadable trace" was
   shown for a 4-parameter linear pointer over row-contrast features. A
   richer reader might decode the ε-trace. The stronger claim — mixture
   weights are unreadable *in principle* — requires a capacity-matched
   nonlinear reader control.

## Next experiment proposal: W-RITE (writer-initiated symbolic residue)

Goal: test whether the model's own D writes the role bit, and if it can be
trained or elicited to.

Design sketch (DESIGN_ONLY, needs manifest freeze):

- Replace the learned `write_logits` writer with a **frozen LM writer**:
  Gemma reads the current table + instruction and emits a short text
  self-report ("the off/off row now points to X"). Decode a bit from the text
  (e.g., whether it names the correct row). Feed that bit, like the learned
  bit, into the next edit.
- Arms: `d_frozen` (natural text, no training), `d_prompted` (few-shot
  convention hint), `d_learned` (our learned writer as ceiling), plus
  `none`/`supplied` as floor/ceiling controls.
- Key endpoint: opposed-history accuracy for `d_frozen` vs `supplied`. If the
  frozen writer's text already contains the bit, B ≈ L is restored by natural
  language alone; if not, prompting vs learning shows what is needed to
  install it.
- Gates: (a) PR #28 S/N/C diagnostics separate reader/carrier failures on
  this writer's text; (b) a decode audit verifies the text genuinely carries
  the bit (not just correlated menu content); (c) same frozen reader/checkpoint
  bindings, exhaustive strings, parity stratification.

Alternative if the writer route stays blocked: a **capacity-matched
nonlinear reader** control on the traced channel — does a 2-layer pointer
learn to read the ε-trace that the linear pointer missed? This would
separate "weights unreadable" from "this reader can't read weights".
