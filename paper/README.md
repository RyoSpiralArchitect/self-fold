# Self Without a Transcript

## Reading, changing and reusing a conditional policy

Follow a walker's rule from its original wording through a table of cases and back to a compact rule. The working paper asks which distinction survives when only a current state is available for a later task.

[Read the paper](main.md) · [Download the PDF](self-fold.pdf)

The paper follows a remembered preference into a conditional decision. A supervised memory drove a native action after the original policy text was removed. On familiar clause patterns all 3,072 decisions and opposed-donor reversals were correct. New wording lowered native accuracy to 1,140/1,536. Passing the updater its own edited table exposed a further limit on reuse. An external two-role rearrangement restored all 9,216 second choices from the same tables. [Single edits and two-edit sequences](../provenance/state_only_20261104/RESULTS.md)

A nine-parameter controller then learned to retain the role of a reference row across edits, even when current action tables were identical. Reward-based learning matched explicit bit supervision. A no-bit control exposed why assessing every step and balancing the order of edits matters. [Learning and exhaustive sequence results](../provenance/role_memory_20261104/RESULTS.md)

Sections 5.1 and 6.1 follow the literal grammar gap into two learned eight-slot maps on a frozen native reader. The state-only chain reached 252/512 after native-gradient training, while a supplied correct class rendered as a canonical rule reached 508/512. [Grammar, stage maps and native controls](../provenance/stage_grammar_20261104/RESULTS.md)

The [development record and experiment key](DEVELOPMENT_RECORD.md) preserves all nineteen previous result sections, human-readable experiment names, internal route and arm identifiers, controls and raw evidence links. The [reference audit](REFERENCE_AUDIT.md) and [bibliography](references.bib) retain the source records. Build `main.tex` with Tectonic or `latexmk -pdf main.tex`.
