# Description-only carriers before learning

Package 1 of the [qualification plan](../../docs/SEMANTIC_GATE_PLAN.md). Development diagnostic, no training and no semantic qualification claim.

## Frozen scope

- Two existing checkpoint fingerprints, Mistral-7B-Instruct-v0.3 and Gemma-2-2B-IT, one middle cut each, eight rows, BF16 eager inference. No layer or slot-count search.
- Two carriers per checkpoint, the current workspace E and eight trailing native positions of the source description. Every donor forward is padded to 256 with the existing neutral period ID. Padding and selection receive no question, alternatives or answers.
- Twelve exposed topics. The six original content groups retain their original source sentences. Six topics from the exposed downstream panel receive two explicit full-sentence source phrasings per pole, frozen in `data/carrier_sources.dev.json`. These are development additions, not a new holdout.
- All 48 source states per route are saved before constructing questions. Reuse each unchanged state across the topical question and a generic matching question, each in both option orders. Native text and uninjected, zero, donor-independent random, norm-only, reversed-row and other-topic donors supply controls.
- Donor-independent random directions use seed 9292 and the route's mean row norm. Norm-only directions use seed 4711. Unrelated donors rotate to the next topic at the same pole and phrasing. Report every control separately.
- Check native hook parity and full-vocabulary neutral self-replay at the actual full-label forward shape for every question/order. Any failure invalidates that cell. Save native offsets or exact token boundaries and actual donor positions and IDs.
- Read each carrier once with D under natural EOS within the fixed context bound. Retain nonempty censored outputs and mark empty outputs. Observer text is never encoded back into these donors.
- Score full A/B log likelihoods. Ties count as unsuccessful. Report every source/phrasing/question/order, then average within each topic and over the twelve topics. The positive development screen is at least 80% macro correctness and at least a 10-point advantage over every empty/zero/random/unrelated-state control. Reverse-row and norm-only outcomes remain separate diagnostics.
- One sequential job per checkpoint contains its two carrier cells. Each job has 20,000 forwards, 20 million processed positions and 1,800 seconds after host construction, with no automatic retry. No generated output or raw score selects extra positions or conditions.

## Closure

Passing this screen selects a candidate route for further learning and review. Failing all four cells closes this bounded native-interface diagnostic and permits the independently specified learning work to proceed once its data and implementation prerequisites are ready. No gate is issued from these scores. The full punctuation mechanism remains outside this package.

The source commit is recorded in each trial before execution. Checkpoint weights and local operator paths are not copied into the repository. Raw state arrays, donor metadata, label scores, generation stop receipts and terminal failures are retained.
