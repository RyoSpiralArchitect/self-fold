# Architecture and state contract

## Data flow

`State` owns exactly one current text or one `[K,d]` tensor. `engine.step` never receives trajectory history. `run_trajectory` retains external history for inspection only.

T: text -> fixed prompt -> greedy text generation -> text.

L: slots -> injection at pre-cut input positions -> decoder -> post-writer output positions -> explicit Transport -> new pre-cut-coordinate slots.

B: the same L update -> D on the updated slots -> current text -> E -> new slots.

`encode` E captures contextualized native slot residuals before the selected cut. `decode` D injects those slots into a fixed verbalization prompt. Neither E nor D is an independently trained semantic codec in this release. The linear Transport is identity unless a checked local artifact is supplied. `identity_unqualified` names that limitation, not a valid semantic route.

## Boundary definition

The host currently supports a `model.model.layers` decoder layout, and rejects anything else. Cut is the input of decoder layer c, writer is the output of decoder layer w, with `1 <= c <= w < depth`. No final-layer residual is silently treated as a token embedding. The post-writer to pre-cut mapping is explicit and still requires qualification.

Native chat templates are used with one user message. Two unique markers are replaced by fixed numbers of neutral placeholder token IDs. No new learned vocabulary entry is added. Input and output slots use distinct token positions. Their positional/context dependence is a known confound, not removed by matching vector dimensions. Before inference, context overflow fails rather than truncates.

## Observation

The selected cut and writer collect residual, attention and MLP outputs on slot positions. Phimoe gate logits are captured when the expected module exists. Gate entropy/top-k summarize candidate routing logits, **not actual expert dispatch frequency or semantic specialization**. Full attention matrices and every layer/token activation are deliberately not retained by default.

T instrumentation is on a fixed generation boundary rather than the same latent slot population; direct Euclidean comparisons between T and L tensors are not a main endpoint. Within-family L/B comparisons and task effects have clearer interpretation. Across families, widths/layers/tokenizers differ; raw neuron coordinates are not comparable.

## Ownership, observation and effects

State tensors clone at construction and at branch boundaries. Every observer and task readout gets a clone plus preserved RNG state. Hooks are removed in `finally`. The generation path uses `use_cache=False` on every forward and recomputes the bounded prefix. This is intentionally conservative and slow. Cache optimization, if later introduced, must pass no-leakage and exact baseline equivalence tests.

The task scorer evaluates full label likelihood in fresh prompts. Labels with different token lengths are recorded and should be balanced before confirmatory evaluation. Exact score ties produce no winner. The oracle answer is used only after scoring. CLI task evaluation happens after all recurrences, at initial and final checkpoints.

## Files

`host.py`: family-neutral hook/generation interface. `engine.py`: recurrence semantics and isolated branches. `prompts.py`: fixed operations. `records.py`: state, safe storage and source seal. `transport.py`: explicit mapping. `gates.py`: evidence-bound semantic admission. `loading.py`: existing local checkpoint identity/load. `resources.py`: read-only inventory. `metrics.py`: descriptive numerical and lexical diagnostics. `cli.py`: bounded run ownership.

## Costs and evidence

Counters track forwards, processed token positions and generated tokens, not FLOPs. Main recurrence and observer counters are separated per transition. Terminal counters include task/branch work. Loading/hash I/O and wall measurements are separate concerns; no throughput superiority is inferred. Each new run records a source manifest including runnable code, tests, protocol and paper files. Editing source invalidates the binding of an old semantic gate.

The run directory stores raw states as safe NPZ, original dtype, content hashes, outputs and terminal status. `load_state(path_without_extension)` restores and verifies state identity. BF16/FP16 are stored losslessly via float32; float64 is retained as float64. The model is not serialized by the run store.
