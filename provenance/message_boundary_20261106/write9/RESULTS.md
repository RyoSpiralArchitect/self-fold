# Carried source reference: directed reports support four updates (write9)

**The directed-report reader passes the predeclared scaffolded readout gate in
all three seeds; the natural-report reader does not.** On the four fitted action
menus, carrying a source reference raises complete four-update histories from
7.4–15.8% to 100%. Fixing that reference to zero yields 50%, and complementing it
yields 0%. This is a positive result for a specified report interface with an
engineered reference path. It is not learned reference retention, general
semantic recurrence or independent qualification.

The [plan](PLAN.md), [input binding](INPUTS.json), code and criteria were sealed
at `3c8960d5522603cb15e348935291f5634ff7bacd` before fitting. All 12 fixed-budget
fits and 42 evaluations completed on Furnace CPU in 1,295 seconds. Only the
controller was trained here: saved Gemma reports and native scores were reused,
with **zero new language-generation or native-scoring calls**.

## Primary result and controls

Each denominator of 1,229,312 comprises 32 test-source realizations of the four
fitted menus crossed with all 14^4 instruction strings. A history is correct
only if **all four updates, four worlds and both answer orders** are correct.
The large counts repeat a small menu bank; they are not independent tasks.

| Report | Seed | Without reference | With carried reference | Only unfitted masks, with reference |
| --- | ---: | ---: | ---: | ---: |
| Directed | 17 | 91,560/1,229,312 (7.4%) | 1,229,312/1,229,312 (100.0%) | 41,472/41,472 (100.0%) |
| Directed | 41 | 194,712/1,229,312 (15.8%) | 1,229,312/1,229,312 (100.0%) | 41,472/41,472 (100.0%) |
| Directed | 73 | 194,712/1,229,312 (15.8%) | 1,229,312/1,229,312 (100.0%) | 41,472/41,472 (100.0%) |
| Natural | 17 | 192,696/1,229,312 (15.7%) | 69,384/1,229,312 (5.6%) | 1,440/41,472 (3.5%) |
| Natural | 41 | 207,872/1,229,312 (16.9%) | 76,216/1,229,312 (6.2%) | 1,728/41,472 (4.2%) |
| Natural | 73 | 174,384/1,229,312 (14.2%) | 129,864/1,229,312 (10.6%) | 2,736/41,472 (6.6%) |

For directed reports, each seed has the same control results:

| Check | Result per seed |
| --- | ---: |
| Complete first update | 448/448 source × instruction units |
| Reference fixed to zero after update 1 | 614,656/1,229,312 complete histories |
| Reference complemented after update 1 | 0/1,229,312 complete histories |
| Opposite action after reference inversion, conditional on a correct second update | 6,272/6,272 distinct two-update units |
| Constructed twins: both opposite histories completely correct | 3,136/3,136 units |
| Same twins after combined report/reference erasure | 0/3,136 units |

Equivalent donors preserve all primary native choices at updates 2 and 4;
opposed donors make all choices correct for the donor policy at both depths
(9,834,496/9,834,496 repeated native decisions per depth). These full-state donor
exchanges differ from complementing just the reference while leaving the
original report and table in place. The latter intervention gives the 0%
original-policy history score above. The paired 6,272-unit check counts each
source/first-instruction/second-instruction combination once, without repeating
it over possible later instructions.

The natural condition fails complete histories, unfitted masks, the required
gain, and the required drop on reference removal in every seed. It nevertheless
has perfect conditional inversion on its already-correct second-update subset
(2,520, 2,240 and 3,024 units out of 6,272). Thus conditional inversion alone does
not establish a useful reader. Natural complete-history counts with fixed-zero
reference are 42,168, 55,272 and 101,584; with complemented reference they are
30,632, 34,496 and 55,440, each out of 1,229,312 in seed order 17/41/73.
The [gate](produce/gate.json) retains every criterion, including failed ones.

## Vocabulary boundary and frozen comparator

The primary four-menu scope was fixed prospectively. The other four menus are
not hidden or pooled away:

| Report | Seed | Unfitted four menus, carried reference | All eight menus, carried reference |
| --- | ---: | ---: | ---: |
| Directed | 17 | 384,160/1,229,312 (31.2%) | 1,613,472/2,458,624 (65.6%) |
| Directed | 41 | 345,744/1,229,312 (28.1%) | 1,575,056/2,458,624 (64.1%) |
| Directed | 73 | 384,160/1,229,312 (31.2%) | 1,613,472/2,458,624 (65.6%) |
| Natural | 17 | 37,800/1,229,312 (3.1%) | 107,184/2,458,624 (4.4%) |
| Natural | 41 | 83,496/1,229,312 (6.8%) | 159,712/2,458,624 (6.5%) |
| Natural | 73 | 135,352/1,229,312 (11.0%) | 265,216/2,458,624 (10.8%) |

Unfitted-menu success remains 28.125–31.25% for directed reports, so this does
not establish vocabulary transfer. The frozen lexical decoder uses the known
option bank directly. With its reference exposed and carried, it retains
2,458,624/2,458,624 directed histories across all eight menus in every seed;
fixed zero gives 50% and complementing gives 0%. Its natural condition retains
326,704/2,458,624 (13.3%). These intact comparator results reproduce write8;
the learned hashbag and handwritten decoder still have different extractors.
All conditions and panels, including donor and twin controls, are saved in
[summary.json](produce/summary.json).

## Interface and matched budget

Write8 repaired the first update but left an information mismatch: its lexical
decoder used the source action-to-role reference, while the learned text pointer
saw only report features and normalized row contrast. Here one bit is derived
**once** from the two initial compiled one-hot action rows: which canonical
option occupies member 0. It is then copied with each new report, without
refreshing it from source IDs, source text or target labels. A specified routing
operation exchanges the two pointer roles when the bit is one. Reference
retention and routing are engineered; interpreting the report features is learned.

Both text conditions retain 73 stored parameters (including five unused writer
parameters), identical initial weights, sampled batches and 512-update budgets.
The six unreferenced fits reproduce write8's final parameter hashes and complete
trajectory summaries exactly. Six decoded comparators are frozen. Training sees
two updates; evaluation composes four. The original integer start address is
unchanged, and all first-update units remain correct.

The 25,984 active directed/natural reports, including 12 censored natural
reports, are unchanged. The bound prior-state report bank is verified but unused.
Reference interventions happen after update 1 and persist. Fixing the bit to
zero preserves report features and supplies a canonical reference; it does not
force a pointer tie. Combined-message erasure is a separate control. Assertions
verify that each intact, donor and erased route copies its incoming reference
through later steps. The [fit receipt](produce/fit_closed.json) records identical
initialization/sampling hashes and all final parameters.

## Remaining ambiguity and report faithfulness

The predeclared audit uses one weight per saved report record from correct
transitions with nonconstant priors. For the primary menus there are 1,568 such
records. The following is the best possible deterministic **role classification**
on identical report-plus-normalized-row-contrast inputs, not a trajectory score:

| Observation | Directed, no reference | Directed, with reference | Natural, no reference | Natural, with reference |
| --- | ---: | ---: | ---: | ---: |
| Exact text | 784/1,568 | 1,568/1,568 | 784/1,568 | 1,210/1,568 |
| 32-dimensional hashbag | 784/1,568 | 1,568/1,568 | 784/1,568 | 1,198/1,568 |

Reference matching removes the directed conflicts. Natural reports retain
conflicts even with exact text; hash compression adds a smaller further loss
in this audit. These bounds concern the restricted pointer input: the full
absolute action table can contain information the pointer does not inspect.
They do not assign failure to the writer alone or prove that a general language
reader cannot interpret the reports.

The directed prompt asks which action **the instruction** assigned to off/off,
not which action the realized output contains. On all eight menus, the saved
lexical decoder identifies the actual off/off action in 3,136/3,136 records
where output equals instruction, and 8,414/9,856 where they differ. The natural
counts are 354/3,136 and 4,100/9,856, with 2,364 and 4,506 missing mentions. These
are decoder-based checks, not independent semantic annotations. Passing the
correct-history task therefore does not establish faithful reporting of output
errors. Full conflict and instruction/actual-output counts are in the
[report audit](produce/report_audit.json).

## Next anchor and closure

This run closes the matched-reference question within its registered scope.
It retains the natural negative result and the unfitted-vocabulary limitation;
there is no added seed, feature search or extended fitting after inspection.

Two changes should be tested separately next: ask the report to describe the
**realized output**, including erroneous transitions, and replace engineered
reference copying with a learned retention path. Each needs its own fixed
budget, matched interventions and success/failure closure. The present result
provides a working reference interface for those tests. The independent
six-check qualification remains open; this run loads none of its inputs.

## Reproduction and evidence

The run is closed in [terminal.json](produce/terminal.json). Code, environment,
request, curves, weights, all 42 trajectories and their logical-array hashes are
under [produce/](produce/). Compression changes storage order only. The source
seal requires a clean worktree; output must go outside that worktree:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/fit_role_reference.py --out /path/to/new-run --workers 2
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/reduce_role_reference.py /path/to/new-run
```

The [fresh-process reduction](reduction.json) verifies file and logical-array
hashes and reconstructs all 126 panels, 12 paired comparisons and the gate.
[Engineering checks](validation.json) and the [369-test receipt](engineering-tests.xml)
are separate from the scientific gate. The 109 pinned inputs are rechecked;
write6/write7/write8 remain byte-identical. [SHA256SUMS](SHA256SUMS) covers the
new local evidence, and [artifact terms](ARTIFACT_LICENSE.md) apply to the arrays.
