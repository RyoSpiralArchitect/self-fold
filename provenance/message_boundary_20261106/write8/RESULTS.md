# Start-address repair — saved-report refit (write8)

The initial-update defect is repaired: all six text-feature fits now complete
the first update on **896/896** source/instruction units, versus **0/896** in
the archived fits. The same-budget refits do **not** establish reliable
four-update continuation. Their complete-history scores range from **9.97%
to 14.28%**, around the unchanged no-message score of **12.5%**.

This is a completed controller-training experiment using saved Gemma reports
and saved native scores. It makes **zero new language-generation or native
scoring calls**. The original qualification challenge remains unopened.

## Intervention and execution

Before any report exists, the controller now uses its existing trainable
`address[0]`. Previously, a zero text-feature vector passed through a bias-free
projection, so both greedy pointers selected row 0 and destroyed the second
action. The repair adds no parameters or target information. Subsequent
reports, including erased zero-feature reports, still use `feat_addr`.

The [protocol](PLAN.md) was sealed at commit
`1ebb1454e02f0d2e037b2c2a50e5c8c5c42e5173` before fitting. All eight arms ran
for 512 updates at batch 128 with seeds 17, 41 and 73; only final checkpoints
were evaluated. The original initialization hashes and sampling schedules
were retained. No feature, seed, parser or budget search was performed.

- All **24 archived fits** exactly reproduced write7's published depth,
  complete-history, role-exchange, donor and twin counts under its old start
  semantics. See [archive replay](produce/archive_replay.json).
- All **24 refits** completed. The **18 discrete/control fits** reproduced
  their original final parameter hashes exactly; all **six text fits**
  received a nonzero start-address gradient. See [fit closure](produce/fit_closed.json).
- A separate process regenerated **all 48 archived/refitted summaries**
  exactly from the saved compact trajectories and native score table. See
  [reduction](reduction.json).
- All **40,320 saved reports**, including **68 token-censored reports**, were
  retained. Hard source rows were restored by matching saved native-call
  byte hashes, without assuming that the compiler emitted the gold action.
  The 32 archived input files remained unchanged. See [input hashes](INPUTS.json)
  and [restored inputs](produce/restored_inputs.json).

The CPU run used one Torch thread and two evaluation workers on Furnace. It
closed after **1,298 seconds** without a technical stop or retry. The
[source seal](produce/source.json), [environment](produce/environment.json),
[request](produce/request.json) and [terminal receipt](produce/terminal.json)
bind the execution.

## Complete-history results

Each seed is evaluated on **2,458,624 histories**: 64 source realizations ×
14^4 instruction strings. A history passes only if all four updates are
correct in all four worlds and both answer orders. These are repeated uses
of eight known action menus, not millions of independent tasks.

The unfitted-mask panel contains **82,944 histories** using only the six
masks excluded from fitting. It does not introduce new topics, action words
or report-generation conditions. The constructed-twin panel contains
**6,272 complete units**, each requiring both opposed histories to be correct
over all worlds and answer orders; it supplies the intermediate state and
bypasses the initial update.

| Report features | Seed | All-four histories / 2,458,624 | Unfitted-mask histories / 82,944 | Intact twins / 6,272 |
| --- | ---: | ---: | ---: | ---: |
| Directed (`lmtext_rt_directed`) | 17 | 245,224 (9.97%) | 8,424 (10.16%) | 0 (0.00%) |
| Directed | 41 | 348,376 (14.17%) | 12,312 (14.84%) | 392 (6.25%) |
| Directed | 73 | 329,168 (13.39%) | 11,664 (14.06%) | 392 (6.25%) |
| Natural (`lmtext_rt_natural`) | 17 | 322,280 (13.11%) | 10,440 (12.59%) | 1,680 (26.79%) |
| Natural | 41 | 351,176 (14.28%) | 10,224 (12.33%) | 2,072 (33.04%) |
| Natural | 73 | 303,576 (12.35%) | 10,344 (12.47%) | 1,680 (26.79%) |

Every text fit passes **896/896 first-update units**. Erasing the report
reduces complete twins to **0/6,272** in every arm. Nonzero intact twins
therefore support a restricted report effect, without establishing a complete
four-update path.

The six unchanged comparison arms have the same endpoints in every seed:

| Channel | All-four histories / 2,458,624 | Intact twins / 6,272 |
| --- | ---: | ---: |
| Directed report decoded to a bit (`lm_rt_directed`) | 2,458,624 (100%) | 6,272 (100%) |
| Natural report decoded to a bit (`lm_rt_natural`) | 326,704 (13.29%) | 1,008 (16.07%) |
| Prior-aware report decoded to a bit (`lm_state_prior_true`) | 339,080 (13.79%) | 1,064 (16.96%) |
| Supplied role bit (`supplied`) | 2,458,624 (100%) | 6,272 (100%) |
| Reward-learned bit (`reward`) | 2,458,624 (100%) | 6,272 (100%) |
| No message (`none`) | 307,328 (12.50%) | 0 (0%) |

Native decision accuracy shows where the repaired text path loses ground:

| Report features | Seed | Update 1 | Update 2 | Update 3 | Update 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Directed | 17 | 100% | 51.34% | 50.40% | 50.28% |
| Directed | 41 | 100% | 51.34% | 52.15% | 51.42% |
| Directed | 73 | 100% | 51.34% | 52.42% | 51.63% |
| Natural | 17 | 100% | 60.27% | 50.89% | 50.05% |
| Natural | 41 | 100% | 61.61% | 51.00% | 50.06% |
| Natural | 73 | 100% | 60.27% | 50.92% | 49.95% |

Each depth has 19,668,992 repeated decisions. Exact numerators, per-topic and
role-exchange counts, opposed/equivalent donor controls, erased routes and
prior mismatches are retained in [refitted summaries](produce/refit_summary.json).
The [archived summaries](produce/archived_summary.json) preserve the comparison.
No inferential confidence interval treats these repeated decisions as
independent observations.

## Post-result diagnosis: a missing role reference

This analysis was added **after** the fixed refit; it did not select a model,
change a threshold or alter the experiment. The pointer reader observes the
table's normalized row contrast and the report features. Its contrast says
which rows differ from row 0, but discards absolute action identity. The
decoded-bit comparator additionally uses the action options and **source
pole** to map an action name to a role. The text-feature pointer does not
receive that reference. Their scores are consequently not an isolated test
of lexical decoding versus learned text features with matched information.

The [saved-input diagnostic](diagnose_aliases.py) counts identical pointer
observations with opposite required off/off roles, restricting to correct
transitions (`out == instruction`). Over 16 source/topic-pole pairs × 14
nonconstant priors × 14 instructions, **all 3,136 records** participate in
such conflicts for both report styles. This holds even when the exact report
text replaces hashbag features while the same normalized contrast is kept.
With equal weight per report record, the best deterministic role classifier
on either observation is correct on **1,568/3,136 (50%)**. This is a finite-bank
role-classification bound, **not a bound on all-four trajectory accuracy** or
natural-language capacity.

For a concrete directed-report witness, two records both say “earliest
reports” and have contrast `[0, 1, 1, 1]`; source pole 0 with output 1 requires
role 1, while source pole 1 with output 14 requires role 0. A larger reader
with those same observations cannot distinguish the pair. A reader given an
additional action/role reference would be a different interface.

Restricting to the original prior 8 gives a weaker, panel-specific bound:
**115/224** for directed text/features, **191/224** for exact natural text and
**190/224** for natural hashbag features. These counts, input definitions and
contradictory witnesses are in [the diagnostic receipt](post_result_aliases.json).
They do not prove that all remaining failures have a single cause.

## Closure and next question

The registered engineering repair and bounded refit are complete. The old
all-four zeros were forced by initialization and cannot diagnose missing
report information. After repair, the text channel still fails to sustain
four updates under this budget. This is not evidence of a general inability
to learn from language, and it does not qualify the original semantic-update
challenge.

The next experiment should first match the role reference available to the
text and decoded-bit channels, then separate retained-reference, erased and
opposed-reference conditions. Its origin and bandwidth must be explicit;
reconstructing it from the gold target would not establish learned retention.
It should test correct and erroneous realized
outputs separately: the directed prompt asks about the instruction, so its
100% path score does not establish faithful reporting of output errors.
These are prospective questions, not additional runs in write8.

## Engineering validation and reproduction

The complete Furnace checkout passed **362 tests, zero failures or skips**
([JUnit receipt](engineering-tests.xml)). Both GitHub engineering workflows
passed at the execution seal. Regression checks cover cache-free hosts,
the realized-output lookup, learnable initialization, report erasure and a
perfect feature channel traversing all four updates. Deliberate substitutions
of the old zero-feature start and requested-output lookup are rejected.

The Mac run passed 360 tests; its two failures were a pre-existing HF
compatibility failure reproduced at the reviewed head and missing historical
objects in a partial clone. No assertions were weakened to accommodate them.
See [validation receipt](validation.json).

To verify the saved results without fitting or model calls:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 scripts/reduce_report_replay.py \
  provenance/message_boundary_20261106/write8/produce
PYTHONPATH=src python3 \
  provenance/message_boundary_20261106/write8/diagnose_aliases.py
```

For a new refit, use the command and fixed budget in [PLAN.md](PLAN.md).
[SHA256SUMS](SHA256SUMS) covers the published write8 files; artifacts derived
from Gemma retain the [artifact terms](ARTIFACT_LICENSE.md).
