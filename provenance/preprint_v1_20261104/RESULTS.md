# Preprint v1 experiments

The [prospective plan](PLAN.md) fixed both endpoints before any new model scores. Each result below is from the pinned Gemma-2-2B-IT model. The two studies have separate criteria and use different adapters.

## Six histories through a discrete message

Each physical table contains three continuous action rows derived from frozen Gemma clause features and the Section 6 value head. The table is identical across all six assignments of previous roles. The previous assignment is supplied to the writer. The later instruction selects a role. The reader uses the message to address one physical row. Full-vocabulary Gemma answers compare that row against the two alternatives to the target, with both A/B mappings. All three unordered pairs were scored for every possible output row before grading. Eight physical test tables cover four held-out combinations of familiar actions and two new lamp-name pairs. The six assignments were familiar. Every complete history requires twelve correct native answers. The 576 logical choices per arm reuse 144 native queries on the test set.

| Method | Distinct codes | Correct address histories | Complete native test histories | Native test choices |
| --- | ---: | ---: | ---: | ---: |
| Supplied three-bit code | 6 | 6/6 | 48/48 | 576/576 |
| Exhaustive return fit, two bits | 4 | 2/6 | 16/48 | 472/576 |
| Gradient fit, two bits, all three seeds | 4 | 2/6 | 16/48 | 472/576 |
| Exhaustive return fit, three bits | 6 | 6/6 | 48/48 | 576/576 |
| Gradient fit, three bits, seeds 17 and 41 | 6 | 6/6 | 48/48 | 576/576 |
| Gradient fit, three bits, seed 73 | 5 | 5/6 | 40/48 | 526/576 |

Both gradient widths trained for 512 updates on the same native-reward table, with the categorical expectation summed exactly. The exhaustive reference considered all partitions with at most four or eight codes and chose its row readout using only fitting rewards. The gradient endpoints were frozen before the eight test tables were constructed. In the third three-bit seed, histories 2 and 3 both received symbol 7. Erasure gave 288/576 individual test choices for every fitted endpoint. It retained one or zero complete histories per physical table, depending on the decoder's zero symbol. The supplied-code criterion passed. The registered all-seed criterion for learning three bits failed. [Frozen writers, schedules and readers](capacity/fit_closed.json) · [Fit, test and erasure by seed](capacity/results.json)

For a fixed table a two-bit message can identify at most four of six histories, since every pair requires incompatible later answers on at least one common query. In a separate post-result check, a supplied code retained the first four assignments and reused the first two symbols for the last two. It attained 32/48 complete histories and 432/576 individual choices. The learned two-bit endpoints attained fewer complete histories but more individual correct choices. [Follow-up definition](REPLAY_PLAN.md) · [Exact replay](replay/receipt.json)

## Independently encoded teachers

The raw experiment started from the Section 5.1 literal focus reader. Only its query/value adapters received 192 updates on independently encoded raw description, case and rule states. Each update included both policies, four later worlds and both displayed orders. The frozen encoder captured a state before seeing the question or options. We evaluated the unchanged parent and the trained adapter on a fresh source grammar and lamp names with familiar question wording. All 160 fresh stage states were captured after saving the new reader.

| Trained reader | Description | Case table | Compact rule |
| --- | ---: | ---: | ---: |
| Matched raw teacher | 254/512 | 263/512 | 123/256 |
| Correct original and opposed reversal | 0/512 | 0/512 | 0/256 |
| Complete matched units | 0/16 | 0/16 | 0/16 |
| Zero slots | 256/512 | 256/512 | 128/256 |
| Teacher presented as ordinary text | 431/512 | 415/512 | 180/256 |

The unchanged parent gave 257/512, 256/512 and 128/256 on matched slots. Its literal-text scores were 463/512, 495/512 and 206/256. A source and a case unit each require 32 native choices. The single-order rule unit requires 16. The paired reversal requires two correct native winners and a sign change in the A/B margin. No stage passed. Under the prospective stopping rule the two-map admission and language-insertion comparison were not opened. This test diagnoses this adapter and capture interface on these texts. [Source bind and saved endpoint](raw/fit_closed.json) · [All conditions and gates](raw/summary.json)

The structured decision reader comes from the Section 5 component adapter after 96 supervised calibration updates on stored action rows. Its parameter file is distinct from the Section 5.1 focus reader. [Original calibration](../state_only_20261104/PLAN.md) · [Frozen weights](../state_only_20261104/run/produce/fit_closed.json)

## Verification

A new process on the pinned backbone reproduced all 216 scored capacity queries, including the fitting queries, and all 10,240 native decisions on the raw panel. Full-vocabulary winners, both A/B scores and the hashes of all 160 raw teachers matched. [Replay receipt](replay/receipt.json) · [Artifact terms](ARTIFACT_LICENSE.md) · [Reproduction instructions](../../docs/REPRODUCING.md#preprint-v1)
