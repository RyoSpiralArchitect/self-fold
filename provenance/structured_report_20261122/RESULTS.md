# Learning to report the state that the next edit uses

Training the Gemma-2-2B-IT writer raised complete role reversals after a text roundtrip from **3/96 to 31/96 pairs**. Direct continuation from the same structured midpoint completed **96/96**, as did supplying its correct report. The training gain passed the registered development criterion. The learned route finished 65 pairs below direct continuation. [Plan fixed before fitting](PLAN.md) · [All arms and paired differences](produce/summary.json)

Each pair starts with two histories that produce the same four action rows but exchange which action is the default. The frozen first update encodes this distinction in one bit. Gemma sees only the table and the role of its off/off row, expressed in words. It writes five lines before learning which edit or question follows. On B, only these words reach the fixed parser E. Both B and L then use the same frozen second update and native Gemma recipient. One pair succeeds if both histories produce opposite correct choices under all 14 subsequent masks, all four worlds and both A/B orders, with the correct full-vocabulary winner and A/B margin sign.

## What training improved

| Route | Encodable reports /192 | Correct individual rows /768 | All four rows /192 | Correct default /192 | Complete continuations /192 | Both opposed continuations /96 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct structured state (L) | — | — | — | — | 192 | 96 |
| Exact report through E | 192 | 768 | 192 | 192 | 192 | 96 |
| Free report, original Gemma | 26 | 145 | 33 | 17 | 17 | 3 |
| Free report, trained writer | 192 | 723 | 162 | 124 | 124 | 31 |
| Fixed fields and two names, original Gemma | 192 | 648 | 123 | 101 | 99 | 10 |
| Fixed fields and two names, trained writer | 192 | 724 | 163 | 123 | 123 | 32 |

Each row/role field can score even when another missing field prevents E from constructing the whole state. The original writer sometimes returned light positions or omitted the default name. The trained writer produced an encodable free report in every presentation. All 384 free generations reached native EOS. In the fixed-format diagnostic both writers chose an action at each of five fields, conditioned on their own earlier choices. The gain from 10 to 32 complete pairs persisted when both had that assistance (27 pairs recovered and five lost). [768 reports, tokens or branch logits](produce/reports.jsonl)

The 96 primary twin presentations comprise 24 different physical assignments on four known menus, each shown in four cyclic row orders. Training used separate assignments with one or three active coordinates. The balanced two-and-two assignments were held out from this writer fit. The 21,504 native decisions per arm reuse 192 reports or direct presentations and a finite set of row judgments. The full-vocabulary recipient was evaluated anew on 128 different row/menu/order inputs, including cross-menu inputs. [Held-out prompts and midpoints](produce/test_states.json) · [Native scores and dispatch indices](produce/native/scores.npz)

| Menu | Free B before /24 | Free B after /24 | Fixed fields before /24 | Fixed fields after /24 |
| --- | ---: | ---: | ---: | ---: |
| Recent or earliest reports | 0 | 11 | 1 | 12 |
| Preserve wording or summarize | 1 | 14 | 4 | 12 |
| Centimeters or inches | 0 | 6 | 3 | 6 |
| Sort mail by sender or date | 2 | 0 | 2 | 2 |

Free B recovered 30 pairs and lost two, a net gain of 29.2 percentage points. Three of four menus improved. Enumerating the four-menu block resamples gave a descriptive 95% interval of +5.2 to +50.0 points. Relative to L, the learned free B difference was −67.7 points (−88.5 to −47.9). Both native anchors passed. The prespecified superiority and within-five-points-of-L criteria remained unmet. With a perfect L there was no numerical headroom for an advantage on this panel.

## Which part of the report still matters?

Keeping the learned report's four rows and supplying only its correct **default action** restored all **96/96** twin continuations. Supplying the four correct rows and keeping the writer's default left the result at **31/96**. Correcting the default is a diagnostic intervention with the answer supplied by the experimenter. On this panel, the frozen row-copy editor can use the two action names even when some of their coordinates in the report are wrong. Exact reporting of the whole state was harder: all five fields were correct in **107/192** learned free reports. [Field replacements and next-edit outcomes](produce/outcomes.json)

A descriptive count after scoring shows what happened on the 65 failed pairs. The trained free writer gave both role donors the **same default name on 62 pairs**, and switched the names in the wrong direction on the other three. For mail sorting it kept the same name on 22/24 and reversed in the wrong direction on 2/24. The four-row text was fully correct on 40/48 individual mail presentations. This motivates fitting explicitly paired role-reversal examples with identical rows, judging both donors together on fresh assignments. The next experiment can test whether that supervision recovers correct reversals while preserving row accuracy.

## Execution and scope

One rank-eight adapter fit trained 245,760 Gemma writer parameters on 256 presentations of 64 midpoints, over 128 updates. Its weights were sealed before held-out generation. The frozen recipient, state encoder, first update and second update received no training. The four menus and action vocabulary were familiar. The one-bit role, its display, the five requested report fields and the text parser were supplied. The earlier 602/768 measured answering one named-row question in a visible table. It uses a different task and denominator unit. The results here concern one structured language boundary. The raw eight-slot language insertion and independent qualification remain open. [Fit receipt](produce/fit_closed.json) · [Training trace](produce/training.jsonl) · [Writer weights](produce/writer.npz)

The first run completed the fit and all reports, then stopped when loading the final recipient. The writer's reported generation cap differed from the host identity bound into the historic recipient. The scoring continuation loaded the unchanged checkpoint with that recipient's original host settings. Every saved writer weight, report and input retained its hash. Recovery performed zero fits and zero report generations. An independent checker reconstructed the rows from the visible prompts, parsed all 768 reports, and agreed with all 1,920 route presentations and 128 native dispatches. [First stop and immutable hashes](INITIAL.json) · [Scoring replay](replay.json) · [Independent verification](verification.json) · [Reproduction](RUNS.md)
