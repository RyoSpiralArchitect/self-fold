# Self Without a Transcript

The working paper covers T/L/B recurrence, the native-EOS comparison, paired donor readouts and position-controlled interventions on Mistral and Gemma. It also reports which downstream locations retain the lexical contrast when the donor's options are reordered or masked. The proposed confirmation study has separate qualification and data requirements. [PDF](self-fold.pdf) · [Read online](main.md) · [LaTeX source](main.tex) · [Donor readouts](../provenance/slot_content_20261101/RESULTS.md) · [Downstream localization](../provenance/downstream_span_20261101/RESULTS.md)

The paper now includes question-independent carrier screens, the first codec-only fit and a read-only opposed-donor audit. Source-text NLL improved while functional accuracy stayed near 50%. [Learning evidence](../provenance/carriers_20261102/RESULTS.md)

The paired training comparison and single-source geometry follow-up add a clause-order control. A neutral reference exposed the first-mentioned tray in 93/96 descriptions, while the trigger action was correct in both readings of only 6/48 policies. [Paired training and geometry evidence](../provenance/policy_binding_20261103/RESULTS.md)

The bibliography has 13 entries. The [reference audit](REFERENCE_AUDIT.md) documents the scope of the literature discussion and the status of contested results.

Build from this directory with Tectonic or TeX Live.

~~~bash
tectonic main.tex
# or
latexmk -pdf main.tex
~~~
