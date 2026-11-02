# SELF-FOLD

English · [日本語](README.ja.md)

**Can a distinction formed by reinterpreting a current self-description support the next update and a later judgment?** SELF-FOLD studies that continuity and the effect of verbalizing an intermediate state. Each update receives only the current description or fixed hidden-state slots. The experimenter keeps the transcript outside the model. Interfaces are fitted during development and frozen for evaluation.

| Arm | Input to the next step |
| --- | --- |
| T · text | The model's new description in words. |
| L · latent | Eight slots after one hidden-state update, `G(Z)`. |
| B · roundtrip | The same update, decoded to text and encoded again, `E(D(G(Z)))`. |

L and B start with identical slots and share the update, layer boundaries and decoding policy. Their difference measures what this particular language roundtrip changes. The open research question is whether any change matters for a later task. [Paper (PDF)](paper/self-fold.pdf) · [Read online](paper/main.md) · [Protocol](docs/PROTOCOL.md)

## Experiments

- **Can a learned interface make lexical slots usable?** A rank-16 row transformation with a frozen native recipient scored 22/24 decisions, comprising 16/16 on four training topics and 6/8 on two previously exposed topics excluded from this fit. Opposing lexical rows reduced the score to 2/24. This pooled engineering screen passed, while the two-topic transfer screen failed. D-only copying of all 24 original E states converged to one output sentence, with 1/24 exact reconstructions. Mixed-order policy training improved the literal all-four-order joint count from 8/32 to 14/32, while the E-slot count stayed at zero. [Bounded readout experiments](provenance/slot_readout_20261103/RESULTS.md)
- **Does writing out the cases make a policy usable?** On four training groups, the case-trained literal reader chose correctly on 58/64 forward-list decisions and 47/64 when the sentences were reversed. The corresponding slot reader scored 35/64 and 31/64. The literal reader handled both opposed policies in both orders on 13/32 units, the same count as the earlier rule-trained reader. On two previously exposed development groups, the case-trained literal reader scored 27/32 forward and 28/32 reversed. [Matched policy-form experiment](provenance/policy_form_20261103/RESULTS.md)
- **A native reader for the two-stage route.** Four bounded Gemma configurations completed 1,024 optimizer updates. Final-teacher native accuracy was 49.78%, 50.09%, 50.00% and 50.69%, against a 50% strongest null. The joint source/recipient candidate received finite gradients in both modules but missed its admission screen. A zero-update audit identified additional BF16 output ties, while FP32 diagnostic projection did not recover reliable policy use. A separate small-set control improved literal training correctness from 67.19% to 87.50%, while raw slots reached 45.31%. Actual learned G composition and the intended positive continuity result remain open. [All configurations, failures, audit and artifacts](provenance/continuity_20261103/RESULTS.md) · [Integrated two-stage protocol](docs/FUNCTIONAL_CONTINUITY.md)

- **Can a stored policy switch actions?** Two fresh 64-update Gemma fits scored 49.35% and 50.00% on functional questions. Neither correctly switched both opposed donors in any of 288 comparisons. In a separate test, subtracting a neutral encoding let a train-only linear readout identify the first-mentioned tray in 93/96 clause-reordered descriptions. It identified the prescribed action in only 51/96, with both word orders correct for 6/48 policies. [Controlled fits, raw scores and next decision](provenance/policy_binding_20261103/RESULTS.md)

- **First codec learning.** A frozen Gemma-2-2B backbone and a 73,728-parameter encoder bridge completed 64 updates. Dev source-text NLL fell from 4.31 to 1.86, while functional group accuracy stayed near 50%. Opposed donors retained almost the same NLL improvement. Before learning, neither model's workspace or description-tail carrier passed the content screen. [Design, failures, learned artifacts and raw results](provenance/carriers_20261102/RESULTS.md)
- **Which downstream position carries the lexical choice?** On six new topics, transferring only the middle-layer state at the period after the second displayed alternative recovered 22% of the whole-downstream donor contrast on Mistral and 27% on Gemma. Reversing the donor's A/B mapping reversed this contrast. Leaving out just that period still retained 88% and 89%, respectively. These ratios describe paired log-probability differences. [Position ablations, donor masks and raw scores](provenance/downstream_span_20261101/RESULTS.md)
- **Can a question recover the source description?** Mistral-7B-Instruct-v0.3 and Gemma-2-2B-IT each identified 24/24 literal descriptions across six preference contrasts. On both models, 24 distinct updated slot tensors yielded a single greedy verbalization per checkpoint. After re-encoding that text, all 12 paired donor comparisons tied. [Design, controls and raw scores](provenance/slot_content_20261101/RESULTS.md)
- **Which positions reproduce a native lexical choice?** With donor and recipient lengths matched, transplanting only eight middle-layer slots gave 6/12 correct choices on Mistral and 7/12 on Gemma. Transplanting the positions that had already processed the question reproduced all 12 native preference signs on each model, including one native error. These are development interventions on two frozen checkpoints. [Position and length controls](provenance/slot_content_20261101/REPLAY_PLAN.md) · [Results](provenance/slot_content_20261101/RESULTS.md)
- **Does a short token limit explain the repeated B state?** On Mistral-7B-Instruct-v0.3, two opposed descriptions and two updates produced one B state per stopping policy. All 18 generations under native EOS stopped naturally. B still verbalized both starts with the same 13-token sentence. The four updated L tensors were distinct within each policy. [Conditions, outputs and state hashes](provenance/mistral_eos_20261101/RESULTS.md)
- **CPU reference.** Four trials on a random-weight decoder check shared initialization, state restoration, compute accounting and the T/L/B paths. [CPU results](provenance/launch_cpu_20261101/RESULTS.md)

## Reproduce

Use Python 3.11 or later in a local checkout.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,hf]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
bash scripts/run_smoke.sh runs/tiny-001
~~~

Each new trial requires a fresh output directory. The experimental records include the tested source commit, model identity, raw scores and stopping reasons. [Pretrained runs and integrity checks](docs/REPRODUCING.md) · [Model and state interfaces](docs/ARCHITECTURE.md) · [Semantic qualification criteria](docs/QUALIFICATION.md)

The [integrated continuity protocol](docs/FUNCTIONAL_CONTINUITY.md) makes two-step reconstruction, operator omission and a single common-state language insertion the prospective target. The [semantic qualification plan](docs/SEMANTIC_GATE_PLAN.md) defines six bounded work packages from question-independent states through interface learning and an independent gate challenge. Each has a positive endpoint and a condition for advancing with an explicit negative diagnosis. The source-binding comparison and the clause-order diagnostic are complete. The next learning target is which condition selects each action. A supported one-step G update and the independent six-check challenge remain open.

The original code, documentation and synthetic fixtures use the Apache License 2.0. Model weights and tokenizers retain their upstream terms. The learned Gemma bridge arrays include the Gemma agreement and notice. [License](LICENSE) · [Asset conditions and provenance](docs/LICENSING.md)
