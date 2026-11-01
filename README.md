# SELF-FOLD

English · [日本語](README.ja.md)

**What survives when a model puts an internal state into words and reads those words back?** SELF-FOLD follows short self-descriptions through a frozen language model. Each update receives only the current description or a fixed set of hidden-state slots. The experimenter keeps the transcript outside the model.

| Arm | Input to the next step |
| --- | --- |
| T · text | The model's new description in words. |
| L · latent | Eight slots after one hidden-state update, `G(Z)`. |
| B · roundtrip | The same update, decoded to text and encoded again, `E(D(G(Z)))`. |

L and B start with identical slots and share the update, layer boundaries and decoding policy. Their difference measures what this particular language roundtrip changes. The open research question is whether any change matters for a later task. [Paper (PDF)](paper/self-fold.pdf) · [Read online](paper/main.md) · [Protocol](docs/PROTOCOL.md)

## Experiments

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

The original code, documentation and synthetic fixtures use the Apache License 2.0. Model weights and tokenizers remain subject to their upstream terms. [License](LICENSE) · [Provenance](docs/LICENSING.md)
