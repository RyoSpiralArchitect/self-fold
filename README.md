# SELF-FOLD

English · [日本語](README.ja.md)

**What survives when a model puts an internal state into words and reads those words back?** SELF-FOLD follows short self-descriptions through a frozen language model. Each update receives only the current description or a fixed set of hidden-state slots. The experimenter keeps the transcript outside the model.

| Arm | Input to the next step |
| --- | --- |
| T · text | The model's new description in words. |
| L · latent | Eight slots after one hidden-state update, `G(Z)`. |
| B · roundtrip | The same update, decoded to text and encoded again, `E(D(G(Z)))`. |

L and B start with identical slots and share the update, layer boundaries and decoding policy. Their difference measures what this particular language roundtrip changes. The open research question is whether any change matters for a later task. [Working paper](paper/main.md) · [Protocol](docs/PROTOCOL.md)

## Experiments

- **Does a short token limit explain the repeated B state?** On Mistral-7B-Instruct-v0.3, two opposed descriptions and two updates produced one B state per stopping policy. All 18 generations under native EOS stopped naturally. B still verbalized both starts with the same 13-token sentence. The four updated L tensors were distinct within each policy. [Conditions, outputs and state hashes](provenance/mistral_eos_20261101/RESULTS.md)
- **CPU reference.** Four trials on a random-weight decoder check shared initialization, state restoration, compute accounting and the T/L/B paths. [CPU results](provenance/launch_cpu_20261101/RESULTS.md)

## Reproduce

From a clone of this repository, using Python 3.11 or later:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,hf]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
bash scripts/run_smoke.sh runs/tiny-001
~~~

Each new trial requires a fresh output directory. The experimental records include the tested source commit, model identity, raw scores and stopping reasons. [Pretrained runs and integrity checks](docs/REPRODUCING.md) · [Model and state interfaces](docs/ARCHITECTURE.md) · [Semantic qualification criteria](docs/QUALIFICATION.md)

The original code, documentation and synthetic fixtures use the Apache License 2.0. Model weights and tokenizers remain subject to their upstream terms. [License](LICENSE) · [Provenance](docs/LICENSING.md)
