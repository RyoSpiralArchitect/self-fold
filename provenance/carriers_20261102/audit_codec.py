"""Read-only replay of codec NLL and fixed donor counterfactuals. No fitting."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from selffold import prompts
from selffold.cli import ROOT, make_host
from selffold.calibration import codec_examples
from selffold.carrier_probe import capture_description
from selffold.codec_learning import load_bridge
from selffold.records import State, file_sha, code_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    host, _ = make_host(SimpleNamespace(backend="hf", profile=ROOT/"configs/models/gemma2b.yaml",
        checkpoint=args.checkpoint, device="cuda:0", dtype="bfloat16", cpu_threads=1, slots=8,
        cut=None, writer=None, max_context=1024, max_new_tokens=None, transport=None,
        reviewed_local_code=None, wall_seconds=600))
    parity = host.native_parity()
    if not parity["exact"]:
        raise ValueError("Native parity failed")
    if host.identity != json.loads((args.trial/"host_identity.json").read_text()):
        raise ValueError("Replay host differs from the fitted host")
    examples = codec_examples(ROOT/"data/codec_preparation_20261102", "dev")
    source = {r["id"]: r for r in examples}
    route = json.loads((args.trial/"request.json").read_text())["capture_route"]
    cached = {r["id"]: capture_description(host, r["source"], route)[0] for r in examples}
    recorded = {r["case"]: r["state_sha256"] for r in map(json.loads, (args.trial/"cached_sources.jsonl").read_text().splitlines()) if r["split"] == "dev"}
    assert {k: v.sha() for k, v in cached.items()} == recorded
    prompt, positions = host.render(prompts.read_prompt())
    validations = [json.loads(x) for x in (args.trial/"validation.jsonl").read_text().splitlines()]
    receipts = []
    with (args.out/"dev_nll.jsonl").open("x") as output:
        for entry in validations:
            path = args.trial/entry["artifact"]
            if file_sha(path) != entry["sha256"]:
                raise ValueError("Fitted artifact changed")
            bridge, _, _ = load_bridge(path, host.identity, device=host.device)
            matched = []
            for case, row in source.items():
                controls = {"matched": cached[case]}
                if entry["step"] in (0, 64):
                    opposite = case.replace("-plus-", "-minus-") if "-plus-" in case else case.replace("-minus-", "-plus-")
                    controls.update(opposite=cached[opposite], zero=State(slots=torch.zeros_like(cached[case].slots)))
                target = host.tokenizer.encode(row["target"], add_special_tokens=False) + [107]
                for control, base in controls.items():
                    with torch.no_grad():
                        injected = State(slots=bridge(base.slots))
                    logits, _ = host.forward(prompt+target, state=injected, input_positions=positions[prompts.INPUT_MARKER])
                    logprob = logits[0, len(prompt)-1:-1].float().log_softmax(-1)
                    nll = -logprob.gather(1, torch.tensor(target, device=host.device)[:, None]).mean().item()
                    output.write(json.dumps({"step": entry["step"], "case": case, "group": row["group"],
                                 "control": control, "nll": nll, "state_sha256": injected.sha()})+"\n")
                    if control == "matched":
                        matched.append(nll)
            difference = abs(sum(matched)/len(matched) - entry["dev_mean_nll"])
            receipts.append({"step": entry["step"], "reported_mean_nll": entry["dev_mean_nll"],
                             "replayed_mean_nll": sum(matched)/len(matched), "absolute_difference": difference})
            if difference > 1e-5:
                raise ValueError("Read-only NLL replay differs from the fitted receipt")
    receipt = {"status": "COMPLETED_READ_ONLY_CODEC_REPLAY", "native_parity": parity, "source": code_manifest(ROOT),
               "audit_script_sha256": file_sha(Path(__file__)), "validation_replay": receipts,
               "compute": host.meter.snapshot(), "optimizer_steps": 0,
               "qualification_loaded": False, "elapsed_seconds": time.monotonic()-started}
    (args.out/"receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":receipt["status"],"replayed_checkpoints":len(receipts)}))


if __name__ == "__main__":
    main()
