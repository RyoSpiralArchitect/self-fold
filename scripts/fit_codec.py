"""A fixed first codec fit. Qualification material and G teachers are never loaded."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from selffold.calibration import codec_examples
from selffold.carrier_probe import capture_description
from selffold.cli import ROOT, make_host
from selffold.codec_learning import EncoderBridge, codec_loss, save_bridge, load_bridge
from selffold.records import State, RunStore, code_manifest, file_sha, digest
from selffold.resources import inventory
from selffold import prompts

CHECKPOINT_SHA = "f3503907cc4d2ace62abccf30393a05919827c1c8969b2f2755f15627b9756ad"


def parameter_sha(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update((name + str(tuple(p.shape)) + str(p.dtype)).encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--capture-route", choices=["workspace", "native_tail"], required=True)
    p.set_defaults(backend="hf", cpu_threads=1, device="cuda:0", dtype="bfloat16", slots=8,
                   cut=None, writer=None, max_context=1024, max_new_tokens=None, until_eos=True,
                   wall_seconds=1800, transport=None, encoder_bridge=None, reviewed_local_code=None)
    args = p.parse_args()
    args.profile = ROOT / "configs/models/gemma2b.yaml"
    folder = ROOT / "data/codec_preparation_20261102"
    store, start = RunStore(args.out), time.monotonic()
    source = code_manifest(ROOT)
    store.write("source.json", source)
    store.write("environment.before.json", inventory(ROOT))
    config = {"steps": 64, "effective_batch": 8, "rank": 16, "learning_rate": 3e-4,
              "optimizer": "AdamW", "weight_decay": 0., "betas": [.9, .999], "epsilon": 1e-8,
              "seed": 47, "validation_steps": [0, 16, 32, 48, 64], "loss": "mean_source_text_plus_native_eos_nll",
              "selection": "lowest_dev_nll_then_earliest_step", "backbone": "frozen",
              "capture_route": args.capture_route, "donor_forward_length": 256,
              "scope": "encoder_codec_only_no_G_teacher_or_qualification_answers", "wall_seconds": 1800}
    store.write("request.json", {**config, "dataset_manifest_sha256": file_sha(folder / "manifest.json"),
                                "checkpoint_sha256": CHECKPOINT_SHA})
    try:
        if source["git_status"] != "" or not source["git_commit"]:
            raise ValueError("Need a clean source seal")
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["qualification"]["status"] != "sealed_before_fitting":
            raise ValueError("Qualification commitment must precede fitting")
        examples = {s: codec_examples(folder, s) for s in ("train", "dev")}
        if (len(examples["train"]), len(examples["dev"])) != (192, 96):
            raise ValueError("Unexpected train/dev denominators")
        host, checkpoint = make_host(args)
        store.write("checkpoint_identity.json", checkpoint)
        store.write("host_identity.json", host.identity)
        if checkpoint["checkpoint_sha256"] != CHECKPOINT_SHA:
            raise ValueError("Unexpected checkpoint")
        parity = host.native_parity()
        store.write("native_parity.json", parity)
        if not parity["exact"]:
            raise ValueError("Native parity failed")
        end_token = host.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if end_token != 107 or end_token not in host.eos_ids:
            raise ValueError("Unexpected native Gemma turn-ending token")
        backbone_before = parameter_sha(host.model)
        cached = {}
        for split, rows in examples.items():
            cached[split] = []
            for row in rows:
                state, layout = capture_description(host, row["source"], args.capture_route)
                cached[split].append(state.slots.detach().clone())
                store.append("cached_sources.jsonl", {"split": split, "case": row["id"],
                    "state_sha256": state.sha(), "source_sha256": digest(row["source"]),
                    "positions": layout["capture_positions"]})
        bridge = EncoderBridge(host.width, config["rank"]).to(host.device)
        if sum(p.numel() for p in bridge.parameters()) > 2_000_000:
            raise ValueError("Trainable parameter cap exceeded")
        optimizer = torch.optim.AdamW(bridge.parameters(), lr=config["learning_rate"],
                    weight_decay=0., betas=(.9, .999), eps=1e-8)
        generator = torch.Generator().manual_seed(47)
        schedule = []
        while len(schedule) < 64 * 8:
            schedule += torch.randperm(192, generator=generator).tolist()
        validation = []

        def check_path(module, label):
            row, slots = examples["dev"][0], cached["dev"][0]
            checking = EncoderBridge(host.width, config["rank"]).to(host.device)
            checking.load_state_dict(module.state_dict())
            expected = float(codec_loss(host, checking, slots, row["target"], end_token_id=end_token).detach())
            with torch.no_grad():
                injected = State(slots=module(slots))
            ids, locations = host.render(prompts.read_prompt())
            target = host.tokenizer.encode(row["target"], add_special_tokens=False) + [end_token]
            logits, _ = host.forward(ids + target, state=injected, input_positions=locations[prompts.INPUT_MARKER])
            actual = float(torch.nn.functional.cross_entropy(logits[0, len(ids)-1:-1].float(),
                                torch.tensor(target, device=host.device)))
            difference = abs(expected - actual)
            store.append("path_parity.jsonl", {"stage": label, "training_path_nll": expected,
                         "inference_path_nll": actual, "absolute_difference": difference, "tolerance": 1e-5})
            if difference > 1e-5:
                raise ValueError("Training and inference paths differ")

        check_path(bridge, "initial")

        def validate(step):
            values = []
            with torch.no_grad():
                for row, slots in zip(examples["dev"], cached["dev"], strict=True):
                    values.append(float(codec_loss(host, bridge, slots, row["target"], end_token_id=end_token)))
            value = sum(values) / len(values)
            artifact = store.path / f"encoder_step_{step:03d}.npz"
            sha = save_bridge(artifact, bridge, host.identity, dataset_sha256=file_sha(folder / "manifest.json"),
                    source_sha256=source["source_sha256"], training={**config, "completed_steps": step}, capture_route=args.capture_route)
            entry = {"step": step, "dev_mean_nll": value, "examples": len(values), "artifact": artifact.name, "sha256": sha}
            validation.append(entry)
            store.append("validation.jsonl", entry)

        validate(0)
        for step in range(1, 65):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in schedule[(step-1)*8:step*8]:
                loss = codec_loss(host, bridge, cached["train"][index], examples["train"][index]["target"], end_token_id=end_token)
                losses.append(float(loss.detach()))
                (loss / 8).backward()
            if any(p.grad is not None for p in host.model.parameters()):
                raise RuntimeError("Backbone received gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=1., error_if_nonfinite=True)
            if float(grad_norm) <= 0:
                raise RuntimeError("Bridge received no learning signal")
            optimizer.step()
            store.append("training.jsonl", {"step": step, "train_mean_nll": sum(losses) / 8,
                                           "gradient_norm_before_clip": float(grad_norm)})
            if step in (16, 32, 48, 64):
                validate(step)
        backbone_after = parameter_sha(host.model)
        if backbone_before != backbone_after:
            raise RuntimeError("Backbone parameters changed")
        selected = min(validation, key=lambda r: (r["dev_mean_nll"], r["step"]))
        trained, _, _ = load_bridge(store.path / selected["artifact"], host.identity, device=host.device)
        check_path(trained, "selected_serialized")
        store.write("selected.json", selected)
        # Only after freezing selection, introduce functional development questions.
        groups = [json.loads(x) for x in (folder / "dev.groups.jsonl").read_text().splitlines()]
        by_id = {r["id"]: (r, s) for r, s in zip(examples["dev"], cached["dev"], strict=True)}
        for group in groups:
            for pole, details in group["poles"].items():
                for phrasing in range(2):
                    case = f"{group['id']}-{pole}-p{phrasing}"
                    row, slots = by_id[case]
                    with torch.no_grad():
                        supplied = {"before": State(slots=slots), "after": State(slots=trained(slots)), "literal": State(text=row["source"])}
                    for route, state in supplied.items():
                        if route != "literal" and phrasing == 0:
                            output = host.decode(state.clone())
                            store.append("generations.jsonl", {"case": case, "route": route, **output.receipt()})
                        fingerprint = state.sha()
                        for world_index, world in enumerate(details["truth_ledger"]):
                            condition = ", ".join(f"{k} is {'on' if v else 'off'}" for k, v in world["world"].items()) or "the policy applies"
                            for first, second in (group["options"], group["options"][::-1]):
                                question = f"Under the current policy, when {condition}, which action applies?\nA = {first}. B = {second}."
                                scores = host.score_choices(state.clone(), question, ["A", "B"])
                                delta = scores["A"]["logprob"] - scores["B"]["logprob"]
                                prediction = None if delta == 0 else ("A" if delta > 0 else "B")
                                gold = "A" if world["choice"] == first else "B"
                                store.append("functional_dev.jsonl", {"case": case, "group": group["id"], "route": route,
                                    "world_index": world_index, "first_option": first, "scores": scores,
                                    "prediction": prediction, "gold": gold, "correct": prediction == gold, "state_sha256": fingerprint})
                        if state.sha() != fingerprint:
                            raise RuntimeError("Development readout mutated source state")
        store.write("environment.after.json", inventory(ROOT))
        store.write("terminal.json", {"status": "COMPLETED_CODEC_DEVELOPMENT_FIT_NOT_QUALIFIED", "selected": selected,
            "optimizer_steps": 64, "trainable_parameters": sum(p.numel() for p in bridge.parameters()),
            "backbone_sha256_before": backbone_before, "backbone_sha256_after": backbone_after,
            "compute": host.meter.snapshot(), "qualification_content_loaded": False,
            "G_teacher_loaded": False, "elapsed_seconds": time.monotonic() - start})
        print(json.dumps(selected))
    except Exception as exc:
        store.write("terminal.json", {"status": "FAILED_NO_RETRY", "error_type": type(exc).__name__,
                                      "message": str(exc), "elapsed_seconds": time.monotonic() - start})
        raise


if __name__ == "__main__":
    main()
