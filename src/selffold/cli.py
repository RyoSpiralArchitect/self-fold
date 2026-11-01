from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys
import time
import traceback
import torch
import yaml
from .records import RunStore, State, canonical, code_manifest, digest, file_sha
from .resources import inventory, admit
from .host import SlotHost
from .tiny import TinyCausalLM, ByteTokenizer
from .engine import run_trajectory, branch_once, task_readout
from .transport import Transport
from .gates import verify_gate

ROOT = Path(__file__).resolve().parents[2]

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def make_host(args):
    torch.set_num_threads(args.cpu_threads)
    if args.backend == "tiny":
        model = TinyCausalLM(seed=47).to(args.device)
        host = SlotHost(model, ByteTokenizer(), slots=args.slots, max_context=args.max_context,
                        max_new_tokens=args.max_new_tokens, wall_seconds=args.wall_seconds)
        host.identity["checkpoint_sha256"] = "random_tiny_seed47_not_pretrained"
        return host, {"status": "random_weight_tiny_only"}
    if not args.profile or not args.checkpoint:
        raise ValueError("HF requires --profile and an existing --checkpoint path")
    profile = yaml.safe_load(Path(args.profile).read_text())
    admit(device=args.device, required_free_gib=profile["admission_free_vram_gib"],
          host_free_gib=profile["admission_available_ram_gib"], disk_free_gib=2, path=ROOT)
    from .loading import load_local
    model, tok, ident = load_local(Path(args.checkpoint), profile, device=args.device, dtype=args.dtype, local_code_review=args.reviewed_local_code)
    transport = Transport.load(Path(args.transport)) if args.transport else Transport()
    host = SlotHost(model, tok, slots=args.slots, cut=args.cut, writer=args.writer,
                    max_context=args.max_context, max_new_tokens=args.max_new_tokens,
                    transport=transport, wall_seconds=args.wall_seconds)
    host.identity.update({"checkpoint_sha256": ident["checkpoint_sha256"], "profile": profile["id"],
                          "torch_version": torch.__version__})
    import transformers
    host.identity["transformers_version"] = transformers.__version__
    return host, ident

def execute(args):
    if not 1 <= args.steps <= 64:
        raise ValueError("Steps must be between 1 and 64")
    if args.command == "smoke" and args.steps > 4:
        raise ValueError("Smoke is bounded to <=4 steps; use an explicit pilot run")
    if args.command == "run" and args.backend == "tiny":
        raise ValueError("Random tiny fixture cannot run a scientific pilot")
    if args.command == "smoke" and not args.allow_uncalibrated:
        raise ValueError("Smoke latent paths require --allow-uncalibrated; no semantic result is implied")
    store = RunStore(Path(args.out))
    source = code_manifest(ROOT)
    store.write("source.json", source)
    store.write("environment.before.json", inventory(ROOT))
    started = time.monotonic()
    try:
        host, identity = make_host(args)
        store.write("checkpoint_identity.json", identity)
        store.write("host_identity.json", host.identity)
        parity = host.native_parity()
        store.write("native_parity.json", parity)
        if not parity["exact"]:
            raise RuntimeError("Native hook parity failed")
        arms = ["T", "L", "B"] if args.command == "smoke" else [args.arm]
        if args.command == "run" and args.arm in {"L", "B"}:
            if not args.qualification:
                raise ValueError("Latent pilot blocked until a bound semantic qualification exists")
            gate = verify_gate(Path(args.qualification), host_identity=host.identity, source_sha256=source["source_sha256"])
            store.write("qualification.json", gate)
        seeds = read_jsonl(args.seeds)
        seed = next((x for x in seeds if x["id"] == args.seed_id), None)
        if seed is None:
            raise ValueError("Unknown seed id")
        store.write("request.json", {"seed": seed, "arms": arms, "regime": args.regime, "steps": args.steps,
                                      "purpose": "engineering" if args.command == "smoke" else "pilot",
                                      "observer": not args.no_observer,
                                      "host_identity_sha256": digest(host.identity)})
        # Same deterministic E(s0) snapshot starts L and B. T starts from the same literal s0.
        latent0 = host.encode(seed["text"]) if any(a != "T" for a in arms) else None
        histories = {}
        for arm in arms:
            initial = State(text=seed["text"]) if arm == "T" else latent0.clone()
            store.save_state(f"states/{arm}_000", initial)
            def sink(row, state, activations, a=arm):
                store.append("transitions.jsonl", row)
                name = f"{a}_{row['step']:03d}"
                store.save_state("states/" + name, state)
                import numpy as np
                np.savez_compressed(store.path / "states" / (name + "_activations.npz"),
                                    **{k: v.float().cpu().numpy() for k,v in activations.items()})
            history, _ = run_trajectory(host, seed["text"], arm=arm, regime=args.regime, steps=args.steps,
                                        initial=initial, observer=not args.no_observer, sink=sink)
            histories[arm] = history
        # Tasks are loaded and presented only after the recurrence is finished.
        if args.tasks:
            tasks = read_jsonl(args.tasks)
            for arm, history in histories.items():
                # v0 CLI evaluates predeclared baseline and final state. Library supports intermediate checkpoints.
                for step_n in (0, args.steps):
                    for task in tasks[:args.task_limit]:
                        row = task_readout(host, history[step_n], task)
                        store.append("tasks.jsonl", {"arm": arm, "step": step_n, **row})
        if args.branch and "L" in histories:
            branches = branch_once(host, histories["L"][-1], continuation_steps=1, start_step=args.steps,
                                  regime=args.regime)
            for label, branch in branches.items():
                for n, s in enumerate(branch["states"]):
                    store.save_state(f"branches/{label}_{n:03d}", s)
                store.write(f"branches/{label}.json", {"readout": branch["readout"],
                                                      "state_sha256": [s.sha() for s in branch["states"]]})
        store.write("environment.after.json", inventory(ROOT))
        store.write("terminal.json", {"status": "ENGINEERING_PASS_NOT_SEMANTICS" if args.command == "smoke" else "PILOT_COMPLETE_NOT_INFERENCE",
                                      "elapsed_seconds": time.monotonic()-started, "compute": host.meter.snapshot(),
                                      "semantic_success": None, "reasoning_gain": None,
                                      "pretrained_family_tested": args.backend == "hf"})
        print(canonical({"output": str(store.path), "status": "complete", "scientific_success": None}))
        return 0
    except Exception as exc:
        store.write("terminal.json", {"status": "FAILED_NO_RETRY", "error_type": type(exc).__name__,
                                      "message": str(exc), "elapsed_seconds": time.monotonic()-started})
        print(f"{type(exc).__name__}: {exc}. Trial preserved at {store.path}", file=sys.stderr)
        return 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="self-fold")
    sub = p.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="Read-only resource inventory, no model loading")
    inv.add_argument("--out")
    matrix = sub.add_parser("plan", help="Describe the intended matrix; never dispatch jobs")
    matrix.add_argument("--seeds", default=str(ROOT/"data/seeds.dev.jsonl"))
    for name in ("smoke", "run"):
        a = sub.add_parser(name)
        a.add_argument("--backend", choices=["tiny", "hf"], default="tiny" if name == "smoke" else "hf")
        a.add_argument("--out", required=True)
        a.add_argument("--device", default="cpu")
        a.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
        a.add_argument("--profile")
        a.add_argument("--checkpoint")
        a.add_argument("--transport")
        a.add_argument("--reviewed-local-code", help="Explicit hash-bound approval for local custom model Python; never automatic")
        a.add_argument("--cut", type=int)
        a.add_argument("--writer", type=int)
        a.add_argument("--slots", type=int, default=8)
        a.add_argument("--steps", type=int, default=2)
        a.add_argument("--max-new-tokens", type=int, default=8 if name == "smoke" else 64)
        a.add_argument("--max-context", type=int, default=1536)
        a.add_argument("--wall-seconds", type=int, default=600)
        a.add_argument("--cpu-threads", type=int, default=1)
        a.add_argument("--arm", choices=["T", "L", "B"], default="T")
        a.add_argument("--regime", choices=["plain", "expand"], default="plain")
        a.add_argument("--seeds", default=str(ROOT/"data/seeds.dev.jsonl"))
        a.add_argument("--seed-id", default="caution_plus_p0_self")
        a.add_argument("--tasks")
        a.add_argument("--task-limit", type=int, default=4)
        a.add_argument("--branch", action="store_true")
        a.add_argument("--no-observer", action="store_true")
        a.add_argument("--qualification")
        a.add_argument("--allow-uncalibrated", action="store_true")
    args = p.parse_args(argv)
    if args.command == "inventory":
        data = inventory(ROOT)
        if args.out:
            q = Path(args.out)
            q.parent.mkdir(parents=True, exist_ok=True)
            with q.open("x") as f:
                f.write(canonical(data)+"\n")
        else:
            print(canonical(data))
        return 0
    if args.command == "plan":
        n = len(read_jsonl(args.seeds))
        print(canonical({"development_seeds": n, "arms": ["T","L","B"], "regimes": ["plain","expand"],
                         "trajectories_per_model": n*6, "steps": 12, "transitions_per_model": n*6*12,
                         "models_proposed": 4, "dispatch": False, "confirmation_dataset": "NOT_CREATED"}))
        return 0
    return execute(args)
