"""Run the four predeclared description-only carrier cells on two pinned checkpoints."""
import argparse
import json
from pathlib import Path
import time
from types import MethodType, SimpleNamespace
from selffold.cli import ROOT, make_host
from selffold.host import SlotHost
from selffold.carrier_probe import run_carrier_probe, description_layout, ROUTES
from selffold.records import RunStore, code_manifest, digest, file_sha
from selffold.resources import inventory

MODELS = {
    "mistral7b": "5d141a13ea9f5fce9c142bc26bdef64ea905fd9d27c64aff472f6875dd4296e9",
    "gemma2b": "f3503907cc4d2ace62abccf30393a05919827c1c8969b2f2755f15627b9756ad",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(backend="hf", cpu_threads=1, device="cuda:0", dtype="bfloat16", slots=8,
                   cut=None, writer=None, max_context=1024, max_new_tokens=None, until_eos=True,
                   wall_seconds=1800, transport=None, reviewed_local_code=None)
    args = p.parse_args()
    args.profile = ROOT / "configs/models" / f"{args.model}.yaml"
    fixture = ROOT / "data/carrier_sources.dev.json"
    store, start = RunStore(args.out), time.monotonic()
    source = code_manifest(ROOT)
    store.write("source.json", source)
    store.write("environment.before.json", inventory(ROOT))
    store.write("request.json", {"model": args.model, "checkpoint_sha256": MODELS[args.model],
                "fixture_sha256": file_sha(fixture), "split": "exposed_development",
                "donor_forward_length": 256, "routes": ["workspace", "native_tail"],
                "sham_seeds": [4711, 9292], "max_forwards": 20000,
                "max_processed_tokens": 20000000, "wall_seconds": 1800})
    try:
        if source["git_status"] != "" or not source["git_commit"]:
            raise ValueError("A clean source seal is required")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
        layout_host = SimpleNamespace(tokenizer=tokenizer, slots=8, max_context=1024,
                                      placeholder_id=tokenizer.encode(".", add_special_tokens=False)[-1])
        layout_host.render = MethodType(SlotHost.render, layout_host)
        groups = json.loads(fixture.read_text())["groups"]
        layouts = []
        for route in ROUTES:
            for group in groups:
                for pole in ("plus", "minus"):
                    for phrasing, text in enumerate(group[pole]):
                        _, _, layout = description_layout(layout_host, text, route)
                        layouts.append({"group": group["id"], "pole": pole, "phrasing": phrasing, **layout})
        store.write("layout_preflight.json", layouts)
        host, identity = make_host(args)
        store.write("checkpoint_identity.json", identity)
        store.write("host_identity.json", host.identity)
        if identity["checkpoint_sha256"] != MODELS[args.model]:
            raise ValueError("Unexpected checkpoint identity")
        parity = host.native_parity()
        store.write("native_parity.json", parity)
        if not parity["exact"]:
            raise ValueError("Native hook parity failed")
        run_carrier_probe(host, store, groups)
        store.write("environment.after.json", inventory(ROOT))
        terminal = {"status": "COMPLETED_DEVELOPMENT_CARRIER_DIAGNOSTIC", "compute": host.meter.snapshot(),
                    "host_sha256": digest(host.identity), "elapsed_seconds": time.monotonic() - start}
        store.write("terminal.json", terminal)
        print(json.dumps(terminal))
    except Exception as exc:
        store.write("terminal.json", {"status": "FAILED_NO_RETRY", "error_type": type(exc).__name__,
                    "message": str(exc), "elapsed_seconds": time.monotonic() - start})
        raise


if __name__ == "__main__":
    main()
