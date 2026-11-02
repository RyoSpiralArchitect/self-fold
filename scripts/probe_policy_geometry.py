"""Capture a source and its neutral rendering, then test fixed train centroid readouts."""
import argparse
import json
from pathlib import Path
import re
import time
import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from selffold.calibration import codec_examples
from selffold.carrier_probe import capture_description
from selffold.cli import ROOT, make_host
from selffold.records import code_manifest, digest, file_sha, RunStore

STUDY = ROOT / "provenance/policy_binding_20261103"
DATA = ROOT / "data/codec_preparation_20261102"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def trigger_action(ledger, family):
    def selected(world):
        return world["request"] and not world["override"] if family == "priority" else all(world.values())
    actions = [entry["choice"] for entry in ledger if selected(entry["world"])]
    if len(actions) != 1:
        raise ValueError("Expected one trigger world")
    return actions[0]


def source_panels():
    panels = {}
    for split in ("train", "dev"):
        cases = codec_examples(DATA, split)
        groups = {x["id"]: x for x in read_rows(DATA / (split + ".groups.jsonl"))}
        for x in cases:
            group, pole, variant = x["id"].rsplit("-", 2)
            g = groups[group]
            x.update(variant=variant, pole=pole, family=g["family"],
                     trigger=trigger_action(g["poles"][pole]["truth_ledger"], g["family"]))
        panels["original" if split == "dev" else split] = cases
    crossed = []
    for row in read_rows(STUDY / "order_controls.jsonl"):
        group, pole, variant = row["id"].rsplit("-", 2)
        family = group.rsplit("-", 1)[0]
        crossed.append({"id": row["id"], "source": row["source"], "group": group, "pole": pole, "variant": variant,
                        "family": family, "trigger": trigger_action(row["truth_ledger"], family)})
    panels["crossed"] = crossed
    if [len(panels[x]) for x in ("train", "original", "crossed")] != [192, 96, 96]:
        raise ValueError("Panel denominators differ from the plan")
    return panels


def run(args, store):
    seal = code_manifest(ROOT)
    store.write("source.json", seal)
    if seal["git_status"] != "" or not seal["git_commit"]:
        raise ValueError("Clean source required")
    store.write("request.json", {"plan_sha256": file_sha(STUDY/"GEOMETRY_PLAN.md"),
                "panel_sha256": file_sha(STUDY/"order_controls.jsonl"), "data_sha256": file_sha(DATA/"manifest.json"),
                "prior_terminal_sha256": file_sha(STUDY/"run/terminal.json"), "optimizer_updates": 0})
    panels = source_panels()
    host, identity = make_host(args)
    store.write("host.json", host.identity)
    store.write("native_parity.json", parity := host.native_parity())
    if (not parity["exact"] or identity["checkpoint_sha256"] != CHECKPOINT_SHA or
            host.identity != json.loads((STUDY/"run/host.json").read_text()) or any(p.requires_grad for p in host.model.parameters())):
        raise ValueError("Host changed")
    backbone = parameter_sha(host.model)
    old_donors = {x["case"]: x for x in read_rows(STUDY/"run/donors.jsonl")}
    cache, features, labels = {}, {}, {}
    for panel, cases in panels.items():
        for row in cases:
            name, text = row["id"], row["source"]
            words = re.findall(r"(?<=use the )(copper|silver)(?= tray)", text)
            if sorted(words) != ["copper", "silver"]:
                raise ValueError("Need one adjective for each tray")
            neutral = re.sub(r"(?<=use the )(copper|silver)(?= tray)", "metal", text)
            for value in (text, neutral):
                if value not in cache:
                    state, layout = capture_description(host, value, "workspace")
                    cache[value] = (state.slots.float().cpu().flatten(), state.sha(), layout)
            raw, raw_sha, a = cache[text]
            ref, ref_sha, b = cache[neutral]
            changes = [(x, y) for x, y in zip(a["native_ids"], b["native_ids"], strict=True) if x != y]
            if (a["capture_positions"] != b["capture_positions"] or sorted(x for x, _ in changes) != [8343, 17410]
                    or len({y for _, y in changes}) != 1 or host.tokenizer.decode([changes[0][1]]).strip() != "metal"):
                raise ValueError("Neutral replacement altered the capture geometry")
            if panel != "crossed" and (old_donors[name]["state_sha256"] != raw_sha or old_donors[name]["source_sha256"] != digest(text)):
                raise ValueError("Original donor replay mismatch")
            features[name] = {"raw": raw, "centered": raw - ref, "reference": ref}
            labels[name] = {"first_mention": words[0] == "copper", "trigger_action": row["trigger"] == "use the copper tray"}
            store.append("captures.jsonl", {"case": name, "panel": panel, "source_sha256": digest(text), "state_sha256": raw_sha,
                         "reference_text_sha256": digest(neutral), "reference_state_sha256": ref_sha,
                         "positions": a["capture_positions"], "replacement_token": changes[0][1]})
    flipped = set(json.loads((STUDY/"run/sham_sites.json").read_text()))
    previous = {(x["case"], x["sham"]): x["copper_margin"] for x in read_rows(STUDY/"run/raw_readout.jsonl")}
    max_error = 0.
    for feature in ("raw", "centered", "reference"):
        train = panels["train"]
        matrix = torch.stack([features[x["id"]][feature] for x in train])
        for label in ("first_mention", "trigger_action"):
            for sham in (False, True):
                chosen = torch.tensor([labels[x["id"]][label] ^ (sham and x["group"] in flipped) for x in train])
                if chosen.sum().item() != 96:
                    raise ValueError("Unbalanced training labels")
                a, b = matrix[chosen].mean(0), matrix[~chosen].mean(0)
                for panel in ("original", "crossed"):
                    for row in panels[panel]:
                        name = row["id"]
                        margin = torch.dot(features[name][feature] - (a+b)/2, a-b).item()
                        if not torch.isfinite(torch.tensor(margin)):
                            raise ValueError("Nonfinite margin")
                        if feature == "raw" and label == "first_mention" and panel == "original":
                            max_error = max(max_error, abs(margin - previous[name, sham]))
                            if max_error > 1e-5:
                                raise ValueError("Previously observed raw margins failed to replay")
                        store.append("scores.jsonl", {"case": name, "group": row["group"], "family": row["family"],
                            "pole": row["pole"], "variant": row["variant"], "panel": panel, "feature": feature,
                            "label": label, "sham": sham, "gold_copper": labels[name][label], "copper_margin": margin})
    if backbone != parameter_sha(host.model):
        raise ValueError("Frozen backbone changed")
    return {"status": "COMPLETED_SINGLE_SOURCE_GEOMETRY_DEV", "source_rows": len(features), "unique_captures": len(cache),
            "original_donor_hashes_replayed": len(old_donors), "prior_margin_max_error": max_error, "optimizer_updates": 0,
            "backbone_sha256": backbone, "compute": host.meter.snapshot(), "qualification_content_loaded": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(backend="hf", cpu_threads=1, device="cuda:0", dtype="bfloat16", slots=8, cut=None,
                   writer=None, max_context=1024, max_new_tokens=None, until_eos=True, transport=None, encoder_bridge=None,
                   reviewed_local_code=None, wall_seconds=600, profile=ROOT / "configs/models/gemma2b.yaml")
    args = p.parse_args()
    store, start = RunStore(args.out), time.monotonic()
    try:
        result = run(args, store)
    except Exception as exc:
        store.write("terminal.json", {"status": "INVALID_STOPPED", "error": str(exc), "type": type(exc).__name__,
                                      "elapsed_seconds": time.monotonic()-start})
        raise
    store.write("terminal.json", {**result, "elapsed_seconds": time.monotonic()-start})


if __name__ == "__main__":
    main()
