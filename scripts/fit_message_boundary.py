"""Train boundary-channel controllers and freeze them before new source captures.

Development arm of docs/G_MESSAGE_BOUNDARY_PLAN.md. Pinned to the same frozen
source compiler and native row reader as role_memory_20261104; only the
boundary channel between edits changes.
"""

from copy import deepcopy
import hashlib
import json

import numpy as np
import torch

from fit_codec import CHECKPOINT_SHA, parameter_sha
from fit_conditional_curriculum import main
from fit_role_memory import bank
from role_memory_native import PARENT, READER, score_rows
from selffold.cli import ROOT, make_host
from selffold.message_boundary import (
    BoundaryMemory,
    harden,
    rollout_boundary,
    train_batch,
    twins_boundary,
)
from selffold.records import RunStore, code_manifest, digest, file_sha
from selffold.role_trials import donor_indices
from selffold.state_only import ClauseReader

DIR = ROOT / "provenance/message_boundary_20261106"
DATA = ROOT / "data/message_boundary_20261106.json"
OLD = ROOT / "data/state_only_20261104.json"
ACTIONS = ROOT / "data/within_topic_wording_20261103.json"


def arm_name(arm):
    return f"{arm['form']}_{arm['channel']}_w{arm['width']}"


def fit_arm(arm, seed, rows, gold, credit, cfg, store):
    actor = BoundaryMemory(seed, arm["width"])
    rng = torch.Generator().manual_seed(1000 + seed)
    start = parameter_sha(actor)
    opt = torch.optim.Adam(actor.parameters(), lr=cfg["learning_rate"])
    masks = torch.tensor(cfg["fit_masks"])
    schedule = hashlib.sha256()
    initial = rows[:, [0, 0, 0, 1]]
    for step in range(cfg["updates"]):
        ix = torch.randint(len(rows), (cfg["batch"],), generator=rng)
        pattern = masks[torch.randint(len(masks), (len(ix), 2), generator=rng)]
        noise = torch.rand(len(ix), 2, 2 + arm["width"], generator=rng)
        for v in (ix, pattern, noise):
            schedule.update(v.numpy().tobytes())
        # The reward channel receives no correct action indices or target bits in its active loss.
        labels = gold[ix] if arm["channel"] == "supervised" else torch.zeros_like(gold[ix])
        loss, observed = train_batch(
            actor,
            initial[ix],
            labels,
            credit[ix],
            pattern,
            noise,
            arm["form"],
            arm["channel"],
            cfg["exploration_floor"],
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        store.append(
            "training.jsonl",
            dict(
                arm=arm_name(arm),
                seed=seed,
                step=step + 1,
                loss=float(loss.detach()),
                first_reward=observed[0],
                second_reward=observed[1],
            ),
        )
    return actor, dict(
        arm=arm_name(arm),
        seed=seed,
        start=start,
        finish=parameter_sha(actor),
        schedule=schedule.hexdigest(),
    )


def run(args, store):
    cfg, design, actions = [json.loads(p.read_text()) for p in (DATA, OLD, ACTIONS)]
    seal = code_manifest(ROOT)
    if (
        seal["git_status"]
        or not seal["git_commit"]
        or file_sha(PARENT / "compiler.npz") != cfg["source_compiler_sha256"]
        or file_sha(READER) != cfg["native_reader_sha256"]
    ):
        raise ValueError("Clean source and bound parent required")
    store.write("source.json", seal)
    store.write(
        "request.json",
        dict(
            design=file_sha(DATA),
            parent_design=file_sha(OLD),
            actions=file_sha(ACTIONS),
            plan=file_sha(DIR / "PLAN.md"),
            boundary_plan=file_sha(ROOT / "docs/G_MESSAGE_BOUNDARY_PLAN.md"),
        ),
    )
    host, identity = make_host(args)
    if (
        identity["checkpoint_sha256"] != CHECKPOINT_SHA
        or not host.native_parity()["exact"]
    ):
        raise ValueError("Pinned native host required")
    before = parameter_sha(host.model)
    store.write("host.json", host.identity)
    groups = actions["groups"]
    menus = [g["options"] for g in groups]
    compiler = ClauseReader(host.width * 2, 16).requires_grad_(False)
    with np.load(PARENT / "compiler.npz", allow_pickle=False) as f:
        compiler.load_state_dict({k: torch.from_numpy(f[k]) for k in f.files})
    keys, hashes, fitrows = bank(host, compiler, design, groups, cfg, "fit")
    store.write("fit_keys.json", dict(keys=keys, source_sha256=hashes))
    np.savez_compressed(store.path / "fit_rows.npz", rows=fitrows.numpy())
    fithard = harden(fitrows)
    if (fithard[:, 0] == fithard[:, 1]).all(-1).any():
        raise ValueError("A fit policy's two named actions collide after the symbolic boundary")
    gold = torch.tensor([[2 * k[0] + k[3], 2 * k[0] + 1 - k[3]] for k in keys])
    displayed = (gold.numpy() % 2)[:, :, None] ^ np.arange(2)
    credits = {}
    for form, rowset in (("soft", fitrows), ("hard", fithard)):
        native = RunStore(store.path / f"fit_native_{form}")
        top, letters = score_rows(
            host, menus, rowset.contiguous(), [k[0] for k in keys], native
        )
        score = (top[:, :, None, :] == letters[displayed][:, None, :, :]).mean(-1)
        credits[form] = torch.from_numpy(score.astype(np.float32))
    params = {}
    fits = []
    for seed in cfg["seeds"]:
        for arm in cfg["arms"]:
            actor, receipt = fit_arm(
                arm, seed, fitrows, gold, credits[arm["form"]], cfg, store
            )
            fits.append(receipt | dict(arm))
            for k, v in actor.state_dict().items():
                params[f"{arm_name(arm)}_{seed}_{k}"] = v.numpy()
    np.savez_compressed(store.path / "fits.npz", **params)
    store.write(
        "fit_closed.json",
        dict(
            fits=fits,
            weights=file_sha(store.path / "fits.npz"),
            fresh_captures=0,
            training_episodes_per_arm_seed=cfg["updates"] * cfg["batch"],
        ),
    )
    keys, hashes, rows = bank(host, compiler, design, groups, cfg, "primary")
    store.write("test_keys.json", dict(keys=keys, source_sha256=hashes))
    hard = harden(rows)
    if (hard[:, 0] == hard[:, 1]).all(-1).any():
        raise ValueError("A test policy's two named actions collide after the symbolic boundary")
    donors = donor_indices(keys)
    eval_masks = cfg["fit_masks"] + cfg["test_masks"]
    packets = {}
    for form, rowset in (("soft", rows), ("hard", hard)):
        packet = RunStore(store.path / f"packet_{form}")
        # Native recipient sees only opaque numerical rows and the menu, with no source key or choice order hint.
        perm = np.random.default_rng(151).permutation(len(rowset))
        np.savez_compressed(
            packet.path / "rows.npz",
            rows=rowset[perm].numpy(),
            menu=np.array([k[0] for k in keys], np.int32)[perm],
        )
        packet.write("menus.json", menus)
        packet.write(
            "binding.json",
            dict(
                host=digest(host.identity),
                **{p.name: file_sha(p) for p in sorted(packet.path.iterdir())},
            ),
        )
        packets[form] = perm
    np.savez_compressed(
        store.path / "join.npz",
        restore_soft=np.argsort(packets["soft"]),
        restore_hard=np.argsort(packets["hard"]),
        donors=donors,
        eval_masks=np.array(eval_masks, np.int32),
    )
    output = {}
    with np.load(store.path / "fits.npz", allow_pickle=False) as f:
        for fit in fits:
            seed = fit["seed"]
            name = fit["arm"]
            arm = next(a for a in cfg["arms"] if arm_name(a) == name)
            actor = BoundaryMemory(seed, arm["width"]).requires_grad_(False)
            prefix = f"{name}_{seed}_"
            actor.load_state_dict(
                {k: torch.from_numpy(f[prefix + k]) for k in actor.state_dict()}
            )
            if parameter_sha(actor) != fit["finish"]:
                raise ValueError("Fitted endpoint replay failed")
            rolled = rollout_boundary(
                actor, arm["form"], arm["channel"], rows, keys, eval_masks
            )
            if "eval_paths" not in output:
                output["eval_paths"] = rolled.pop("paths")
            else:
                rolled.pop("paths")
            twin = twins_boundary(actor, arm["form"], arm["channel"], rows, eval_masks)
            rolled["twins"] = twin["twins"]
            rolled["twins_written"] = twin["written"]
            output.update({prefix + k: v for k, v in rolled.items()})
    np.savez_compressed(store.path / "trajectories.npz", **output)
    store.write(
        "rollout_closed.json",
        dict(
            paths=file_sha(store.path / "trajectories.npz"),
            scored_fresh_native=0,
            final_choices="exact IDs of copied rows under each arm's carried form",
        ),
    )
    if parameter_sha(host.model) != before or any(
        p.grad is not None for p in host.model.parameters()
    ):
        raise ValueError("Backbone mutated")
    return dict(
        status="BOUNDARY_CONTROLLERS_CLOSED",
        fit_sources=len(fitrows),
        eval_sources=len(rows),
        arms=[arm_name(a) for a in cfg["arms"]],
        backbone=before,
        qualification_loaded=False,
        compute=host.meter.snapshot(),
    )


if __name__ == "__main__":
    main(run)
