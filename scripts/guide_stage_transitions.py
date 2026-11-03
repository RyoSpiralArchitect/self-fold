"""Backpropagate the frozen native readout through both state maps at a fixed endpoint."""

import time
from pathlib import Path
import numpy as np
import torch

from conditional_curriculum_learning import weighted_native_loss
from fit_codec import parameter_sha
from fit_conditional_curriculum import parser
from fit_continuity_recipient import batch_inputs
from fit_stage_transitions import read, run, tensors
from selffold.conditional_curriculum import tasks_for
from selffold.recipient_learning import batched_recipient_logits, load_recipient
from selffold.records import RunStore, file_sha
from selffold.stage_transitions import StageMap


class Residual(torch.nn.Module):
    def __init__(self, base, gain, rng, device):
        super().__init__()
        self.base = StageMap(**{k: v.to(device) for k, v in vars(base).items()})
        n = len(base.center)
        self.down = torch.nn.Parameter(
            torch.randn(n, 8, generator=rng).to(device) / n**0.5
        )
        self.up = torch.nn.Parameter(torch.zeros(8, n, device=device))
        self.gain = gain

    def merged(self):
        b = self.base
        return StageMap(
            b.center,
            b.scale,
            b.target,
            torch.cat((b.down, self.down), 1),
            torch.cat((b.up, self.gain * self.up)),
        )

    def forward(self, z):
        return self.merged().slots(z)


def fit_guided(args, host, store, cases, bank, keys, groups, fixture):
    parent = args.maps
    if (
        read(parent / "source.json")["git_commit"]
        != "b8c4511b9334b3e4b5dd69235f2e757e4a595b06"
        or file_sha(parent / "maps.npz")
        != read(parent / "maps_closed.json")["artifact"]
    ):
        raise ValueError("Wrong feature-only maps")
    with np.load(parent / "maps.npz", allow_pickle=False) as f:
        base = {
            name: StageMap(
                **{
                    k: torch.from_numpy(f[name + "_" + k])
                    for k in StageMap.__annotations__
                }
            )
            for name in ("g1", "g2", "decoder")
        }
    x = tensors(bank, keys).to(host.device)
    y = tensors(bank, keys, "cases", 0).to(host.device)
    z = tensors(bank, keys, "rule", 0).to(host.device)
    scales = [
        float((a.float() - a.float().mean(0)).square().mean().clamp_min(1e-12))
        for a in (y, z)
    ]
    rng = torch.Generator().manual_seed(9107)
    first = Residual(base["g1"], scales[0] ** 0.5, rng, host.device)
    second = Residual(base["g2"], scales[1] ** 0.5, rng, host.device)
    params = [*first.parameters(), *second.parameters()]
    lookup = {key: i for i, key in enumerate(keys)}
    units = sorted({k[:3] for k in keys})
    schedule = [
        units[i]
        for _ in range(2)
        for i in torch.randperm(len(units), generator=rng).tolist()
    ]
    if len(schedule) != 192:
        raise ValueError("Expected two complete passes over 96 units")
    store.write("guide_schedule.json", schedule)
    opt = torch.optim.AdamW(params, lr=0.001, weight_decay=0)
    reader, _, _ = load_recipient(args.literal / "focus.npz", host)
    before = parameter_sha(reader)
    ids = [host.tokenizer.encode(s, add_special_tokens=False)[0] for s in ("A", "B")]
    try:
        with torch.enable_grad():
            for step, unit in enumerate(schedule):
                opt.zero_grad(set_to_none=True)
                total = 0.0
                edge = None
                norm_before = host.meter.snapshot()
                for pole in (0, 1):
                    for order in (0, 1):
                        k = (*unit, pole, "source", order)
                        i = lookup[k]
                        c = cases[k]
                        tasks = [
                            t
                            for w in range(4)
                            for t in tasks_for(c, groups[k[0]], fixture, w)
                        ]
                        for j in (0, 4):
                            batch = tasks[j : j + 4]
                            mid = first(x[i : i + 1])
                            final = second(mid)
                            target = torch.tensor(
                                [ids[0 if t["gold"] == "A" else 1] for t in batch],
                                device=host.device,
                            )
                            native = []
                            for state in (mid, final):
                                inputs = batch_inputs(
                                    host, batch, {c["key"]: state[0]}, "matched"
                                )
                                logits = batched_recipient_logits(host, reader, *inputs)
                                native.append(
                                    weighted_native_loss(
                                        logits,
                                        target,
                                        [t["world"] for t in batch],
                                        margin=2.0,
                                        label_ids=ids,
                                    )
                                )
                            if edge is None:
                                grad = torch.autograd.grad(
                                    native[1], mid, retain_graph=True
                                )[0]
                                edge = float(grad.float().norm())
                            reconstruction = (
                                mid.float() - y[i]
                            ).square().mean() / scales[0] + (
                                final.float() - z[i]
                            ).square().mean() / scales[
                                1
                            ]
                            loss = (
                                native[0] + native[1]
                            ) * 0.5 + 0.1 * reconstruction / 16
                            loss.backward()
                            total += float(loss.detach())
                norms = [
                    float(
                        torch.stack(
                            [p.grad.float().norm() for p in module.parameters()]
                        ).norm()
                    )
                    for module in (first, second)
                ]
                if (
                    edge is None
                    or not np.isfinite(edge)
                    or edge <= 0
                    or min(norms) <= 0
                    or any(
                        p.grad is not None
                        for p in [*host.model.parameters(), *reader.parameters()]
                    )
                ):
                    raise ValueError("Broken gradient boundary")
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                opt.step()
                after = host.meter.snapshot()
                store.append(
                    "guide_training.jsonl",
                    dict(
                        step=step + 1,
                        unit=unit,
                        loss=total,
                        mid_to_final_gradient=edge,
                        residual_gradients=norms,
                        forwards=after["forwards"] - norm_before["forwards"],
                        padded=after["processed_tokens"]
                        - norm_before["processed_tokens"],
                    ),
                )
    finally:
        unchanged = parameter_sha(reader) == before
        reader.close()
    if not unchanged:
        raise ValueError("Frozen reader changed")
    for name, module in (("g1", first), ("g2", second)):
        base[name] = StageMap(
            **{k: v.detach().cpu() for k, v in vars(module.merged()).items()}
        )
    store.write(
        "guide_closed.json",
        dict(
            parent_maps=file_sha(parent / "maps.npz"),
            reader_sha256=file_sha(args.literal / "focus.npz"),
            frozen_reader_parameters=before,
            updates=len(schedule),
            added_parameters=sum(p.numel() for p in params),
            stage_labels=192 * 32 * 2,
            fit_normalizers=scales,
            fresh_captures=0,
        ),
    )
    rule_keys = [k for k in bank if k[4] == "rule"]
    return base, tensors(bank, rule_keys).float().mean(0).to(x.dtype)


if __name__ == "__main__":
    p = parser()
    p.add_argument("--literal", type=Path, required=True)
    p.add_argument("--maps", type=Path, required=True)
    a = p.parse_args()
    store = RunStore(a.out)
    start = time.monotonic()
    try:
        r = run(a, store, fit_hook=fit_guided)
    except Exception as exc:
        store.write("terminal.json", dict(status="TECHNICAL_STOP", error=str(exc)))
        raise
    store.write("terminal.json", r | dict(elapsed_seconds=time.monotonic() - start))
