"""Join mixture-boundary trajectories (member-space weights) to native scores."""

import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def bits(mask):
    return (np.asarray(mask)[..., None] >> np.arange(4)) & 1


def expected(keys, masks, labels):
    polarity = np.array([k[3] for k in keys]).reshape(
        (-1,) + (1,) * (np.ndim(masks) + 2)
    )
    return labels[polarity ^ bits(masks)[..., None] ^ np.arange(2)]


def w_index(w_values, descriptor):
    """Map realized descriptors to scored-variant indices; exact within fp16."""
    w = np.asarray(w_values)
    if np.array_equal(w, [0.0, 1.0]):
        if not np.isin(descriptor, [0., 1.]).all():
            raise ValueError("Hard descriptors must be exactly zero or one")
        return descriptor.astype(np.int64)
    idx = np.argmin(np.abs(w[None] - descriptor.astype(np.float64).ravel()[:, None]), -1)
    idx = idx.reshape(descriptor.shape)
    if np.abs(w[idx] - descriptor).max() > 1e-2:
        raise ValueError("Descriptor outside scored variants")
    return idx


def select_top(top, widx, case_ids):
    """top: (cases, variants, orders) -> winner token per (…, worlds, orders)."""
    ids = np.asarray(case_ids).reshape(-1, *([1] * (widx.ndim - 1)))
    return top[ids, widx]


def native_labels(run):
    files = sorted(Path(run).glob("fit_native_*/native.json"))
    if not files:
        raise ValueError("No fitted native label binding")
    labels = np.asarray(read(files[0])["labels"])
    if labels.shape != (2,) or labels[0] == labels[1]:
        raise ValueError("Expected distinct A/B label IDs")
    if any(not np.array_equal(labels, read(path)["labels"]) for path in files[1:]):
        raise ValueError("Native label mappings disagree across forms")
    return labels


def align_twins(twins, semantics):
    if twins.shape[-3:] != (2, 2, 4):
        raise ValueError("Unexpected twin axes")
    if semantics == "carried_message_bits_v2":
        return twins
    if semantics != "legacy_sampled_writer":
        raise ValueError(f"Unknown rollout semantics: {semantics}")
    # Invert the old stack(-1).reshape(...,2,2,4), then move worlds last.
    unpacked = twins.reshape(*twins.shape[:-3], 4, 2, 2)
    return np.moveaxis(unpacked, -3, -1)


def summarize(run):
    run = Path(run)
    keys = read(run / "test_keys.json")["keys"]
    with np.load(run / "join.npz") as f:
        donors = f["donors"]
        eval_masks = f["eval_masks"].tolist()
    labels = native_labels(run)
    variants = {}
    for npz in run.glob("test_variants_*_b*/variants.npz"):
        tag = npz.parent.name  # test_variants_{form}_b{beta*100}
        form = tag.split("_")[2]
        beta = int(tag.split("_b")[1]) / 100
        with np.load(npz) as f:
            variants[(form, beta)] = (f["w"].tolist(), f["top"])
    fits = read(run / "fit_closed.json")["fits"]
    out = {}
    with np.load(run / "trajectories.npz") as f:
        paths = f["eval_paths"]
        path_masks = np.asarray(eval_masks)[paths]
        pbits = bits(path_masks)
        role = np.concatenate([np.zeros((len(paths), 1), int), pbits[..., 0]], -1)
        flips = np.diff(role, axis=-1)[:, :3]
        for fit in fits:
            name = f"{fit['arm']}_{fit['seed']}_"
            form, beta = fit["form"], fit["beta"]
            w_values, top = variants[(form, beta)]
            hits = []
            for d in range(4):
                widx = w_index(w_values, f[name + "depths"][:, :, d])
                hits.append(
                    select_top(top, widx, donors[0])
                    == expected(keys, path_masks[:, d], labels)
                )
            hits = np.stack(hits, 2)  # (n, strings, depth, world, order)
            per_depth = [[int(hits[:, :, d].sum()), int(hits[:, :, d].size)] for d in range(4)]
            complete = hits.all((2, 3, 4))
            nflips = (flips != 0).sum(-1)
            by_flips = {
                str(k): [int(complete[:, nflips == k].sum()), int((nflips == k).sum() * len(keys))]
                for k in range(4)
            }
            opposed = {}
            for label, table, dep in (
                ("depth2", f[name + "second"], path_masks[:, 1]),
                ("depth4", f[name + "fourth"], path_masks[:, 3]),
            ):
                truth = expected(keys, dep, labels)
                hit = select_top(top, w_index(w_values, table[:, 0]), donors[0]) == truth
                opp = select_top(top, w_index(w_values, table[:, 1]), donors[1]) == truth[..., ::-1]
                erased = select_top(top, w_index(w_values, table[:, 3]), donors[0]) == truth
                opposed[label] = dict(
                    matched=[int(hit.sum()), int(hit.size)],
                    opposed_correct=[int((hit & opp).sum()), int(hit.size)],
                    erased_intact=[int(erased.sum()), int(erased.size)],
                )
            semantics_key = name + "written_semantics"
            semantics = str(f[semantics_key].item()) if semantics_key in f else "legacy_sampled_writer"
            twins = align_twins(f[name + "twins"], semantics)
            tgot = select_top(top, w_index(w_values, twins), np.arange(len(keys)))
            # expected: (n, masks, worlds, orders) -> expand reps axis.
            base = expected(keys, np.asarray(eval_masks), labels)
            truth0 = np.broadcast_to(base[:, None], (len(keys), twins.shape[1], len(eval_masks), 4, 2))
            twin_out = {}
            for ci, cond in enumerate(("intact", "erased")):
                hit0 = tgot[:, :, :, 0, ci] == truth0
                hit1 = tgot[:, :, :, 1, ci] == truth0[..., ::-1]
                twin_out[cond] = [int((hit0 & hit1).sum()), int(hit0.size)]
            written = f[name + "written"].astype(int)
            convention = pbits[:, :3, 0]
            agree = {}
            for w in range(written.shape[-1]):
                agree[f"bit{w}"] = float((written[..., w] == convention[None]).mean())
            out.setdefault(fit["arm"], {})[str(fit["seed"])] = dict(
                native_per_depth=per_depth,
                all_four_correct=[int(complete.sum()), int(complete.size)],
                all_four_by_role_exchanges=by_flips,
                opposed=opposed,
                twins_both_histories=twin_out,
                rollout_semantics=semantics,
                legacy_twin_axis_repair_applied=semantics == "legacy_sampled_writer",
                **{("written_matches_offoff_bit" if semantics == "carried_message_bits_v2"
                    else "legacy_sampled_writer_matches_offoff_bit"): agree},
            )
    return dict(schema="mixture_boundary_summary_v2", eval_masks=eval_masks,
                strings=int(len(paths)), sources=len(keys), arms=out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    print(json.dumps(summarize(args.run), sort_keys=True))


if __name__ == "__main__":
    main()
