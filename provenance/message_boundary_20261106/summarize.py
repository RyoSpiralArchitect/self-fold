"""Join frozen boundary trajectories to native scores; count development endpoints.

Endpoint questions (docs/G_MESSAGE_BOUNDARY_PLAN.md):
- does dropping the mixture (hard) without a message lose what soft rows carried?
- does a learned message over the symbolic residue recover it?
- what did the learned message come to encode?
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "data/message_boundary_20261106.json").read_text())


def read(path):
    return json.loads(path.read_text())


def bits(mask):
    return (np.asarray(mask)[..., None] >> np.arange(4)) & 1


def expected(keys, masks, labels):
    """Gold A/B token id per (source, mask, world, order)."""
    polarity = np.array([k[3] for k in keys]).reshape(
        (-1,) + (1,) * (np.ndim(masks) + 2)
    )
    return labels[polarity ^ bits(masks)[..., None] ^ np.arange(2)]


def select(matrix, rows, case_ids):
    """matrix: (cases, members, orders) -> values at rows (…, worlds), all orders."""
    if rows.dtype != np.uint8 or np.any(rows > 1):
        raise ValueError("Invalid stored row address")
    indices = np.asarray(case_ids).reshape(-1, *([1] * (rows.ndim - 1)))
    return matrix[indices, rows]


def load_scores(run, form, restore):
    with np.load(run.parent / f"decide_{form}" / "scores.npz") as f:
        dispatch, top, ab = f["dispatch"], f["top"], f["ab"]
    margin = ab[:, 0] - ab[:, 1]
    # Packet order -> authored source order.
    return top[dispatch][restore], margin[dispatch][restore]


def summarize(run):
    run = Path(run)
    keys = read(run / "test_keys.json")["keys"]
    with np.load(run / "join.npz") as f:
        restore = {k: f[k] for k in ("restore_soft", "restore_hard")}
        donors = f["donors"]
        eval_masks = f["eval_masks"].tolist()
    labels = np.array(read(run.parent / "decide_soft" / "native.json")["labels"])
    scores = {
        form: load_scores(run, form, restore[f"restore_{form}"])
        for form in ("soft", "hard")
    }
    fits = read(run / "fit_closed.json")["fits"]
    out = {}
    with np.load(run / "trajectories.npz") as f:
        paths = f["eval_paths"]
        path_masks = np.asarray(eval_masks)[paths]  # (strings, 4)
        pbits = bits(path_masks)  # off/off bit per consumed mask
        # Role exchange count at the off/off row over the first three edits;
        # the initial layout always starts with the default there.
        role = np.concatenate(
            [np.zeros((len(paths), 1), int), pbits[..., 0]], -1
        )
        prefix_parity = (np.diff(role, axis=-1)[:, :3] != 0).sum(-1) % 2
        for fit in fits:
            name = f"{fit['arm']}_{fit['seed']}_"
            form = fit["form"]
            top, margin = scores[form]
            depths = f[name + "depths"]  # (sources, strings, 4 edits, 4 worlds)
            hits = []
            for d in range(4):
                got = select(top, depths[:, :, d], donors[0])
                truth = expected(keys, path_masks[:, d], labels)
                hits.append(got == truth)
            hits = np.stack(hits, 2)  # (n, strings, depth, world, order)
            per_depth = [
                [int(hits[:, :, d].sum()), int(hits[:, :, d].size)] for d in range(4)
            ]
            complete = hits.all((2, 3, 4))  # (sources, strings)
            by_parity = {}
            for p in (0, 1):
                sel = prefix_parity == p
                by_parity[str(p)] = [
                    int(complete[:, sel].sum()),
                    int(sel.sum() * len(keys)),
                ]
            # Opposed-donor reversal at depth two and four.
            opposed = {}
            for label, table, depth in (
                ("depth2", f[name + "second"], path_masks[:, 1]),
                ("depth4", f[name + "fourth"], path_masks[:, 3]),
            ):
                truth = expected(keys, depth, labels)
                hit = select(top, table[:, 0], donors[0]) == truth
                opp = select(top, table[:, 1], donors[1]) == truth[..., ::-1]
                signed = (
                    select(margin, table[:, 0], donors[0])
                    * select(margin, table[:, 1], donors[1])
                    < 0
                )
                erased = select(top, table[:, 3], donors[0]) == truth
                opposed[label] = dict(
                    matched=[int(hit.sum()), int(hit.size)],
                    both_opposed_reversed=[int((hit & opp & signed).sum()), int(hit.size)],
                    erased_intact=[int(erased.sum()), int(erased.size)],
                )
            # Twins: identical boundary table, opposite required answers.
            twins = f[name + "twins"]  # (n, reps, masks, history, intact/erased, worlds)
            n, reps, nm, hists, conds, worlds = twins.shape
            base_truth = expected(keys, np.asarray(eval_masks), labels)
            # current-mask axis of twins enumerates eval_masks; expand per rep.
            truth0 = np.broadcast_to(
                base_truth[:, None], (n, reps, nm, worlds, 2)
            )
            case_ids = np.arange(n)
            twin_out = {}
            for ci, cond in enumerate(("intact", "erased")):
                got0 = select(top, twins[:, :, :, 0, ci], case_ids)
                got1 = select(top, twins[:, :, :, 1, ci], case_ids)
                m0 = select(margin, twins[:, :, :, 0, ci], case_ids)
                m1 = select(margin, twins[:, :, :, 1, ci], case_ids)
                hit0 = got0 == truth0
                hit1 = got1 == truth0[..., ::-1]
                both = hit0 & hit1 & (m0 * m1 < 0)
                twin_out[cond] = [int(both.sum()), int(both.size)]
            written = f[name + "written"].astype(int)
            convention = pbits[:, :3, 0]  # off/off bit of the mask at each boundary
            agree = {}
            for w in range(written.shape[-1]):
                agree[f"bit{w}"] = float(
                    (written[..., w] == convention[None]).mean()
                )
            arm = out.setdefault(fit["arm"], {})
            arm[str(fit["seed"])] = dict(
                native_per_depth=per_depth,
                all_four_correct=[int(complete.sum()), int(complete.size)],
                all_four_by_prefix_parity=by_parity,
                opposed=opposed,
                twins_both_histories=twin_out,
                written_matches_offoff_bit=agree,
            )
    return dict(
        schema="message_boundary_summary_v1",
        eval_masks=eval_masks,
        strings=int(len(paths)),
        sources=len(keys),
        arms=out,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    print(json.dumps(summarize(args.run), sort_keys=True))


if __name__ == "__main__":
    main()
