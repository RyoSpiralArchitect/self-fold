"""Paired lexical supervision from the two public policy poles."""
import random
import torch


def first_divergence(left, right):
    """Both teachers must have the same geometry, including their native end token."""
    if len(left) != len(right) or not left or left[-1] != right[-1]:
        raise ValueError("Opposed targets need equal length and a common ending")
    for i, (a, b) in enumerate(zip(left, right, strict=True)):
        if a != b:
            return i
    raise ValueError("Opposed targets are identical")


def prepare_pairs(rows, tokenizer, end_id):
    cases = {}
    paired = {}
    for row in rows:
        group, pole, wording = row["id"].rsplit("-", 2)
        if group != row["group"] or pole not in {"plus", "minus"} or wording not in {"p0", "p1"}:
            raise ValueError("Unexpected case identifier")
        if row["id"] in cases:
            raise ValueError("Duplicate source")
        ids = tokenizer.encode(row["source"], add_special_tokens=False) + [end_id]
        cases[row["id"]] = {**row, "ids": ids, "pole": pole, "wording": wording}
        paired.setdefault((group, wording), {})[pole] = row["id"]
    pairs = []
    for key in sorted(paired):
        pair = [paired[key][p] for p in ("plus", "minus")]
        left, right = (cases[name]["ids"] for name in pair)
        i = first_divergence(left, right)
        words = [tokenizer.decode([ids[i]]).strip() for ids in (left, right)]
        if set(words) != {"copper", "silver"}:
            raise ValueError(f"First differing tokens must name the two trays: {words}")
        for name, word, foil in zip(pair, words, (right[i], left[i]), strict=True):
            cases[name].update(index=i, foil=foil, word=word)
        pairs.append(pair)
    return cases, pairs


def pair_schedule(pairs, *, steps=64):
    rng = torch.Generator().manual_seed(47)
    indices = []
    while len(indices) < 4 * steps:
        indices.extend(torch.randperm(len(pairs), generator=rng).tolist())
    return [[case for i in indices[4*s:4*s+4] for case in pairs[i]] for s in range(steps)]


def raw_readout(train, dev, slots):
    """Signed nearest-centroid scores for copper; fit with train labels only."""
    groups = sorted({row["group"] for row in train.values()})
    families = sorted({g.rsplit("-", 1)[0] for g in groups})
    rng = random.Random(237)
    flipped = set()
    for family in families:
        sites = [g for g in groups if g.rsplit("-", 1)[0] == family]
        if len(sites) != 12:
            raise ValueError("Expected twelve sites per family")
        flipped.update(rng.sample(sites, 6))
    matrix = torch.stack([slots[k].float().cpu().flatten() for k in train])
    outputs = []
    for sham in (False, True):
        copper = torch.tensor([(v["word"] == "copper") ^ (sham and v["group"] in flipped)
                               for v in train.values()], dtype=torch.bool)
        a, b = matrix[copper].mean(0), matrix[~copper].mean(0)
        for k, row in dev.items():
            x = slots[k].float().cpu().flatten()
            # Half the difference of the squared distances, positive toward copper.
            margin = torch.dot(x - (a + b) / 2, a - b).item()
            outputs.append({"case": k, "group": row["group"], "word": row["word"],
                            "sham": sham, "copper_margin": margin})
    return outputs, sorted(flipped)


def world_question(group, pole, world_index, order):
    entry = group["poles"][pole]["truth_ledger"][world_index]
    condition = ", ".join(f"{k} is {'on' if v else 'off'}" for k, v in entry["world"].items()) or "the policy applies"
    first, second = group["options"][::1 if order == 0 else -1]
    question = f"Under the current policy, when {condition}, which action applies?\nA = {first}. B = {second}."
    return question, "A" if entry["choice"] == first else "B"
