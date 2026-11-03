"""Finite role assignments communicated through a discrete message."""
from itertools import combinations, permutations, product

import numpy as np
import torch

ASSIGNMENTS = np.array(list(permutations(range(3))), dtype=np.int64)
PAIRS = tuple(combinations(range(3), 2))


def triplet(topic):
    return (2 * topic, 2 * ((topic + 1) % 8) + 1, 2 * ((topic + 3) % 8))


def native_rewards(winners, labels):
    """Return success per table, target, emitted row, competing action and label order."""
    results = np.empty((len(winners), 3, 3, 2, 2), dtype=bool)
    for target in range(3):
        foils = [a for a in range(3) if a != target]
        for f, rival in enumerate(foils):
            pair = tuple(sorted((target, rival)))
            index = PAIRS.index(pair)
            for order in range(2):
                gold = labels[pair.index(target) ^ order]
                results[:, target, :, f, order] = winners[:, :, index, order] == gold
    return results


def partitions(length, symbols):
    """Restricted growth strings enumerate each unlabelled partition once."""
    def extend(prefix):
        if len(prefix) == length:
            yield tuple(prefix)
            return
        for k in range(min(symbols, max(prefix) + 2)):
            yield from extend([*prefix, k])
    yield from extend([0])


def exact_code(rewards, bits):
    """Fit integer codes using only externally graded action rewards on fitting menus."""
    symbols = 2 ** bits
    best_block = {}
    for size in range(1, 7):
        for block in combinations(range(6), size):
            best = None
            for choice in product(range(3), repeat=3):
                correct = rewards[:, ASSIGNMENTS[list(block)], np.asarray(choice)]
                score = (int(correct.sum()), int(correct.all(axis=(2, 3, 4)).sum()))
                if best is None or score > best[0]:
                    best = score, choice
            best_block[block] = best
    best = None
    for words in partitions(6, symbols):
        blocks = [tuple(i for i, w in enumerate(words) if w == code)
                  for code in range(max(words) + 1)]
        scores = [best_block[b][0] for b in blocks]
        value = tuple(map(sum, zip(*scores, strict=True)))
        if best is None or value > best[0]:
            decoder = np.zeros((symbols, 3), dtype=np.int64)
            for code, block in enumerate(blocks):
                decoder[code] = best_block[block][1]
            best = value, np.array(words), decoder
    return best[1], best[2]


def learn_code(rewards, bits, seed):
    """Differentiate the exact expected return, then freeze a hard writer and row reader."""
    symbols = 2 ** bits
    rng = torch.Generator().manual_seed(seed)
    writer = torch.nn.Parameter(0.2 * torch.randn(6, symbols, generator=rng))
    reader = torch.nn.Parameter(0.2 * torch.randn(symbols, 3, 3, generator=rng))
    returns = torch.from_numpy(rewards.mean(axis=(0, 3, 4)).astype(np.float32))[ASSIGNMENTS]
    optimizer = torch.optim.Adam([writer, reader], lr=0.05)
    trace = []
    for step in range(512):
        optimizer.zero_grad(set_to_none=True)
        expected = torch.einsum("hm,mra,hra->", writer.softmax(-1), reader.softmax(-1), returns) / 18
        (-expected).backward()
        optimizer.step()
        if step in (0, 127, 255, 511):
            trace.append([step + 1, expected.item()])
    return (writer.detach().numpy(), reader.detach().numpy()), trace


def evaluate(words, decoder, rewards, erase=False):
    messages = np.zeros_like(words) if erase else words
    addresses = decoder[messages]
    right = rewards[:, ASSIGNMENTS, addresses]
    complete = right.all(axis=(2, 3, 4))
    return dict(codes=words.tolist(), used=len(set(map(int, messages))),
                resolved_addresses=addresses.tolist(),
                address_decisions=[int((addresses == ASSIGNMENTS).sum()), 18],
                address_histories=[int((addresses == ASSIGNMENTS).all(-1).sum()), 6],
                native_decisions=[int(right.sum()), int(right.size)],
                native_histories=[int(complete.sum()), int(complete.size)],
                histories_per_table=complete.sum(-1).tolist())
