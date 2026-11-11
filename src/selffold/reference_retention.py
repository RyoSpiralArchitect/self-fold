"""Reward-trained binary reference cells beside an already frozen report reader."""
from __future__ import annotations

import itertools

import numpy as np
import torch
from torch import nn

from .mixture_boundary import eval_step, init_weights, realize
from .role_memory import advantage
from .role_reference import attach, intervene
from .roundtrip import pattern_of


class RetentionCell(nn.Module):
    """2 or 34 fitted scalars; no hardcoded identity initialization or target bit."""
    def __init__(self, seed, report_conditioned=False):
        super().__init__()
        rng = torch.Generator().manual_seed(10000 + seed)
        self.reference = nn.Parameter(torch.randn(2, generator=rng) * .05)
        self.report = (nn.Parameter(torch.randn(32, generator=rng) * .05)
                       if report_conditioned else None)

    def forward(self, features, incoming, uniform=None, floor=.12):
        if incoming.ndim != 1 or not bool(((incoming == 0) | (incoming == 1)).all()):
            raise ValueError('The incoming reference is one binary state')
        logits = self.reference[0] * (2 * incoming.float() - 1) + self.reference[1]
        if self.report is not None:
            logits = logits + features @ self.report
        if uniform is None:
            return (logits > 0).long(), None
        probability = floor / 2 + (1-floor) * logits.sigmoid()
        outgoing = (uniform < probability).long()
        logp = torch.where(outgoing.bool(), probability, 1-probability).log()
        return outgoing, logp


def report_at(bank, cases, prior, mask, after):
    features = bank[cases, prior, mask, after]
    if torch.isnan(features).any():
        raise ValueError('Report lookup lies outside the bound feasible corpus')
    return features


def training_loss(cell, reader, members, reference, cases, masks, uniform, bank, reward, floor=.12):
    """Only the first memory write gets credit from the second native judgment."""
    weights = init_weights(len(members), 1.)
    table = realize(weights, members)
    prior = pattern_of(weights)
    table, weights, _, _ = eval_step(reader, table, weights,
        reader.initial_message(len(members)), masks[:, 0], 'hard', 'lmtext', members)
    features = report_at(bank, cases, prior, masks[:, 0], pattern_of(weights))
    next_reference, logp = cell(features, reference, uniform, floor)
    _, after, _, _ = eval_step(reader, table, weights, attach(features, next_reference),
                               masks[:, 1], 'hard', 'lmtext', members)
    credit = reward(after.argmax(-1), masks[:, 1]).mean((-1, -2))
    # No reference labels or report-target losses enter optimization.
    return -(logp * advantage(credit)).mean(), float(credit.mean())


@torch.no_grad()
def rollout(reader, cell, members, reference, masks, bank, mode='intact'):
    """Every next reference is produced by the cell from the incoming payload."""
    n = len(members)
    paths = torch.tensor(list(itertools.product(range(len(masks)), repeat=4)))
    count = len(paths)
    instructions = torch.tensor(masks)[paths].repeat(n, 1)
    mem = members.repeat_interleave(count, 0)
    cases = torch.arange(n).repeat_interleave(count)
    weights = init_weights(n*count, 1.)
    table = realize(weights, mem)
    incoming = reader.initial_message(n*count)
    # The sole source-reference read; the loop subsequently uses the payload.
    initial = reference.repeat_interleave(count)
    depths, refs, copy_hits = [], [], []
    for t in range(4):
        prior = pattern_of(weights)
        carried = initial if t == 0 else incoming[:, -1].long()
        table, weights, _, _ = eval_step(reader, table, weights, incoming,
            instructions[:, t], 'hard', 'lmtext', mem)
        depths.append(pattern_of(weights).reshape(n, count).numpy().astype(np.uint8))
        if t == 3:
            continue
        features = report_at(bank, cases, prior, instructions[:, t], pattern_of(weights))
        outgoing = carried.clone() if cell is None else cell(features, carried)[0]
        copy_hits.append((outgoing == carried).reshape(n, count).numpy())
        if t == 0:
            outgoing = intervene(outgoing, mode)
        incoming = attach(features, outgoing)
        refs.append(outgoing.reshape(n, count).numpy().astype(np.uint8))
    return dict(paths=paths.numpy().astype(np.uint8), depths=np.stack(depths, -1),
        references=np.stack(refs, -1), copy_hits=np.stack(copy_hits, -1),
        initial_reference=reference.numpy().astype(np.uint8))


def summarize(data, keys, cfg, top, labels):
    instructions = np.asarray(cfg['fit_masks'] + cfg['test_masks'])[data['paths']]
    n, paths = len(keys), len(instructions)
    pole = np.asarray([k[3] for k in keys])
    complete = np.ones((n, paths), bool)
    hits = []
    for t in range(4):
        members = (data['depths'][:, :, t, None] >> np.arange(4)) & 1
        got = top[np.arange(n)[:, None, None], members]
        target = labels[pole[:, None, None, None]
            ^ ((instructions[:, t, None] >> np.arange(4)) & 1)[None, ..., None] ^ np.arange(2)]
        hit = (got == target).all((-1, -2))
        hits.append(hit)
        complete &= hit
    novel = np.isin(instructions, cfg['test_masks']).all(-1)
    first_ids = np.unique(instructions[:, 0], return_index=True)[1]
    def count(a):
        return [int(a.sum()), int(a.size)]
    result = {}
    for name, ids in (
        ('all_menus', list(range(n))),
        ('primary_fit_menus', [i for i,k in enumerate(keys) if k[0] in cfg['fit_topics']]),
        ('unfitted_menus', [i for i,k in enumerate(keys) if k[0] not in cfg['fit_topics']])):
        result[name] = dict(all_four_correct=count(complete[ids]),
            novel_masks=count(complete[ids][:, novel]),
            first_complete=count(hits[0][ids][:, first_ids]),
            complete_per_depth=[count(h[ids]) for h in hits],
            reference_matches_initial=count(data['references'][ids] == data['initial_reference'][ids,None,None]),
            cell_copies_incoming=count(data['copy_hits'][ids]))
    return result


@torch.no_grad()
def audit_cell(cell, features, topics, fit_topics):
    """Counterfactual incoming 0 and 1 for each record; this is retention, not semantics."""
    predictions = torch.stack([cell(features, torch.full((len(features),), q))[0]
                               for q in (0, 1)], -1).numpy().astype(np.uint8)
    hit = predictions == np.arange(2)
    result = {}
    for name, mask in (('all_menus', np.ones(len(topics), bool)),
                       ('primary_fit_menus', np.isin(topics, fit_topics)),
                       ('unfitted_menus', ~np.isin(topics, fit_topics))):
        result[name] = [int(hit[mask].sum()), int(hit[mask].size)]
    return result, predictions


def retention_gate(summaries, audits, pairs, seeds, thresholds):
    def rate(p):
        return p[0]/p[1] if p[1] else 0.
    result = {}
    for kind in ('minimal', 'report_conditioned'):
        checks = {}
        for seed in seeds:
            name = f'{kind}_{seed}'
            intact = summaries[f'{name}_intact']['primary_fit_menus']
            zero = summaries[f'{name}_fixed_zero']['primary_fit_menus']
            flip = pairs[name]['primary_fit_menus']['flipped_when_intact_complete']
            bits = audits[name]['directed']['primary_fit_menus']
            conditions = dict(all_four=rate(intact['all_four_correct']) >= thresholds['all_four'],
                novel_masks=rate(intact['novel_masks']) >= thresholds['novel_masks'],
                retain_reference=rate(bits) >= thresholds['retention'],
                erasure_drop=rate(intact['all_four_correct'])-rate(zero['all_four_correct']) >= thresholds['erasure_drop'],
                conditional_inversion=rate(flip) >= thresholds['conditional_inversion'],
                initial_complete=intact['first_complete'] == [448, 448])
            checks[str(seed)] = dict(passed=all(conditions.values()), checks=conditions,
                all_four=intact['all_four_correct'], retention=bits, inversion=flip)
        result[kind] = dict(passed=all(x['passed'] for x in checks.values()), seeds=checks)
    return dict(kinds=result, scope='learned_binary_retention_frozen_reader_known_menus',
                qualification_loaded=False, reader_and_reference_routing_frozen=True)
