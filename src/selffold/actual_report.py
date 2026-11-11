"""Matched questions about an instruction versus the realized output table."""
from __future__ import annotations

import itertools
import re

import numpy as np
import torch

from .lm_writer import decode_state_slot
from .lm_writer_reports import fingerprint
from .mixture_boundary import eval_step, realize
from .role_reference import ReferenceReader, attach, source_reference
from .roundtrip import feasible_outs, rt_prompt, text_features


def requests(design, groups):
    rows = []
    for topic, pole, prior, mask in itertools.product(
            design['topics'], (0, 1), design['priors'], design['masks']):
        options = groups[topic]['options']
        for out in feasible_outs(prior, mask):
            directed = rt_prompt('rt_directed', options, pole, prior, mask, out)
            actual = directed.split('Question:')[0] + (
                'Question: which action does the table now contain in its '
                'off/off row? Answer with the action name only.')
            pair = f'{topic}:{pole}:{prior}:{mask}:{out}'
            order = ('instruction', 'actual') if len(rows) % 4 == 0 else ('actual', 'instruction')
            for question in order:
                rows.append(dict(id=f'{pair}:{question}', pair=pair, topic=topic,
                    pole=pole, prior=prior, mask=mask, out=out, question=question,
                    options=options, prompt=directed if question == 'instruction' else actual))
    return rows


def normalize(text):
    return re.sub(r'\W+', ' ', text.casefold()).strip()


def score_report(req, row):
    if row['request_sha256'] != fingerprint(req) or row['id'] != req['id']:
        raise ValueError('Report is not bound to this request')
    receipt = row['generation']
    if (type(receipt['token_count']) is not int or receipt['token_count'] < 0
            or receipt['censored'] != (receipt['stop'] in ('length', 'context_limit'))):
        raise ValueError('Invalid generation stop receipt')
    slot, missing = decode_state_slot(row['text'], req['options'])
    expected = dict(actual=(req['out'] & 1) ^ req['pole'],
                    instruction=(req['mask'] & 1) ^ req['pole'])
    strict = [i for i, option in enumerate(req['options'])
              if normalize(row['text']) == normalize(option)]
    # A capped but informative report is scored normally; censoring is separate.
    return dict(id=req['id'], pair=req['pair'], question=req['question'],
        topic=req['topic'], pole=req['pole'], output_error=req['out'] != req['mask'],
        coordinate_conflict=expected['actual'] != expected['instruction'],
        collapsed=req['out'] in (0, 15), missing=bool(missing),
        censored=bool(receipt['censored']), decoded_slot=None if missing else int(slot),
        actual_correct=not missing and slot == expected['actual'],
        instruction_correct=not missing and slot == expected['instruction'],
        requested_correct=not missing and slot == expected[req['question']],
        strict_requested_correct=len(strict) == 1 and strict[0] == expected[req['question']])


def report_summary(reqs, rows, design):
    if len(rows) != len(reqs) or len({r['id'] for r in rows}) != len(reqs):
        raise ValueError('Incomplete or duplicate report panel')
    indexed = {r['id']: r for r in rows}
    scored = [score_report(q, indexed[q['id']]) for q in reqs]
    def count(rs, key):
        return [sum(bool(r[key]) for r in rs), len(rs)]
    result = {}
    for question in ('instruction', 'actual'):
        chosen = [r for r in scored if r['question'] == question]
        panels = dict(all=chosen, correct_outputs=[r for r in chosen if not r['output_error']],
            erroneous_outputs=[r for r in chosen if r['output_error']],
            coordinate_conflict=[r for r in chosen if r['coordinate_conflict']],
            collapsed=[r for r in chosen if r['collapsed']],
            noncollapsed=[r for r in chosen if not r['collapsed']])
        panels.update({f'topic_{t}': [r for r in chosen if r['topic'] == t] for t in design['topics']})
        result[question] = {name: {key: count(rs, key) for key in
            ('actual_correct', 'instruction_correct', 'requested_correct',
             'strict_requested_correct', 'missing', 'censored')} for name, rs in panels.items()}
    pairs = {}
    for row in scored:
        pairs.setdefault(row['pair'], {})[row['question']] = row
    conflicts = [p for p in pairs.values() if p['actual']['coordinate_conflict']]
    selective = [p['actual']['requested_correct'] and p['instruction']['requested_correct'] for p in conflicts]
    result['paired_question_selectivity'] = [sum(selective), len(selective)]
    def rate(pair):
        return pair[0] / pair[1] if pair[1] else 0.
    limits = design['thresholds']
    checks = dict(actual_all=rate(result['actual']['all']['actual_correct']) >= limits['actual_all'],
        actual_conflicts=rate(result['actual']['coordinate_conflict']['actual_correct']) >= limits['actual_conflicts'],
        question_selectivity=rate(result['paired_question_selectivity']) >= limits['question_selectivity'],
        strict_requested=rate(result['actual']['all']['strict_requested_correct']) >= limits['strict_requested'],
        every_menu=all(rate(result['actual'][f'topic_{t}']['actual_correct']) >= limits['every_menu'] for t in design['topics']))
    result['gate'] = dict(passed=all(checks.values()), checks=checks,
        scope='machine_decoded_offoff_reporting_on_declared_transitions', qualification_loaded=False)
    return result, scored


def frozen_reader(root, seed, previous_run):
    model = ReferenceReader(seed, 1, 32).requires_grad_(False)
    with np.load(root / previous_run / 'fits.npz', allow_pickle=False) as f:
        prefix = f'referenced_rt_directed_{seed}_'
        model.load_state_dict({k: torch.from_numpy(f[prefix+k]) for k in model.state_dict()})
    return model


@torch.no_grad()
def continuation_probe(root, reqs, rows, design, archive):
    """Frozen controller and archived native row scores; no new model calls."""
    by_id = {r['id']: r for r in rows}
    outputs = []
    for seed in design['seeds']:
        actor = frozen_reader(root, seed, design['previous_run'])
        for q in reqs:
            indices = [i for i, key in enumerate(archive['keys'])
                       if key[0] == q['topic'] and key[3] == q['pole']]
            case = indices[0]
            offset = len(archive['fit_keys'])
            members = archive['members'][offset+case:offset+case+1]
            if members.argmax(-1).tolist() != [[2*q['topic']+q['pole'],2*q['topic']+1-q['pole']]]:
                raise ValueError('Rendered transition does not match the recovered compiled rows')
            if any(not torch.equal(archive['members'][offset+i], members[0]) for i in indices):
                raise ValueError('Compiled rows differ across matching source realizations')
            ref = source_reference(members)
            bits = ((q['out'] >> torch.arange(4)) & 1)[None]
            weights = torch.nn.functional.one_hot(bits, 2).float()
            reports = dict(generated=by_id[q['id']]['text'],
                           oracle=q['options'][(q['out'] & 1) ^ q['pole']])
            for kind, report in reports.items():
                incoming = attach(text_features(report)[None], ref)
                for mask in design['continuation_masks']:
                    _, after, _, _ = eval_step(actor, realize(weights, members), weights,
                        incoming, torch.tensor([mask]), 'hard', 'lmtext', members)
                    got = archive['top'][case, after.argmax(-1).numpy()[0]]
                    target = archive['labels'][q['pole'] ^ ((mask >> np.arange(4)) & 1)[:, None] ^ np.arange(2)]
                    outputs.append(dict(id=q['id'], seed=seed, kind=kind, topic=q['topic'],
                        question=q['question'], output_error=q['out'] != q['mask'],
                        collapsed=q['out'] in (0, 15), next_mask=mask,
                        correct=int((got == target).sum()), total=int(got.size),
                        complete=bool((got == target).all())))
    summary = {}
    for seed in design['seeds']:
        summary[str(seed)] = {}
        for question, kind in itertools.product(('instruction', 'actual'), ('generated', 'oracle')):
            chosen = [r for r in outputs if (r['seed'], r['question'], r['kind']) == (seed, question, kind)]
            for name, panel in (
                ('primary_noncollapsed', [r for r in chosen if r['topic'] in design['fit_topics'] and not r['collapsed']]),
                ('primary_collapsed', [r for r in chosen if r['topic'] in design['fit_topics'] and r['collapsed']]),
                ('unfitted_noncollapsed', [r for r in chosen if r['topic'] not in design['fit_topics'] and not r['collapsed']])):
                summary[str(seed)][f'{question}_{kind}_{name}'] = [sum(r['complete'] for r in panel), len(panel)]
    checks = {str(seed): all(summary[str(seed)][f'actual_{kind}_primary_noncollapsed'][0]
              / summary[str(seed)][f'actual_{kind}_primary_noncollapsed'][1] >= design['thresholds']['frozen_continuation']
              for kind in ('generated', 'oracle')) for seed in design['seeds']}
    return dict(seeds=summary, gate=dict(passed=all(checks.values()), checks=checks,
        scope='one_frozen_continuation_known_menus_noncollapsed_inputs')), outputs
