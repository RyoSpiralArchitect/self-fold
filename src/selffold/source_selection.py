"""A matched, typed-source cue; no programmatic selection of an action answer."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch

from .actual_report import continuation_probe, frozen_reader, report_summary, score_report
from .reference_evaluation import paired_reference
from .reference_retention import RetentionCell, audit_cell, rollout, summarize
from .role_reference import source_reference
from .roundtrip import feasible_outs, render_table, text_features


def requests(design, groups):
    rows = []
    for topic, pole, prior, mask in itertools.product(
            design['topics'], (0, 1), design['priors'], design['masks']):
        options = groups[topic]['options']
        for out in feasible_outs(prior, mask):
            transition = f'{topic}:{pole}:{prior}:{mask}:{out}'
            previous = 'PREVIOUS TABLE:\n' + render_table(options, pole, prior)
            records = dict(instruction='INSTRUCTION RECORD:\n' + render_table(options, pole, mask),
                           observation='OBSERVATION RECORD (the table now):\n' + render_table(options, pole, out))
            for layout in design['layouts']:
                order = ('instruction', 'observation') if layout.startswith('instruction') else ('observation', 'instruction')
                common = ('You report records of a small state machine. The records may disagree.\n'
                    'Valid action names (copy exactly one complete line, without quotes or explanation):\n'
                    + '\n'.join(options) + '\n\n' + previous + '\n\n'
                    + '\n\n'.join(records[k] for k in order) + '\n\n')
                # Rotate the four matched conditions; do not systematically run one arm first.
                conditions = list(itertools.product(design['arms'], ('instruction', 'actual')))
                shift = (len(rows) // 4) % len(conditions)
                for arm, question in conditions[shift:] + conditions[:shift]:
                    query = ('Question: which action did this instruction assign to the off/off row?'
                             if question == 'instruction' else
                             'Question: which action does the table now contain in its off/off row?')
                    cue = ''
                    if arm == 'source_cue':
                        record = 'INSTRUCTION RECORD' if question == 'instruction' else 'OBSERVATION RECORD'
                        cue = (f'\nRequested record: {record}. Read the off/off entry of that record. '
                               'The other records are not the answer to this question.')
                    pair = f'{layout}:{transition}'
                    rows.append(dict(id=f'{arm}:{pair}:{question}', pair=pair, transition=transition,
                        arm=arm, layout=layout, topic=topic, pole=pole, prior=prior, mask=mask, out=out,
                        options=options, question=question, prompt=common+query+cue+
                        '\nAnswer with one of the valid action names only.'))
    if len(rows) != design['generation_calls']:
        raise ValueError('Unexpected report panel size')
    return rows


def rate(pair):
    return pair[0] / pair[1] if pair[1] else 0.


def score_panel(reqs, rows, design):
    if len(rows) != len(reqs) or len({r['id'] for r in rows}) != len(reqs):
        raise ValueError('Incomplete or duplicate report panel')
    by_id = {r['id']: r for r in rows}
    if set(by_id) != {q['id'] for q in reqs}:
        raise ValueError('Foreign report identifiers')
    scored = [score_report(q, by_id[q['id']]) | {k:q[k] for k in ('arm','layout','transition')}
              for q in reqs]
    summaries, gates, joint = {}, {}, {}
    for arm in design['arms']:
        summaries[arm] = {}
        for label, layout in [('pooled', None)] + [(x, x) for x in design['layouts']]:
            selected = [q for q in reqs if q['arm'] == arm and (layout is None or q['layout'] == layout)]
            part, _ = report_summary(selected, [by_id[q['id']] for q in selected], design)
            summaries[arm][label] = part
        transitions = {}
        for s in scored:
            if s['arm'] == arm and s['coordinate_conflict']:
                transitions.setdefault(s['transition'], []).append(s)
        if any(len(v) != 2*len(design['layouts']) for v in transitions.values()):
            raise ValueError('Incomplete source/order conflict block')
        joint[arm] = {k: all(s['requested_correct'] for s in v) for k,v in transitions.items()}
        count = [sum(joint[arm].values()), len(joint[arm])]
        checks = dict(
            report_criteria_all_orders=all(v['gate']['passed'] for v in summaries[arm].values()),
            instruction_accuracy_all_orders=all(rate(v['instruction']['all']['requested_correct'])
                >= design['thresholds']['instruction_all'] for v in summaries[arm].values()),
            instruction_naming_all_orders=all(rate(v['instruction']['all']['strict_requested_correct'])
                >= design['thresholds']['strict_requested'] for v in summaries[arm].values()),
            joint_order_selectivity=rate(count) >= design['thresholds']['joint_order_selectivity'])
        gates[arm] = dict(passed=all(checks.values()), checks=checks,
                         joint_order_selectivity=count)
    if set(joint['copy_only']) != set(joint['source_cue']):
        raise ValueError('Unpaired intervention panel')
    paired = {name: 0 for name in ('both', 'cue_only', 'copy_only', 'neither')}
    for key in joint['copy_only']:
        control, cue = joint['copy_only'][key], joint['source_cue'][key]
        paired['both' if control and cue else 'cue_only' if cue else 'copy_only' if control else 'neither'] += 1
    n = len(joint['copy_only'])
    gain = (paired['cue_only'] - paired['copy_only']) / n
    benefit = dict(passed=gain >= design['thresholds']['matched_gain'],
                   difference=gain, minimum_difference=design['thresholds']['matched_gain'],
                   paired=paired, total=n)
    return dict(arms=summaries, admission=gates, matched_benefit=benefit,
                scope='supplied_record_cue_known_menus_two_table_orders', qualification_loaded=False), scored


def probe(root, reqs, rows, design, archive):
    by_id = {r['id']:r for r in rows}
    summaries, judgments = {}, []
    for arm, layout in itertools.product(design['arms'], design['layouts']):
        subset = [q for q in reqs if (q['arm'],q['layout']) == (arm,layout)]
        summary, records = continuation_probe(root, subset, [by_id[q['id']] for q in subset], design, archive)
        summaries[f'{arm}/{layout}'] = summary
        judgments.extend(dict(r, arm=arm, layout=layout) for r in records)
    return summaries, judgments


def load_cell(run: Path, kind, seed):
    if kind == 'copy':
        return None
    cell = RetentionCell(seed, kind == 'report_conditioned').requires_grad_(False)
    with np.load(run/'fits.npz', allow_pickle=False) as weights:
        cell.load_state_dict({k:torch.from_numpy(weights[f'{kind}_{seed}_{k}']) for k in cell.state_dict()})
    return cell


@torch.no_grad()
def compose(root, reqs, rows, design, archive, cfg):
    by_id = {r['id']:r for r in rows}
    ids = [i for i,k in enumerate(archive['keys']) if k[0] in design['topics']]
    keys = [archive['keys'][i] for i in ids]
    offset = len(archive['fit_keys'])
    members = archive['members'][np.asarray(ids)+offset]
    top = archive['top'][ids]
    cfg = dict(cfg, fit_masks=[], test_masks=design['masks'])
    arrays, summaries, audits, pairs = {}, {}, {}, {}
    for arm, layout, question in itertools.product(design['arms'], design['layouts'], ('instruction','actual')):
        chosen = [q for q in reqs if (q['arm'],q['layout'],q['question']) == (arm,layout,question)]
        bank = torch.full((len(ids),16,16,16,32), float('nan'))
        feats, topics = [], []
        for q in chosen:
            feature = text_features(by_id[q['id']]['text'])
            selected = [i for i,k in enumerate(keys) if (k[0],k[3]) == (q['topic'],q['pole'])]
            bank[selected,q['prior'],q['mask'],q['out']] = feature
            feats.append(feature); topics.append(q['topic'])
        for seed in design['seeds']:
            reader = frozen_reader(root, seed, design['previous_run'])
            for kind in ('copy', *design['kinds']):
                cell = load_cell(root/design['retention_run'], kind, seed)
                prefix = f'{arm}__{layout}__{question}__{kind}_{seed}'
                if cell is not None:
                    audits[prefix] = audit_cell(cell, torch.stack(feats), np.asarray(topics), cfg['fit_topics'])[0]
                for mode in design['reference_modes']:
                    name = f'{prefix}__{mode}'
                    value = rollout(reader, cell, members, source_reference(members), design['masks'], bank, mode)
                    arrays[name] = value
                    summaries[name] = summarize(value, keys, cfg, top, archive['labels'])
                pairs[prefix] = paired_reference(arrays[f'{prefix}__intact'], arrays[f'{prefix}__opposed'],
                    keys, cfg, top, archive['labels'])
    if len(arrays) != design['composition_evaluations']:
        raise ValueError('Incomplete composition panel')
    gates = {}
    for arm, kind in itertools.product(design['arms'], ('copy', *design['kinds'])):
        cells = {}
        for layout, seed in itertools.product(design['layouts'], design['seeds']):
            prefix = f'{arm}__{layout}__actual__{kind}_{seed}'
            intact = summaries[f'{prefix}__intact']['primary_fit_menus']['all_four_correct']
            zero = summaries[f'{prefix}__fixed_zero']['primary_fit_menus']['all_four_correct']
            flip = pairs[prefix]['primary_fit_menus']['flipped_when_intact_complete']
            checks = dict(all_four=rate(intact)>=design['thresholds']['all_four'],
                erasure_drop=rate(intact)-rate(zero)>=design['thresholds']['erasure_drop'],
                conditional_inversion=rate(flip)>=design['thresholds']['conditional_inversion'])
            if kind != 'copy':
                checks['fresh_reference_retention'] = rate(audits[prefix]['primary_fit_menus'])>=design['thresholds']['retention']
            cells[f'{layout}/{seed}'] = dict(passed=all(checks.values()),checks=checks,
                                            all_four=intact, conditional_inversion=flip)
        gates[f'{arm}/{kind}'] = dict(passed=all(c['passed'] for c in cells.values()),cells=cells)
    return dict(summaries=summaries,arrays=arrays,audits=audits,pairs=pairs,gates=gates)


def combined_gate(report, continuation, composition, design):
    continuation_pass = all(continuation[f'source_cue/{layout}']['gate']['passed'] for layout in design['layouts'])
    report_pass = report['admission']['source_cue']['passed']
    combined = {kind: report_pass and continuation_pass and composition['gates'][f'source_cue/{kind}']['passed']
                for kind in design['kinds']}
    return dict(report_admission=report_pass, continuation=continuation_pass,
        combined=combined, attributable_cue_benefit=report['matched_benefit']['passed'],
        supplied_source_cue=True, reference_extraction_and_routing_supplied=True,
        model_and_readers_frozen=True, qualification_loaded=False)
