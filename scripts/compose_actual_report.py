"""Frozen four-update composition of fresh reports and final retention cells."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from fit_reference_retention import cell_from, inputs
from selffold.actual_report import frozen_reader, requests, score_report
from selffold.records import RunStore, code_manifest, file_sha
from selffold.reference_evaluation import logical_hash, paired_reference, save_compact
from selffold.reference_retention import audit_cell, rollout, summarize
from selffold.report_replay import read
from selffold.role_reference import source_reference
from selffold.roundtrip import text_features

ROOT = Path(__file__).resolve().parents[1]


def prepare(report_run, retention_run):
    design, cfg, _, archive = inputs()
    report_design = read(ROOT/'data/actual_report_20261111.json')
    if read(report_run/'terminal.json')['status'] != 'ACTUAL_REPORT_PANEL_CLOSED':
        raise ValueError('Report generation is incomplete')
    if read(retention_run/'terminal.json')['status'] != 'REFERENCE_RETENTION_CLOSED':
        raise ValueError('Retention fitting/evaluation is incomplete')
    bindings = {}
    for label, directory, names in (
        ('reports', report_run, ('requests.json', 'reports.jsonl', 'records_closed.json',
          'report_summary.json', 'continuation_summary.json', 'terminal.json')),
        ('retention', retention_run, ('fits.npz', 'fit_closed.json', 'gate.json', 'terminal.json'))):
        for name in names:
            bindings[f'{label}/{name}'] = file_sha(directory/name)
    assert file_sha(report_run/'reports.jsonl') == read(report_run/'records_closed.json')['sha256']
    assert file_sha(retention_run/'fits.npz') == read(retention_run/'fit_closed.json')['weights']
    reqs = requests(report_design, read(ROOT/report_design['actions'])['groups'])
    assert reqs == read(report_run/'requests.json')
    rows = [json.loads(line) for line in (report_run/'reports.jsonl').read_text().splitlines()]
    if len(rows) != len(reqs) or len({r['id'] for r in rows}) != len(reqs):
        raise ValueError('Incomplete report panel')
    by_id = {r['id']: r for r in rows}
    ids = [i for i,k in enumerate(archive['keys']) if k[0] in report_design['topics']]
    keys = [archive['keys'][i] for i in ids]
    banks = {question: torch.full((len(ids),16,16,16,32), float('nan'))
             for question in ('instruction','actual')}
    panels = {question: ([],[]) for question in banks}
    for q in reqs:
        score_report(q, by_id[q['id']])
        features = text_features(by_id[q['id']]['text'])
        chosen = [i for i,k in enumerate(keys) if (k[0],k[3]) == (q['topic'],q['pole'])]
        banks[q['question']][chosen,q['prior'],q['mask'],q['out']] = features
        panels[q['question']][0].append(features)
        panels[q['question']][1].append(q['topic'])
    offset = len(archive['fit_keys'])
    return dict(design=design, cfg=dict(cfg,fit_masks=[],test_masks=report_design['masks']),
        keys=keys, members=archive['members'][np.asarray(ids)+offset],
        top=archive['top'][ids], labels=archive['labels'], banks=banks, bindings=bindings,
        feature_panels={q:(torch.stack(v[0]),np.asarray(v[1])) for q,v in panels.items()})


def calculate(report_run, retention_run):
    context = prepare(report_run, retention_run)
    design, cfg = context['design'], context['cfg']
    summaries, arrays, audits, pairs = {}, {}, {}, {}
    for seed in design['seeds']:
        reader = frozen_reader(ROOT, seed, design['previous_run'])
        for kind in ('copy', *design['kinds']):
            cell = cell_from(retention_run, kind, seed)
            for question, bank in context['banks'].items():
                prefix = f'{question}_{kind}_{seed}'
                if cell is not None:
                    feats, topics = context['feature_panels'][question]
                    audits[prefix] = audit_cell(cell, feats, topics, cfg['fit_topics'])[0]
                for mode in design['reference_modes']:
                    name = f'{prefix}_{mode}'
                    data = rollout(reader, cell, context['members'], source_reference(context['members']),
                                   cfg['test_masks'], bank, mode)
                    arrays[name] = data
                    summaries[name] = summarize(data, context['keys'], cfg, context['top'], context['labels'])
                pairs[prefix] = paired_reference(arrays[f'{prefix}_intact'], arrays[f'{prefix}_opposed'],
                    context['keys'], cfg, context['top'], context['labels'])
    def rate(pair):
        return pair[0]/pair[1] if pair[1] else 0.
    gates = {}
    for kind in ('copy', *design['kinds']):
        checks = {}
        for seed in design['seeds']:
            prefix = f'actual_{kind}_{seed}'
            intact = summaries[f'{prefix}_intact']['primary_fit_menus']['all_four_correct']
            zero = summaries[f'{prefix}_fixed_zero']['primary_fit_menus']['all_four_correct']
            flip = pairs[prefix]['primary_fit_menus']['flipped_when_intact_complete']
            thresholds = design['thresholds']
            criteria = dict(all_four=rate(intact)>=thresholds['all_four'],
                            erasure_drop=rate(intact)-rate(zero)>=thresholds['erasure_drop'],
                            conditional_inversion=rate(flip)>=thresholds['conditional_inversion'])
            checks[str(seed)] = dict(passed=all(criteria.values()), checks=criteria,
                                    all_four=intact, inversion=flip)
        gates[kind] = dict(passed=all(c['passed'] for c in checks.values()), seeds=checks)
    report_gate = read(report_run/'report_summary.json')['gate']['passed']
    continuation_gate = read(report_run/'continuation_summary.json')['gate']['passed']
    retention = read(retention_run/'gate.json')['kinds']
    combined = {kind: report_gate and continuation_gate and retention[kind]['passed'] and gates[kind]['passed']
                for kind in design['kinds']}
    gate = dict(composition=gates, combined_with_learned_retention=combined,
                report_gate=report_gate, continuation_gate=continuation_gate,
                scope='four_known_menus_two_unfitted_instructions_frozen_components',
                qualification_loaded=False)
    return context['bindings'], summaries, arrays, audits, pairs, gate


def run(out, report_run, retention_run):
    store = RunStore(out); started = time.monotonic()
    try:
        source = code_manifest(ROOT)
        if not source['git_commit'] or source['git_status']:
            raise ValueError('Clean source required')
        store.write('source.json', source)
        bindings, summaries, arrays, audits, pairs, gate = calculate(report_run, retention_run)
        store.write('inputs.json', bindings)
        (out/'trajectories').mkdir()
        files = {}
        for name, data in arrays.items():
            path = out/'trajectories'/f'{name}.npz'
            spec = save_compact(path, data)
            files[name] = dict(path=str(path.relative_to(out)), sha256=file_sha(path), arrays=spec)
        for name, value in [('summary',summaries),('fresh_report_retention',audits),
                            ('paired_reference',pairs),('gate',gate),('trajectory_files',files)]:
            store.write(f'{name}.json',value)
        store.write('terminal.json', dict(status='FROZEN_COMPOSITION_CLOSED', evaluations=len(arrays),
            fits=0, generation_calls=0, new_native_calls=0, qualification_loaded=False,
            elapsed_seconds=time.monotonic()-started))
    except Exception as exc:
        store.write('terminal.json', dict(status='TECHNICAL_STOP',error=str(exc),
                                         elapsed_seconds=time.monotonic()-started))
        raise


def reduce(run, report_run, retention_run):
    bindings, summaries, arrays, audits, pairs, gate = calculate(report_run, retention_run)
    for name, value in [('inputs',bindings),('summary',summaries),('fresh_report_retention',audits),
                        ('paired_reference',pairs),('gate',gate)]:
        assert value == read(run/f'{name}.json'),name
    files = read(run/'trajectory_files.json')
    assert set(files) == set(arrays)
    for name, record in files.items():
        path = run/record['path']; assert file_sha(path) == record['sha256']
        with np.load(path,allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays[name]) == set(record['arrays'])
            for field, value in arrays[name].items():
                assert np.array_equal(value,saved[field])
                assert logical_hash(value) == record['arrays'][field]['c_order_sha256']
    return dict(exact=True,evaluations=len(arrays),panels=len(summaries)*3,
        paired_conditions=len(pairs),generation_calls=0,new_native_calls=0,qualification_loaded=False)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',type=Path,required=True)
    p.add_argument('--retention',type=Path,required=True)
    p.add_argument('--out',type=Path); p.add_argument('--reduce',type=Path)
    a=p.parse_args(); torch.set_default_device('cpu'); torch.set_num_threads(1)
    if a.reduce:
        print(json.dumps(reduce(a.reduce,a.reports,a.retention),sort_keys=True))
    elif a.out:
        run(a.out,a.reports,a.retention)
    else:
        p.error('Use --out RUN or --reduce RUN')
