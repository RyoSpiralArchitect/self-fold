"""Freeze the readout, learn reference writes, and compose with new reports."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import time

import numpy as np
import torch

from fit_codec import parameter_sha
from selffold.actual_report import frozen_reader
from selffold.records import RunStore, code_manifest, file_sha
from selffold.reference_evaluation import logical_hash, paired_reference, save_compact
from selffold.reference_retention import (RetentionCell, audit_cell, retention_gate,
                                        rollout, summarize, training_loss)
from selffold.report_replay import load_archive, read, reward_function, verify_files
from selffold.role_reference import source_reference
from selffold.roundtrip import text_features

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT/'data/reference_retention_20261111.json'
STATE = None


def inputs():
    design = read(DESIGN)
    cfg = read(ROOT/design['base_config'])
    manifest = read(ROOT/design['inputs_manifest'])
    return design, cfg, manifest, load_archive(ROOT, cfg, manifest)


def cell_from(run, kind, seed):
    if kind == 'copy':
        return None
    model = RetentionCell(seed, kind == 'report_conditioned').requires_grad_(False)
    with np.load(run/'fits.npz', allow_pickle=False) as f:
        model.load_state_dict({k: torch.from_numpy(f[f'{kind}_{seed}_{k}']) for k in model.state_dict()})
    return model


def feature_panels(archive):
    panels = {}
    for style in ('directed', 'natural'):
        rows = [json.loads(s) for s in (archive['archive']/f'reports_rt_{style}.jsonl').read_text().splitlines()]
        panels[style] = (torch.stack([text_features(r['text']) for r in rows]),
                         np.asarray([archive['pairs'][r['source']][0] for r in rows]))
    return panels


def fit(design, cfg, archive, store):
    panels = feature_panels(archive)
    reference = source_reference(archive['members'])
    fits, parameters, audits, predictions = [], {}, {}, {}
    nfit = len(archive['fit_keys'])
    available = torch.tensor(cfg['fit_masks'])
    for seed in design['seeds']:
        reader = frozen_reader(ROOT, seed, design['previous_run'])
        reader_sha = parameter_sha(reader)
        for kind in design['kinds']:
            name = f'{kind}_{seed}'
            cell = RetentionCell(seed, kind == 'report_conditioned')
            assert sum(p.numel() for p in cell.parameters()) == design['parameters'][kind]
            start = parameter_sha(cell)
            audits[name] = {}
            for stage in ('initial', 'final'):
                if stage == 'final':
                    rng = torch.Generator().manual_seed(20000+seed)
                    optimizer = torch.optim.Adam(cell.parameters(), lr=design['learning_rate'])
                    schedule = hashlib.sha256()
                    for update in range(design['updates']):
                        cases = torch.randint(nfit, (design['batch'],), generator=rng)
                        masks = available[torch.randint(len(available), (len(cases), 2), generator=rng)]
                        uniform = torch.rand(len(cases), generator=rng)
                        for value in (cases, masks, uniform):
                            schedule.update(value.numpy().tobytes())
                        reward = reward_function(cases.numpy(), archive['fit_keys'], archive['labels'], archive['fit_top'])
                        loss, observed = training_loss(cell, reader, archive['members'][cases],
                            reference[cases], cases, masks, uniform, archive['feats']['rt_directed'], reward,
                            floor=design['exploration_floor'])
                        optimizer.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(cell.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        store.append('training.jsonl', dict(fit=name, update=update+1,
                            loss=float(loss.detach()), second_native_reward=observed))
                audits[name][stage] = {}
                for style, (features, topics) in panels.items():
                    summary, predicted = audit_cell(cell, features, topics, cfg['fit_topics'])
                    audits[name][stage][style] = summary
                    predictions[f'{name}_{stage}_{style}'] = predicted
            if parameter_sha(reader) != reader_sha:
                raise ValueError('Frozen reader changed')
            fits.append(dict(name=name, parameters=sum(p.numel() for p in cell.parameters()),
                start=start, finish=parameter_sha(cell), schedule=schedule.hexdigest(),
                frozen_reader_sha256=reader_sha))
            parameters.update({f'{name}_{k}': v.detach().numpy() for k,v in cell.state_dict().items()})
            print(json.dumps(dict(event='fit_closed', **fits[-1])), flush=True)
    for seed in design['seeds']:
        assert len({f['schedule'] for f in fits if f['name'].endswith(f'_{seed}')}) == 1
    with (store.path/'fits.npz').open('xb') as f:
        np.savez_compressed(f, **parameters)
    save_compact(store.path/'audit_predictions.npz', predictions)
    store.write('fit_closed.json', dict(fits=fits, weights=file_sha(store.path/'fits.npz'),
        audit_predictions=file_sha(store.path/'audit_predictions.npz')))
    store.write('cell_audits.json', audits)


def evaluate(job):
    kind, seed, mode = job
    design, cfg, archive, out = STATE
    reader = frozen_reader(ROOT, seed, design['previous_run'])
    cell = cell_from(out, kind, seed)
    nfit = len(archive['fit_keys'])
    members = archive['members'][nfit:]
    data = rollout(reader, cell, members, source_reference(members),
                   cfg['fit_masks']+cfg['test_masks'], archive['feats']['rt_directed'][nfit:], mode)
    if kind == 'copy':
        old = ROOT/design['previous_run']/'trajectories'/f'referenced_rt_directed_{seed}_{mode}.npz'
        with np.load(old, allow_pickle=False) as f:
            if (not np.array_equal(data['depths'], f['depths'])
                    or not np.array_equal(data['paths'], f['paths'])
                    or not np.all(data['references'] == f['boundary_reference'][:,None,None])):
                raise ValueError('Engineered-copy baseline changed')
    name = f'{kind}_{seed}_{mode}'
    path = out/'trajectories'/f'{name}.npz'
    arrays = save_compact(path, data)
    return name, summarize(data, archive['keys'], cfg, archive['top'], archive['labels']), dict(
        path=str(path.relative_to(out)), sha256=file_sha(path), arrays=arrays)


def pairs_for(run, cfg, archive, names):
    pairs = {}
    for name in names:
        if not name.endswith('_intact'):
            continue
        prefix = name.removesuffix('_intact')
        with np.load(run/'trajectories'/f'{name}.npz', allow_pickle=False) as original, \
             np.load(run/'trajectories'/f'{prefix}_opposed.npz', allow_pickle=False) as flipped:
            pairs[prefix] = paired_reference(original, flipped, archive['keys'], cfg,
                                             archive['top'], archive['labels'])
    return pairs


def run(out, workers):
    global STATE
    store = RunStore(out); started = time.monotonic()
    try:
        design, cfg, manifest, archive = inputs()
        source = code_manifest(ROOT)
        if not source['git_commit'] or source['git_status']:
            raise ValueError('Clean source required')
        store.write('source.json', source)
        store.write('environment.json', dict(python=platform.python_version(),
            torch=torch.__version__, numpy=np.__version__, device='cpu', torch_threads=1,
            evaluation_workers=workers))
        store.write('request.json', dict(design=design, design_sha256=file_sha(DESIGN),
            plan_sha256=file_sha(ROOT/design['plan']), inputs_sha256=file_sha(ROOT/design['inputs_manifest']),
            generation_calls=0, new_native_calls=0))
        fit(design, cfg, archive, store)
        (out/'trajectories').mkdir()
        STATE = design, cfg, archive, out
        jobs = [(kind, seed, mode) for seed in design['seeds']
                for kind in ('copy', *design['kinds']) for mode in design['reference_modes']]
        summaries, files = {}, {}
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('fork')) as pool:
            for future in as_completed([pool.submit(evaluate, job) for job in jobs]):
                name, summary, record = future.result()
                summaries[name], files[name] = summary, record
                print(json.dumps(dict(event='evaluation_closed', name=name,
                    primary=summary['primary_fit_menus']['all_four_correct'])), flush=True)
        store.write('summary.json', summaries); store.write('trajectory_files.json', files)
        pairs = pairs_for(out, cfg, archive, summaries)
        store.write('paired_reference.json', pairs)
        audits = {k:v['final'] for k,v in read(out/'cell_audits.json').items()}
        store.write('gate.json', retention_gate(summaries, audits, pairs, design['seeds'], design['thresholds']))
        verify_files(ROOT, manifest['files'])
        store.write('terminal.json', dict(status='REFERENCE_RETENTION_CLOSED', fits=6,
            evaluations=27, engineered_copy_replays=9, generation_calls=0, new_native_calls=0,
            old_inputs_unchanged=True, qualification_loaded=False, elapsed_seconds=time.monotonic()-started))
    except Exception as exc:
        store.write('terminal.json', dict(status='TECHNICAL_STOP', error=str(exc),
            elapsed_seconds=time.monotonic()-started))
        raise


def reduce(run):
    design, cfg, _, archive = inputs()
    closed = read(run/'fit_closed.json')
    assert file_sha(run/'fits.npz') == closed['weights']
    assert file_sha(run/'audit_predictions.npz') == closed['audit_predictions']
    panels = feature_panels(archive)
    audits = read(run/'cell_audits.json')
    with np.load(run/'audit_predictions.npz', allow_pickle=False) as saved:
        for kind in design['kinds']:
            for seed in design['seeds']:
                name = f'{kind}_{seed}'
                for stage in ('initial', 'final'):
                    cell = RetentionCell(seed, kind == 'report_conditioned') if stage == 'initial' else cell_from(run, kind, seed)
                    for style, (features, topics) in panels.items():
                        summary, predictions = audit_cell(cell, features, topics, cfg['fit_topics'])
                        assert summary == audits[name][stage][style]
                        assert np.array_equal(predictions, saved[f'{name}_{stage}_{style}'])
    summaries = {}
    for name, record in read(run/'trajectory_files.json').items():
        path = run/record['path']; assert file_sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as data:
            assert set(data.files) == set(record['arrays'])
            for key, spec in record['arrays'].items():
                assert (logical_hash(data[key]) == spec['c_order_sha256'] and
                        list(data[key].shape) == spec['shape'] and str(data[key].dtype) == spec['dtype'])
            summaries[name] = summarize(data, archive['keys'], cfg, archive['top'], archive['labels'])
    pairs = pairs_for(run, cfg, archive, summaries)
    gate = retention_gate(summaries, {k:v['final'] for k,v in audits.items()}, pairs, design['seeds'], design['thresholds'])
    assert summaries == read(run/'summary.json')
    assert pairs == read(run/'paired_reference.json')
    assert gate == read(run/'gate.json')
    return dict(exact=True, fits=6, evaluations=len(summaries), panels=len(summaries)*3,
                paired_conditions=len(pairs), generation_calls=0, new_native_calls=0,
                qualification_loaded=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path); p.add_argument('--workers', type=int, choices=[1,2], default=2)
    p.add_argument('--reduce', type=Path)
    a = p.parse_args()
    torch.set_default_device('cpu'); torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if a.reduce:
        print(json.dumps(reduce(a.reduce), sort_keys=True))
    elif a.out:
        run(a.out, a.workers)
    else:
        p.error('Use --out RUN or --reduce RUN')
