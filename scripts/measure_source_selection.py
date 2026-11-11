"""Generate one sealed source-cue comparison and replay its frozen consumers."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from selffold.cli import make_host
from selffold.lm_writer_reports import fingerprint, generate_kv
from selffold.records import RunStore, code_manifest, file_sha
from selffold.reference_evaluation import logical_hash, save_compact
from selffold.report_replay import load_archive, read, verify_files
from selffold.source_selection import combined_gate, compose, probe, requests, score_panel

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT/'data/source_selection_20261111.json'


def inputs():
    design = read(DESIGN)
    manifest = read(ROOT/design['inputs_manifest'])
    cfg = read(ROOT/design['base_config'])
    archive = load_archive(ROOT, cfg, manifest)
    reqs = requests(design, read(ROOT/design['actions'])['groups'])
    previous = read(ROOT/design['previous_report_run']/'requests.json')
    def keys(rows):
        return {(r['topic'],r['pole'],r['prior'],r['mask'],r['out']) for r in rows}
    if keys(reqs) & keys(previous):
        raise ValueError('Current transitions overlap the write10 generation panel')
    if set(design['masks']) & set(cfg['fit_masks']):
        raise ValueError('The selected instructions were fitted by the frozen reader')
    if {r['out'] for r in reqs} - set(design['priors']):
        raise ValueError('Missing reachable prior states')
    return design, manifest, cfg, archive, reqs


def calculate(design, cfg, archive, reqs, rows):
    reporting, scored = score_panel(reqs, rows, design)
    continuation, judgments = probe(ROOT, reqs, rows, design, archive)
    composition = compose(ROOT, reqs, rows, design, archive, cfg)
    gate = combined_gate(reporting, continuation, composition, design)
    return dict(report_summary=reporting, scored_reports=scored,
        continuation_summary=continuation, continuation_judgments=judgments,
        composition_summary=composition['summaries'], fresh_retention=composition['audits'],
        paired_reference=composition['pairs'], composition_gate=composition['gates'],
        gate=gate), composition['arrays']


def run(out, checkpoint):
    store = RunStore(out); started = time.monotonic(); completed = 0
    try:
        design, manifest, cfg, archive, reqs = inputs()
        source = code_manifest(ROOT)
        if not source['git_commit'] or source['git_status']:
            raise ValueError('A clean source seal is required')
        store.write('source.json', source)
        store.write('design.json', dict(design=design, design_sha256=file_sha(DESIGN),
            plan_sha256=file_sha(ROOT/design['plan']), input_manifest_sha256=file_sha(ROOT/design['inputs_manifest'])))
        store.write('requests.json', reqs)
        args = SimpleNamespace(backend='hf', checkpoint=checkpoint,
            profile=str(ROOT/'configs/models/gemma2b.yaml'), device=design['device'],
            dtype=design['dtype'], cpu_threads=design['cpu_threads'], slots=8, cut=None,
            writer=None, max_context=design['max_context'], max_new_tokens=design['max_new_tokens'],
            wall_seconds=design['wall_seconds'], reviewed_local_code=None, transport=None, encoder_bridge=None)
        host, identity = make_host(args)
        if identity['checkpoint_sha256'] != design['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
        host.max_forwards = 100_000
        host.max_processed_tokens = 3_000_000
        store.write('host.json', dict(host.identity, device=design['device'], dtype=design['dtype'],
            generation_policy='kv_capped_argmax', max_new_tokens=design['max_new_tokens'],
            cpu_threads=design['cpu_threads'], cuda_device=torch.cuda.get_device_name()))
        parity = host.native_parity(); store.write('native_parity.json',parity)
        if not parity['exact']:
            raise ValueError('Native host parity failed')
        rows = []
        for req in reqs:
            output = generate_kv(host, req['prompt'])
            receipt = dict(output.receipt()); receipt.pop('text',None)
            row = dict(id=req['id'],request_sha256=fingerprint(req),text=output.text,generation=receipt)
            store.append('reports.jsonl',row); rows.append(row); completed += 1
            if completed % 64 == 0:
                print(json.dumps(dict(event='generated',count=completed,total=len(reqs))),flush=True)
        store.write('records_closed.json',dict(count=completed,sha256=file_sha(out/'reports.jsonl'),
            compute=host.meter.snapshot(),elapsed_seconds=time.monotonic()-started))
        del host
        gc.collect(); torch.cuda.empty_cache()
        torch.set_num_threads(1)
        values, arrays = calculate(design,cfg,archive,reqs,rows)
        for name,value in values.items(): store.write(name+'.json',value)
        (out/'trajectories').mkdir()
        files = {}
        for name,data in arrays.items():
            path = out/'trajectories'/f'{name}.npz'
            spec = save_compact(path,data)
            files[name] = dict(path=str(path.relative_to(out)),sha256=file_sha(path),arrays=spec)
        store.write('trajectory_files.json',files)
        verify_files(ROOT,manifest['files'])
        store.write('terminal.json',dict(status='SOURCE_SELECTION_CLOSED',generation_calls=completed,
            new_native_scoring_calls=0,fits=0,composition_evaluations=len(arrays),
            continuation_judgments=len(values['continuation_judgments']),old_inputs_unchanged=True,
            qualification_loaded=False,elapsed_seconds=time.monotonic()-started))
    except Exception as exc:
        store.write('terminal.json',dict(status='TECHNICAL_STOP',error=str(exc),generation_calls=completed,
            elapsed_seconds=time.monotonic()-started))
        raise


def reduce(run):
    design, _, cfg, archive, reqs = inputs()
    assert read(run/'terminal.json')['status'] == 'SOURCE_SELECTION_CLOSED'
    assert reqs == read(run/'requests.json')
    assert file_sha(run/'reports.jsonl') == read(run/'records_closed.json')['sha256']
    rows = [json.loads(line) for line in (run/'reports.jsonl').read_text().splitlines()]
    values, arrays = calculate(design,cfg,archive,reqs,rows)
    for name,value in values.items():
        assert value == read(run/(name+'.json')),name
    saved_files = read(run/'trajectory_files.json')
    assert set(saved_files) == set(arrays)
    for name,spec in saved_files.items():
        path = run/spec['path']; assert file_sha(path) == spec['sha256']
        with np.load(path,allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays[name]) == set(spec['arrays'])
            for field,value in arrays[name].items():
                record = spec['arrays'][field]
                assert np.array_equal(value,saved[field])
                assert logical_hash(value) == record['c_order_sha256']
                assert list(value.shape) == record['shape'] and str(value.dtype) == record['dtype']
    return dict(exact=True,report_records=len(rows),continuation_judgments=len(values['continuation_judgments']),
        composition_evaluations=len(arrays),composition_panels=len(arrays)*3,
        paired_conditions=len(values['paired_reference']),new_generation_calls=0,new_native_scoring_calls=0,
        qualification_loaded=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--checkpoint')
    parser.add_argument('--reduce',type=Path)
    args = parser.parse_args()
    torch.set_default_device('cpu'); torch.set_num_threads(1)
    if args.reduce:
        print(json.dumps(reduce(args.reduce),sort_keys=True))
    elif args.out and args.checkpoint:
        run(args.out,args.checkpoint)
    else:
        parser.error('Use --reduce RUN or --out RUN --checkpoint PATH')
