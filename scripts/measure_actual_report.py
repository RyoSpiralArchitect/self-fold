"""Generate the sealed matched-question panel, then probe the frozen reader."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from selffold.actual_report import requests, report_summary, continuation_probe
from selffold.cli import make_host
from selffold.lm_writer_reports import fingerprint, generate_kv
from selffold.records import RunStore, code_manifest, file_sha
from selffold.report_replay import load_archive, read, verify_files

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / 'data/actual_report_20261111.json'


def reduce(run):
    design = read(DESIGN)
    manifest = read(ROOT / design['inputs_manifest'])
    archive = load_archive(ROOT, read(ROOT / design['base_config']), manifest)
    reqs = requests(design, read(ROOT / design['actions'])['groups'])
    if read(run / 'requests.json') != reqs:
        raise ValueError('Request panel changed')
    rows = [json.loads(line) for line in (run / 'reports.jsonl').read_text().splitlines()]
    summary, scored = report_summary(reqs, rows, design)
    probe, judgments = continuation_probe(ROOT, reqs, rows, design, archive)
    for name, actual in [('report_summary', summary), ('scored_reports', scored),
                         ('continuation_summary', probe), ('continuation_judgments', judgments)]:
        if actual != read(run / f'{name}.json'):
            raise ValueError(f'Saved reduction differs: {name}')
    closure = read(run / 'records_closed.json')
    if closure['sha256'] != file_sha(run / 'reports.jsonl'):
        raise ValueError('Generated report bytes changed')
    return dict(exact=True, report_records=len(rows), continuation_judgments=len(judgments),
                new_generation_calls=0, new_native_scoring_calls=0, qualification_loaded=False)


def run(out, checkpoint):
    store = RunStore(out)
    started = time.monotonic()
    completed = 0
    try:
        design = read(DESIGN)
        manifest = read(ROOT / design['inputs_manifest'])
        source = code_manifest(ROOT)
        if not source['git_commit'] or source['git_status']:
            raise ValueError('A clean source seal is required')
        archive = load_archive(ROOT, read(ROOT / design['base_config']), manifest)
        reqs = requests(design, read(ROOT / design['actions'])['groups'])
        if len(reqs) != design['generation_calls']:
            raise ValueError('Generation panel size differs')
        store.write('source.json', source)
        store.write('design.json', design | dict(design_sha256=file_sha(DESIGN),
            plan_sha256=file_sha(ROOT / design['plan']), inputs_sha256=file_sha(ROOT / design['inputs_manifest'])))
        store.write('requests.json', reqs)
        args = SimpleNamespace(backend='hf', checkpoint=checkpoint,
            profile=str(ROOT / 'configs/models/gemma2b.yaml'),
            device='cpu', dtype='float32', cpu_threads=6, slots=8,
            cut=None, writer=None, max_context=1024, max_new_tokens=48,
            wall_seconds=design['wall_seconds'], reviewed_local_code=None,
            transport=None, encoder_bridge=None)
        host, identity = make_host(args)
        if identity['checkpoint_sha256'] != design['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
        host.max_forwards = 30000
        host.max_processed_tokens = 2_000_000
        store.write('host.json', dict(host.identity, device='cpu', cpu_threads=6,
            generation_policy='kv_capped_argmax', max_new_tokens=48))
        parity = host.native_parity()
        store.write('native_parity.json', parity)
        if not parity['exact']:
            raise ValueError('Native host parity failed')
        rows = []
        for req in reqs:
            result = generate_kv(host, req['prompt'])
            receipt = dict(result.receipt()); receipt.pop('text', None)
            row = dict(id=req['id'], request_sha256=fingerprint(req), text=result.text,
                       generation=receipt)
            store.append('reports.jsonl', row)
            rows.append(row); completed += 1
            if completed % 32 == 0:
                print(json.dumps(dict(event='generated', count=completed, total=len(reqs))), flush=True)
        store.write('records_closed.json', dict(count=len(rows), sha256=file_sha(out/'reports.jsonl')))
        # Do not refit the controller after seeing report outcomes.
        torch.set_num_threads(1)
        summary, scored = report_summary(reqs, rows, design)
        probe, judgments = continuation_probe(ROOT, reqs, rows, design, archive)
        for name, value in [('report_summary', summary), ('scored_reports', scored),
                            ('continuation_summary', probe), ('continuation_judgments', judgments)]:
            store.write(f'{name}.json', value)
        verify_files(ROOT, manifest['files'])
        store.write('terminal.json', dict(status='ACTUAL_REPORT_PANEL_CLOSED',
            generation_calls=completed, new_native_scoring_calls=0, controller_fits=0,
            old_inputs_unchanged=True, qualification_loaded=False,
            compute=host.meter.snapshot(), elapsed_seconds=time.monotonic()-started))
    except Exception as exc:
        store.write('terminal.json', dict(status='TECHNICAL_STOP', error=str(exc),
            generation_calls=completed, elapsed_seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    p.add_argument('--checkpoint')
    p.add_argument('--reduce', type=Path)
    a = p.parse_args()
    torch.set_default_device('cpu'); torch.set_num_threads(1)
    if a.reduce:
        print(json.dumps(reduce(a.reduce), sort_keys=True))
    elif a.out and a.checkpoint:
        run(a.out, a.checkpoint)
    else:
        p.error('Use --reduce RUN or --out RUN --checkpoint PATH')
