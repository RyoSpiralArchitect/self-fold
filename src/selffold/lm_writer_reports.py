"""Prompt-bound, complete and atomic report caches for future W-RITE runs.

Legacy unbound JSONL caches are never imported automatically. A cache is only
published after every declared source/mask has a generation receipt. Missing
mentions are decoder outcomes; missing records are errors, never zero bits.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .lm_writer import (DECODER_VERSION, STATE_DECODER_VERSION, STYLES,
                        decode_bit, decode_state, mentions_offoff, prompt_for)


def _decoder_for(style):
    if style.startswith("state_"):
        return decode_state, STATE_DECODER_VERSION
    return decode_bit, DECODER_VERSION


def _decode(style, text, options, pole):
    return _decoder_for(style)[0](text, options, pole)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def report_requests(style, masks, keys, poles, groups):
    if style not in STYLES or len(keys) != len(poles) or not keys:
        raise ValueError('Invalid style or source bank')
    if not masks or len(set(masks)) != len(masks) or any(type(m) is not int or not 0 <= m < 16 for m in masks):
        raise ValueError('Expected unique four-world masks')
    requests = []
    for source, (key, pole) in enumerate(zip(keys, poles)):
        options = groups[key[0]]['options']
        for mask in masks:
            requests.append(dict(source=source, source_key=list(key), mask=mask,
                                 style=style, options=options, pole=int(pole),
                                 prompt=prompt_for(style, options, int(pole), mask)))
    return requests


def validate_records(records, requests):
    expected = {(r['source'], r['mask']): r for r in requests}
    if len(expected) != len(requests) or len(records) != len(requests):
        raise ValueError('Report cache is incomplete or has extra records')
    seen = set()
    for row in records:
        key = (row.get('source'), row.get('mask'))
        if key not in expected or key in seen:
            raise ValueError('Duplicate or foreign report key')
        req = expected[key]
        if row.get('request_sha256') != fingerprint(req) or row.get('style') != req['style']:
            raise ValueError('Report cache does not match the exact prompt/source binding')
        receipt = row.get('generation')
        if (not isinstance(row.get('text'), str) or not isinstance(receipt, dict)
                or not isinstance(receipt.get('stop'), str)
                or type(receipt.get('token_count')) is not int
                or receipt['token_count'] < 0
                or receipt.get('censored') != (receipt['stop'] in {'length', 'context_limit'})):
            raise ValueError('Missing or invalid generation receipt')
        seen.add(key)
    return records


def cached_reports(host, requests, binding, cache_dir, output_dir):
    """Reuse only an exact binding; leave interrupted generation as .partial."""
    if not requests:
        raise ValueError('An empty report bank cannot be cached')
    contract = dict(schema='lm_report_cache_v3', writer=binding, requests=requests)
    key = fingerprint(contract)
    style = requests[0]['style']
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{style}-{key}.json'
    if path.exists():
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('contract_sha256') != key:
            raise ValueError('Wrong report cache binding')
        records = payload.get('records', [])
        if payload.get('records_sha256') != fingerprint(records):
            raise ValueError('Report cache content hash mismatch')
        validate_records(records, requests)
        mode = 'cache_reuse'
    else:
        records = []
        # This file preserves generated rows on interruption, but is never a cache hit.
        partial = output_dir / f'reports_{style}.generation.partial.jsonl'
        with partial.open('x', encoding='utf-8') as fh:
            for req in requests:
                out = host.generate(req['prompt'])
                receipt = dict(out.receipt())
                receipt.pop('text', None)
                row = dict(source=req['source'], mask=req['mask'], style=style,
                           request_sha256=fingerprint(req), text=out.text, generation=receipt)
                records.append(row)
                fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                fh.flush()
        validate_records(records, requests)
        payload = dict(contract_sha256=key, records_sha256=fingerprint(records), records=records)
        tmp = path.with_suffix('.partial')
        with tmp.open('x', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        # Atomic no-overwrite publication. A competing writer must not replace a cache.
        os.link(tmp, path)
        tmp.unlink()
        mode = 'generated'
    return records, dict(contract_sha256=key, records_sha256=fingerprint(records),
                         mode=mode, count=len(records), writer=binding)


def build_report_matrix(host, cfg, keys, poles, groups, output_dir, cache_dir, binding):
    masks = cfg['fit_masks'] + cfg['test_masks']
    styles = tuple(dict.fromkeys(a['channel'][3:] for a in cfg['arms']
                                 if a['channel'].startswith('lm_')))
    n = len(keys)
    bits, missing, offoff = {}, {}, {}
    output_dir = Path(output_dir)
    for style in styles:
        requests = report_requests(style, masks, keys, poles, groups)
        records, receipt = cached_reports(host, requests, binding, cache_dir, output_dir)
        bits[style] = torch.full((n, 16), -1, dtype=torch.long)
        missing[style] = np.full((n, 16), np.nan)
        offoff[style] = np.full((n, 16), np.nan)
        with (output_dir / f'reports_{style}.jsonl').open('x', encoding='utf-8') as fh:
            for row in records:
                i, m = row['source'], row['mask']
                options = groups[keys[i][0]]['options']
                bit, miss = _decode(style, row['text'], options, int(poles[i]))
                bits[style][i, m] = bit
                missing[style][i, m] = miss
                offoff[style][i, m] = mentions_offoff(row['text'])
                # Redecode the text; never copy stale stored bit/missing annotations.
                fh.write(json.dumps(dict(row, bit=bit, missing=miss,
                                         decoder_version=_decoder_for(style)[1]),
                                    ensure_ascii=False) + '\n')
        with (output_dir / f'reports_{style}.binding.json').open('x', encoding='utf-8') as fh:
            json.dump(dict(receipt, decoder_version=DECODER_VERSION, requests=requests), fh,
                      sort_keys=True, ensure_ascii=False, allow_nan=False)
    return bits, missing, offoff


def save_report_matrices(output_dir, masks, bits, missing, offoff):
    """Save every active style; rates exclude the two unsampled mask columns."""
    output_dir = Path(output_dir)
    if not masks or len(set(masks)) != len(masks) or any(not 0 <= m < 16 for m in masks):
        raise ValueError('Invalid sampled mask set')
    if not bits or set(bits) != set(missing) or set(bits) != set(offoff):
        raise ValueError('Report matrix styles disagree')
    for style, matrix in bits.items():
        if matrix.ndim != 2 or matrix.shape[1] != 16 or matrix.shape[0] == 0:
            raise ValueError('Expected a nonempty [sources,16] report matrix')
        if not torch.all((matrix[:, masks] == 0) | (matrix[:, masks] == 1)):
            raise ValueError('Unobserved report bit in sampled masks')
        for observation in (missing[style], offoff[style]):
            if observation.shape != tuple(matrix.shape) or not np.isin(observation[:, masks], [0, 1]).all():
                raise ValueError('Incomplete report statistics')
    path = output_dir / 'lm_matrix.npz'
    with path.open('xb') as fh:
        np.savez_compressed(fh, **{s: b.cpu().numpy() for s, b in bits.items()},
                            evaluated_masks=np.asarray(masks, dtype=np.int32))
    summary = dict(schema='lm_report_matrix_v3', decoder_version=DECODER_VERSION,
                   evaluated_masks=masks,
                   report_counts={s: int(len(b) * len(masks)) for s, b in bits.items()},
                   missing={s: {str(m): float(a[:, m].mean()) for m in masks}
                            for s, a in missing.items()},
                   offoff_rate={s: float(a[:, masks].mean()) for s, a in offoff.items()})
    with (output_dir / 'lm_matrix.json').open('x', encoding='utf-8') as fh:
        json.dump(summary, fh, sort_keys=True, allow_nan=False)
    return summary
