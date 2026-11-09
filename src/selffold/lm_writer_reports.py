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
                        decode_bit, decode_state, decode_state_slot,
                        mentions_offoff, prompt_for)

RT_DECODER_VERSION = "rowbound_v4"


def _decoder_for(style):
    if style.startswith("rt_"):
        # Pole-free slot decode; the caller maps slot -> role bit via the pole.
        # A missing report falls back to bit 0, matching the state_* decoders'
        # missing policy so cross-channel comparisons see the same convention.
        def rt(text, options, pole):
            slot, miss = decode_state_slot(text, options)
            return (0 if miss else slot ^ int(pole)), miss
        return rt, RT_DECODER_VERSION
    if style.startswith("state_"):
        return decode_state, STATE_DECODER_VERSION
    return decode_bit, DECODER_VERSION


def _decode(style, text, options, pole):
    return _decoder_for(style)[0](text, options, pole)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def report_requests(style, masks, keys, poles, groups, priors=None):
    if style not in STYLES or len(keys) != len(poles) or not keys:
        raise ValueError('Invalid style or source bank')
    if not masks or len(set(masks)) != len(masks) or any(type(m) is not int or not 0 <= m < 16 for m in masks):
        raise ValueError('Expected unique four-world masks')
    requests = []
    prior_list = priors if priors is not None else [None]
    for source, (key, pole) in enumerate(zip(keys, poles)):
        options = groups[key[0]]['options']
        for prior in prior_list:
            for mask in masks:
                req = dict(source=source, source_key=list(key), mask=mask,
                           style=style, options=options, pole=int(pole),
                           prompt=prompt_for(style, options, int(pole), mask,
                                             prior_mask=prior))
                if prior is not None:
                    req['prior'] = int(prior)
                requests.append(req)
    return requests


def source_pairs(keys):
    """Ordered unique (group, pole) pairs plus the pair index of each source."""
    pairs, seen, index = [], {}, []
    for k in keys:
        p = (int(k[0]), int(k[3]))
        if p not in seen:
            seen[p] = len(pairs)
            pairs.append(p)
        index.append(seen[p])
    return pairs, np.asarray(index)


def rt_report_requests(style, masks, pairs, groups):
    """RoundTrip requests over unique (group, pole) pairs, keyed by `out`.

    Sources sharing a pair get identical prompts and are generated once. One
    request per (pair, prior, mask, out) with out ranging over
    feasible_outs(prior, mask) — the exact bound any pick can reach, so both
    fitted-eval and on-policy-training lookups stay inside the cache.
    """
    from .roundtrip import RT_STYLES, feasible_outs, rt_prompt
    if style not in RT_STYLES or not pairs or not masks:
        raise ValueError('Invalid roundtrip style or empty bank')
    if any(type(m) is not int or not 0 <= m < 16 for m in masks):
        raise ValueError('Expected four-world masks')
    requests = []
    for pair, (gi, pole) in enumerate(pairs):
        options = groups[gi]['options']
        for prior in range(16):
            for mask in masks:
                for out in feasible_outs(prior, mask):
                    requests.append(dict(
                        source=pair, group=int(gi), pole=int(pole),
                        prior=int(prior), mask=int(mask), out=int(out),
                        style=style, options=options,
                        prompt=rt_prompt(style, options, int(pole),
                                         prior, mask, out)))
    return requests


def build_rt_matrix(host, cfg, keys, poles, groups, output_dir, cache_dir, binding):
    """RoundTrip reports indexed by realized (source, prior, mask, out).

    Generation is deduplicated over unique (group, pole) pairs and broadcast
    back onto the full source axis. Cells outside feasible_outs stay -1 and
    are marked ungenerated; lookups there are protocol errors, not zero bits.
    """
    masks = sorted(set(cfg['fit_masks']) | set(cfg['test_masks']))
    def _rt_style(ch):
        if ch.startswith('lm_rt_'):
            return ch[3:]
        if ch.startswith('lmtext_'):
            return ch[7:]
        return None
    styles = tuple(dict.fromkeys(s for s in (_rt_style(a['channel'])
                                             for a in cfg['arms']) if s))
    n = len(keys)
    pairs, index = source_pairs(keys)
    from .roundtrip import TEXT_FEAT_DIM, TEXT_FEAT_VERSION, text_features
    bits, missing, offoff, generated = {}, {}, {}, {}
    feats, record_counts = {}, {}
    output_dir = Path(output_dir)
    for style in styles:
        requests = rt_report_requests(style, masks, pairs, groups)
        records, receipt = cached_reports(host, requests, binding, cache_dir, output_dir)
        pb = torch.full((len(pairs), 16, 16, 16), -1, dtype=torch.long)
        pm = np.full((len(pairs), 16, 16, 16), np.nan)
        po = np.full((len(pairs), 16, 16, 16), np.nan)
        pg = np.zeros((len(pairs), 16, 16, 16), bool)
        pf = torch.full((len(pairs), 16, 16, 16, TEXT_FEAT_DIM), float('nan'))
        with (output_dir / f'reports_{style}.jsonl').open('x', encoding='utf-8') as fh:
            for row in records:
                p, prior, m, o = row['source'], row['prior'], row['mask'], row['out']
                gi, pole = pairs[p]
                bit, miss = _decode(style, row['text'], groups[gi]['options'], pole)
                pb[p, prior, m, o] = bit
                pm[p, prior, m, o] = miss
                po[p, prior, m, o] = mentions_offoff(row['text'])
                pg[p, prior, m, o] = True
                pf[p, prior, m, o] = text_features(row['text'])
                fh.write(json.dumps(dict(row, bit=bit, missing=miss,
                                         decoder_version=_decoder_for(style)[1]),
                                    ensure_ascii=False) + '\n')
        with (output_dir / f'reports_{style}.binding.json').open('x', encoding='utf-8') as fh:
            json.dump(dict(receipt, decoder_version=_decoder_for(style)[1],
                           text_feature_version=TEXT_FEAT_VERSION,
                           pair_dedup=True, requests=requests), fh,
                      sort_keys=True, ensure_ascii=False, allow_nan=False)
        bits[style] = pb[torch.from_numpy(index)]
        missing[style] = pm[index]
        offoff[style] = po[index]
        generated[style] = pg[index]
        feats[style] = pf[torch.from_numpy(index)]
        record_counts[style] = len(records)
    return bits, missing, offoff, generated, feats, record_counts


def validate_records(records, requests):
    expected = {(r['source'], r['mask'], r.get('prior'), r.get('out')): r
                for r in requests}
    if len(expected) != len(requests) or len(records) != len(requests):
        raise ValueError('Report cache is incomplete or has extra records')
    seen = set()
    for row in records:
        key = (row.get('source'), row.get('mask'), row.get('prior'), row.get('out'))
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
                for field in ('prior', 'out'):
                    if field in req:
                        row[field] = req[field]
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


def build_report_matrix(host, cfg, keys, poles, groups, output_dir, cache_dir, binding, n_fit=0):
    masks = cfg['fit_masks'] + cfg['test_masks']
    styles = tuple(dict.fromkeys(a['channel'][3:] for a in cfg['arms']
                                 if a['channel'].startswith('lm_')
                                 and not a['channel'].startswith('lm_rt_')))
    n = len(keys)
    bits, missing, offoff, generated = {}, {}, {}, {}
    output_dir = Path(output_dir)
    for style in styles:
        priors = list(range(16)) if style == 'state_prior_true' else None
        # Prior-conditioned reports are eval-time-only: fit rows use the
        # perfect convention bit (mask&1) so generation stays bounded.
        if priors:
            requests = report_requests(style, masks, keys[n_fit:], poles[n_fit:],
                                       groups, priors=priors)
            for r in requests:
                r['source'] += n_fit
        else:
            requests = report_requests(style, masks, keys, poles, groups)
        records, receipt = cached_reports(host, requests, binding, cache_dir, output_dir)
        dims = (n, 16, 16) if priors else (n, 16)
        bits[style] = torch.full(dims, -1, dtype=torch.long)
        missing[style] = np.full(dims, np.nan)
        offoff[style] = np.full(dims, np.nan)
        generated[style] = np.ones(dims, bool)
        with (output_dir / f'reports_{style}.jsonl').open('x', encoding='utf-8') as fh:
            for row in records:
                i, m = row['source'], row['mask']
                options = groups[keys[i][0]]['options']
                bit, miss = _decode(style, row['text'], options, int(poles[i]))
                idx = (i, row['prior'], m) if priors else (i, m)
                bits[style][idx] = bit
                missing[style][idx] = miss
                offoff[style][idx] = mentions_offoff(row['text'])
                # Redecode the text; never copy stale stored bit/missing annotations.
                fh.write(json.dumps(dict(row, bit=bit, missing=miss,
                                         decoder_version=_decoder_for(style)[1]),
                                    ensure_ascii=False) + '\n')
        with (output_dir / f'reports_{style}.binding.json').open('x', encoding='utf-8') as fh:
            json.dump(dict(receipt, decoder_version=_decoder_for(style)[1], requests=requests), fh,
                      sort_keys=True, ensure_ascii=False, allow_nan=False)
        if priors and n_fit:
            # Eval-time-only style: fit rows never generated -> perfect
            # convention fill (bit = current mask bit0), marked synthetic.
            conv = (torch.arange(16) & 1)
            bits[style][:n_fit] = conv.view(1, 1, 16)
            missing[style][:n_fit] = 0.0
            offoff[style][:n_fit] = 0.0
            generated[style] = np.zeros(dims, bool)
            generated[style][n_fit:] = True
    return bits, missing, offoff, generated


def save_report_matrices(output_dir, masks, bits, missing, offoff,
                         generated=None, record_counts=None):
    """Save every active style; rates exclude unsampled masks and synthetic rows.

    Matrix axis semantics differ per style: 2D is [source, mask], 3D is
    [source, prior, mask], 4D (rt) is [source, prior, instruction, output].
    Per-mask panels always select the instruction axis — for 4D that is
    axis 2, keeping every feasible output cell — so collapsed outputs are
    never filtered out. `record_counts` reports unique generated records,
    distinct from broadcast per-source cells.
    """
    output_dir = Path(output_dir)
    if not masks or len(set(masks)) != len(masks) or any(not 0 <= m < 16 for m in masks):
        raise ValueError('Invalid sampled mask set')
    if not bits or set(bits) != set(missing) or set(bits) != set(offoff):
        raise ValueError('Report matrix styles disagree')
    if generated is None:
        generated = {s: np.ones(tuple(b.shape), bool) for s, b in bits.items()}

    def _mask_sel(a, g):
        # (view, gen_view): instruction-mask panels for validation/stats.
        if a.ndim == 4:
            return a[:, :, masks, :], g[:, :, masks, :]
        return a[..., masks], g[..., masks]

    for style, matrix in bits.items():
        if matrix.ndim not in (2, 3, 4) or matrix.shape[-1] != 16 or matrix.shape[0] == 0:
            raise ValueError('Expected a nonempty [sources,...,16] report matrix')
        gen = generated[style]
        if gen.shape != tuple(matrix.shape):
            raise ValueError('Report matrix and generation mask disagree')
        view, genv = _mask_sel(matrix, gen)
        if not bool(((view == 0) | (view == 1) | ~torch.from_numpy(genv)).all()):
            raise ValueError('Unobserved report bit in sampled masks')
        for observation in (missing[style], offoff[style]):
            if observation.shape != tuple(matrix.shape):
                raise ValueError('Incomplete report statistics')
            _, genv = _mask_sel(observation, gen)
            if not (np.isin(_, [0, 1]) | ~genv).all():
                raise ValueError('Incomplete report statistics')
    path = output_dir / 'lm_matrix.npz'
    with path.open('xb') as fh:
        np.savez_compressed(fh, **{s: b.cpu().numpy() for s, b in bits.items()},
                            evaluated_masks=np.asarray(masks, dtype=np.int32))

    def _per_mask(a, g):
        # Generated cells only; panel selects the instruction axis.
        out = {}
        for m in masks:
            sel = g[:, :, m, :] if a.ndim == 4 else g[..., m]
            vals = a[:, :, m, :][sel] if a.ndim == 4 else a[..., m][sel]
            out[str(m)] = float(np.nanmean(vals)) if len(vals) else None
        return out

    def _rate(a, g):
        view, genv = _mask_sel(a, g)
        vals = view[genv]
        return float(np.nanmean(vals)) if len(vals) else None

    cell_counts = {}
    for s, a in bits.items():
        _, genv = _mask_sel(a, generated[s])
        cell_counts[s] = int(genv.sum())
    summary = dict(schema='lm_report_matrix_v3', decoder_version=DECODER_VERSION,
                   evaluated_masks=masks,
                   report_counts={s: int((record_counts or {}).get(s, cell_counts[s]))
                                  for s in generated},
                   broadcast_cells=cell_counts,
                   missing={s: _per_mask(a, generated[s]) for s, a in missing.items()},
                   offoff_rate={s: _rate(a, generated[s]) for s, a in offoff.items()})
    with (output_dir / 'lm_matrix.json').open('x', encoding='utf-8') as fh:
        json.dump(summary, fh, sort_keys=True, allow_nan=False)
    return summary
