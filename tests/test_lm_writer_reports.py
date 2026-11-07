"""No model downloads: report/cache contracts and lexical decoding regressions."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest

from selffold.lm_writer import decode_bit, mentions_offoff, prompt_for
from selffold.lm_writer_reports import (
    build_report_matrix, cached_reports, fingerprint, report_requests,
    save_report_matrices, validate_records,
)


@dataclass
class Output:
    text: str = 'Go west.'

    def receipt(self):
        return dict(text=self.text, stop='eos', token_count=3, prompt_tokens=30,
                    effective_max_new_tokens=48, stop_token_id=107, censored=False)


class Host:
    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def generate(self, prompt):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('interrupted')
        return Output()


def fixture(style='directed'):
    return report_requests(style, [6, 9], [(0, 0, 0, 0, 0)], [0],
                           [{'options': ['go west', 'go east']}])


def dirs(tmp_path):
    out = tmp_path / 'run'
    out.mkdir()
    return tmp_path / 'cache', out


@pytest.mark.parametrize('text, expected', [
    ('GO WEST.', (0, 0)), ('go east', (1, 0)), ('least effort', (0, 1)),
    ('The feast was fine.', (0, 1)), ('westbound', (0, 1)), ('', (0, 1)),
])
def test_word_boundaries_and_case(text, expected):
    assert decode_bit(text, ['go west', 'go east'], 0) == expected


def test_bare_action_verb_remains_readable():
    assert decode_bit('Summarize.', ['summarize the notes', 'preserve verbatim notes'], 0) == (0, 0)
    assert decode_bit('preserve', ['summarize the notes', 'preserve verbatim notes'], 0) == (1, 0)


@pytest.mark.parametrize('text', ['OFF/OFF', 'off-off', 'off / off'])
def test_coordinate_matching(text):
    assert mentions_offoff(text)
    assert not mentions_offoff('on/off')


def test_prompt_unknown_style_rejected():
    with pytest.raises(ValueError):
        prompt_for('typo', ['go west', 'go east'], 0, 6)


def test_cache_reuse_requires_exact_binding(tmp_path):
    cache, out = dirs(tmp_path)
    host = Host()
    req = fixture()
    rows, receipt = cached_reports(host, req, {'checkpoint': 'one', 'cap': 48}, cache, out)
    assert host.calls == 2 and receipt['mode'] == 'generated'
    again, reused = cached_reports(host, req, {'checkpoint': 'one', 'cap': 48}, cache, out)
    assert again == rows and reused['mode'] == 'cache_reuse' and host.calls == 2
    second = tmp_path / 'second'
    second.mkdir()
    _, changed = cached_reports(host, req, {'checkpoint': 'two', 'cap': 48}, cache, second)
    assert changed['contract_sha256'] != receipt['contract_sha256'] and host.calls == 4


def test_changed_prompt_or_source_mapping_has_different_key(tmp_path):
    cache, out = dirs(tmp_path)
    req = fixture()
    _, old = cached_reports(Host(), req, {}, cache, out)
    req[0]['prompt'] += '\nA different question.'
    other = tmp_path / 'other'
    other.mkdir()
    _, new = cached_reports(Host(), req, {}, cache, other)
    assert new['contract_sha256'] != old['contract_sha256']


def test_interrupted_generation_is_never_published_as_cache(tmp_path):
    cache, out = dirs(tmp_path)
    with pytest.raises(RuntimeError):
        cached_reports(Host(fail_at=2), fixture(), {}, cache, out)
    assert not list(cache.glob('*.json'))
    partial = list(out.glob('*.partial.jsonl'))
    assert len(partial) == 1 and len(partial[0].read_text().splitlines()) == 1


@pytest.mark.parametrize('mode', ['truncated', 'duplicate', 'foreign', 'wrong_prompt', 'no_receipt'])
def test_bad_cache_records_fail_closed(tmp_path, mode):
    cache, out = dirs(tmp_path)
    rows, _ = cached_reports(Host(), fixture(), {}, cache, out)
    if mode == 'truncated':
        rows.pop()
    elif mode == 'duplicate':
        rows[1] = dict(rows[0])
    elif mode == 'foreign':
        rows[0]['source'] = 99
    elif mode == 'wrong_prompt':
        rows[0]['request_sha256'] = 'wrong'
    else:
        rows[0].pop('generation')
    with pytest.raises(ValueError):
        validate_records(rows, fixture())


def test_cache_hash_tampering_is_detected(tmp_path):
    cache, out = dirs(tmp_path)
    cached_reports(Host(), fixture(), {}, cache, out)
    path = next(cache.glob('*.json'))
    payload = json.loads(path.read_text())
    payload['records'][0]['text'] = 'go east'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='hash'):
        cached_reports(Host(), fixture(), {}, cache, out)


def test_all_styles_saved_and_rates_use_only_generated_masks(tmp_path):
    cache, out = dirs(tmp_path)
    cfg = dict(fit_masks=[6], test_masks=[9], arms=[
        dict(channel='lm_' + s) for s in ('directed', 'natural', 'guided')])
    result = build_report_matrix(Host(), cfg, [(0,0,0,0,0)], [0],
                                 [{'options': ['go west', 'go east']}], out, cache, {})
    bits, missing, offoff, generated = result
    for array in offoff.values():
        array[:, [6,9]] = 1
    summary = save_report_matrices(out, [6,9], *result)
    assert summary['offoff_rate'] == dict(directed=1., natural=1., guided=1.)
    assert summary['report_counts'] == dict(directed=2, natural=2, guided=2)
    with np.load(out/'lm_matrix.npz', allow_pickle=False) as saved:
        assert set(saved.files) == {'directed','natural','guided','evaluated_masks'}
        assert (saved['guided'][:, [0,15]] == -1).all()
    # Raw evidence has a stop reason and the decoder actually used on this run.
    row = json.loads((out/'reports_guided.jsonl').read_text().splitlines()[0])
    assert row['generation']['stop'] == 'eos' and row['decoder_version'] == 'distinctive_tokens_v3'


def test_unused_masks_cannot_silently_become_bit_zero(tmp_path):
    cache, out = dirs(tmp_path)
    cfg = dict(fit_masks=[6], test_masks=[9], arms=[dict(channel='lm_directed')])
    result = build_report_matrix(Host(), cfg, [(0,0,0,0,0)], [0],
                                 [{'options': ['go west', 'go east']}], out, cache, {})
    with pytest.raises(ValueError):
        save_report_matrices(out, [6,9,0], *result)


def test_prior_style_separates_synthetic_fit_from_generated_eval(tmp_path):
    cache, out = dirs(tmp_path)
    cfg = dict(fit_masks=[6], test_masks=[9], arms=[dict(channel='lm_state_prior_true')])
    keys = [(0, 0, 0, 0, 0), (1, 0, 0, 0, 1)]  # one synthetic fit + one generated eval
    bits, missing, offoff, generated = build_report_matrix(
        Host(), cfg, keys, [0, 1], [{'options': ['go west', 'go east']},
                                    {'options': ['go west', 'go east']}],
        out, cache, {}, n_fit=1)
    assert bits['state_prior_true'].shape == (2, 16, 16)
    assert not generated['state_prior_true'][0].any()       # fit row synthetic
    assert generated['state_prior_true'][1].all()           # eval row generated
    summary = save_report_matrices(out, [6, 9], bits, missing, offoff, generated)
    # 1 generated source x 16 priors x 2 masks = 32, not len(matrix)*2
    assert summary['report_counts'] == dict(state_prior_true=32)


def test_duplicate_masks_rejected():
    with pytest.raises(ValueError):
        report_requests('directed', [6,6], [(0,0,0,0,0)], [0], [{'options':['a','b']}])
