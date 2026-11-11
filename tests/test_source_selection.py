"""Guard source-cue attribution, order controls, and the frozen interface."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from selffold.lm_writer_reports import fingerprint
from selffold.reference_retention import RetentionCell
from selffold.role_reference import ReferenceReader
from selffold.roundtrip import feasible_outs, text_features
from selffold.source_selection import combined_gate, compose, probe, requests, score_panel

ROOT = Path(__file__).resolve().parents[1]


def design(small=False):
    cfg = json.loads((ROOT/'data/source_selection_20261111.json').read_text())
    if small:
        cfg.update(topics=[0],fit_topics=[0],seeds=[17],generation_calls=448,composition_evaluations=72)
    return cfg


def perfect(reqs, capped=False):
    result = []
    for q in reqs:
        target = q['out'] if q['question']=='actual' else q['mask']
        result.append(dict(id=q['id'],request_sha256=fingerprint(q),text=q['options'][(target&1)^q['pole']],
            generation=dict(stop='length' if capped else 'eos',censored=capped,token_count=48 if capped else 3)))
    return result


def test_panel_is_matched_closed_order_balanced_and_disjoint_from_write10():
    d=design(); groups=json.loads((ROOT/d['actions']).read_text())['groups']
    reqs=requests(d,groups)
    assert len(reqs)==len({q['id'] for q in reqs})==1792
    assert len({q['transition'] for q in reqs})==224
    old=json.loads((ROOT/d['previous_report_run']/'requests.json').read_text())
    key=lambda q:(q['topic'],q['pole'],q['prior'],q['mask'],q['out'])
    assert {key(q) for q in reqs}.isdisjoint({key(q) for q in old})
    assert {q['out'] for q in reqs} <= set(d['priors'])
    for topic,pole,prior,mask in itertools.product(d['topics'],(0,1),d['priors'],d['masks']):
        subset=[q for q in reqs if (q['topic'],q['pole'],q['prior'],q['mask'])==(topic,pole,prior,mask)]
        assert {q['out'] for q in subset}==set(feasible_outs(prior,mask))
    for transition in {q['transition'] for q in reqs}:
        matched=[q for q in reqs if q['transition']==transition]
        assert len(matched)==8
        for layout in d['layouts']:
            cells=[q for q in matched if q['layout']==layout]
            assert len({q['prompt'].split('Question:')[0] for q in cells})==1
            for q in cells:
                prefix=q['prompt'].split('Question:')[0]
                assert (prefix.index('INSTRUCTION RECORD:')<prefix.index('OBSERVATION RECORD')) == layout.startswith('instruction')
                if q['arm']=='source_cue':
                    cue=q['prompt'].split('Requested record: ')[1].split('\n')[0]
                    assert cue.startswith('INSTRUCTION RECORD' if q['question']=='instruction' else 'OBSERVATION RECORD')
                    assert all(option not in cue for option in q['options'])


def test_perfect_reporting_passes_admission_without_inventing_a_cue_gain():
    d=design(True); reqs=requests(d,[dict(options=['alpha','beta'])])
    summary,scored=score_panel(reqs,perfect(reqs,capped=True),d)
    assert all(v['passed'] for v in summary['admission'].values())
    assert not summary['matched_benefit']['passed']
    assert summary['matched_benefit']['difference']==0
    assert summary['admission']['source_cue']['joint_order_selectivity']==[28,28]
    assert all(s['censored'] for s in scored)


@pytest.mark.parametrize('heuristic',['observation','last_record'])
def test_always_reading_observation_or_the_last_record_fails(heuristic):
    d=design(True); reqs=requests(d,[dict(options=['alpha','beta'])]); rows=perfect(reqs)
    for q,r in zip(reqs,rows):
        pattern=q['out'] if heuristic=='observation' or q['layout'].startswith('instruction') else q['mask']
        r['text']=q['options'][(pattern&1)^q['pole']]
    summary,_=score_panel(reqs,rows,d)
    assert all(not v['passed'] for v in summary['admission'].values())
    assert summary['admission']['source_cue']['joint_order_selectivity']==[0,28]


def test_attributable_benefit_requires_the_matched_control_and_rejects_foreign_rows():
    d=design(True); reqs=requests(d,[dict(options=['alpha','beta'])]); rows=perfect(reqs)
    for q,r in zip(reqs,rows):
        if q['arm']=='copy_only': r['text']=q['options'][(q['out']&1)^q['pole']]
    summary,_=score_panel(reqs,rows,d)
    assert summary['admission']['source_cue']['passed']
    assert summary['matched_benefit']['passed'] and summary['matched_benefit']['difference']==1
    with pytest.raises(ValueError): score_panel(reqs,rows[:-1]+[rows[0]],d)
    with pytest.raises(ValueError): score_panel(reqs,[rows[0]|dict(request_sha256='changed')]+rows[1:],d)
    with pytest.raises(ValueError): score_panel(reqs,[rows[0]|dict(id='foreign')]+rows[1:],d)


def test_one_order_cannot_substitute_for_correct_selection_in_both_orders():
    d=design(True); reqs=requests(d,[dict(options=['alpha','beta'])]); rows=perfect(reqs)
    for q,r in zip(reqs,rows):
        if q['layout']=='observation_then_instruction':
            target=q['mask'] if q['question']=='actual' else q['out']
            r['text']=q['options'][(target&1)^q['pole']]
    summary,_=score_panel(reqs,rows,d)
    assert summary['arms']['source_cue']['instruction_then_observation']['gate']['passed']
    assert not summary['admission']['source_cue']['passed']
    assert summary['admission']['source_cue']['joint_order_selectivity']==[0,28]


def test_frozen_composition_covers_every_variant_and_preserves_interventions(monkeypatch):
    d=design(True); opts=['alpha','beta']; reqs=requests(d,[dict(options=opts)]); rows=perfect(reqs)
    reader=ReferenceReader(17,1,32).requires_grad_(False)
    with torch.no_grad():
        reader.address[:]=torch.tensor([[-1.,1.],[1.,-1.]])
        reader.feat_addr[:]=torch.linalg.pinv(torch.stack([text_features(x) for x in opts])) @ torch.tensor([[-1.,1.],[1.,-1.]])
    cell=RetentionCell(17).requires_grad_(False)
    with torch.no_grad(): cell.reference[:]=torch.tensor([5.,0.])
    monkeypatch.setattr('selffold.source_selection.frozen_reader',lambda *a:reader)
    monkeypatch.setattr('selffold.actual_report.frozen_reader',lambda *a:reader)
    monkeypatch.setattr('selffold.source_selection.load_cell',lambda run,kind,seed:None if kind=='copy' else cell)
    labels=np.array([101,202])
    archive=dict(keys=[(0,0,0,p,0) for p in (0,1)],fit_keys=[],
        members=torch.stack([torch.eye(16)[[p,1-p]] for p in (0,1)]),labels=labels,
        top=labels[np.arange(2)[:,None,None]^np.arange(2)[None,:,None]^np.arange(2)])
    cfg=dict(fit_topics=[0],fit_masks=[],test_masks=d['masks'])
    result=compose(ROOT,reqs,rows,d,archive,cfg)
    assert len(result['arrays'])==72
    assert all(g['passed'] for g in result['gates'].values())
    for name,summary in result['summaries'].items():
        expected=32 if name.endswith('__intact') else 16 if name.endswith('__fixed_zero') else 0
        assert summary['primary_fit_menus']['all_four_correct']==[expected,32]
    continuation,_=probe(ROOT,reqs,rows,d,archive)
    reporting,_=score_panel(reqs,rows,d)
    gate=combined_gate(reporting,continuation,result,d)
    assert all(gate['combined'].values()) and not gate['attributable_cue_benefit']
    reporting['admission']['source_cue']['passed']=False
    assert not any(combined_gate(reporting,continuation,result,d)['combined'].values())
