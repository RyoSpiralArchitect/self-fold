"""Question selectivity, irreversible collapse and causal reference recurrence."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from selffold.actual_report import (continuation_probe, report_summary, requests,
                                   score_report)
from selffold.lm_writer_reports import fingerprint
from selffold.reference_retention import RetentionCell, rollout, summarize, training_loss
from selffold.role_reference import ReferenceReader, source_reference
from selffold.roundtrip import feasible_outs, text_features

ROOT = Path(__file__).resolve().parents[1]


def design():
    return json.loads((ROOT/'data/actual_report_20261111.json').read_text())


def reports(reqs, censored=False):
    result = []
    for q in reqs:
        target = q['out'] if q['question'] == 'actual' else q['mask']
        text = q['options'][(target & 1) ^ q['pole']]
        result.append(dict(id=q['id'],request_sha256=fingerprint(q),text=text,
            generation=dict(stop='length' if censored else 'eos',token_count=48 if censored else 3,
                            censored=censored)))
    return result


def fixture():
    options = ['alpha', 'beta']
    keys = [(0,0,0,p,0) for p in (0,1)]
    members = torch.stack([torch.eye(16)[[p,1-p]] for p in (0,1)])
    reader = ReferenceReader(17,1,32).requires_grad_(False)
    features = torch.stack([text_features(t) for t in options])
    with torch.no_grad():
        reader.address[:] = torch.tensor([[-1.,1.],[1.,-1.]])
        reader.feat_addr[:] = torch.linalg.pinv(features) @ torch.tensor([[-1.,1.],[1.,-1.]])
    bank = torch.empty(2,16,16,16,32)
    for p in (0,1):
        bank[p] = features[(torch.arange(16)&1)^p]
    labels = np.array([101,202])
    top = labels[np.arange(2)[:,None,None] ^ np.arange(2)[None,:,None] ^ np.arange(2)]
    return options,keys,members,reader,bank,labels,top


def test_matched_questions_cover_the_complete_reachable_state_closure():
    d=design()
    groups=json.loads((ROOT/d['actions']).read_text())['groups']
    panel=requests(d,groups)
    assert len(panel)==448 and len({q['id'] for q in panel})==448
    assert len({q['pair'] for q in panel})==224
    assert set(d['masks']).isdisjoint(json.loads((ROOT/d['base_config']).read_text())['fit_masks'])
    for a,b in zip(panel[::2],panel[1::2]):
        assert a['pair']==b['pair'] and a['question']!=b['question']
        assert a['prompt'].split('Question:')[0]==b['prompt'].split('Question:')[0]
    assert {q['out'] for q in panel} <= set(d['priors'])
    for topic,pole,prior,mask in itertools.product(d['topics'],(0,1),d['priors'],d['masks']):
        cells=[q for q in panel if (q['topic'],q['pole'],q['prior'],q['mask'])==(topic,pole,prior,mask)]
        assert {q['out'] for q in cells}==set(feasible_outs(prior,mask))


def test_report_scoring_keeps_requested_truth_censoring_and_errors_separate():
    d=design(); d['topics']=[0]
    reqs=requests(d,[dict(options=['alpha','beta'])]); rows=reports(reqs,censored=True)
    summary,scored=report_summary(reqs,rows,d)
    assert summary['gate']['passed']
    assert summary['actual']['all']['censored']==[56,56]
    assert summary['paired_question_selectivity']==[28,28]
    r=next(r for r in scored if r['question']=='actual' and r['coordinate_conflict'])
    assert r['actual_correct'] and not r['instruction_correct']
    changed=rows[0]|dict(text='')
    assert score_report(reqs[0],changed)['missing']
    with pytest.raises(ValueError):
        score_report(reqs[0],changed|dict(request_sha256='foreign'))
    with pytest.raises(ValueError):
        report_summary(reqs,rows[:-1]+[rows[0]],d)


def test_always_reporting_output_does_not_pass_question_selectivity():
    d=design(); d['topics']=[0]
    reqs=requests(d,[dict(options=['alpha','beta'])]); rows=reports(reqs)
    for q,r in zip(reqs,rows):
        r['text']=q['options'][(q['out']&1)^q['pole']]
    summary,_=report_summary(reqs,rows,d)
    assert summary['gate']['checks']['actual_all']
    assert not summary['gate']['checks']['question_selectivity']
    assert not summary['gate']['passed']


def test_faithful_reports_repair_reversed_tables_but_not_lost_actions(monkeypatch):
    options,keys,members,reader,_,labels,top=fixture()
    d=design(); d.update(topics=[0],fit_topics=[0],seeds=[17])
    reqs=requests(d,[dict(options=options)])
    monkeypatch.setattr('selffold.actual_report.frozen_reader',lambda *a:reader)
    archive=dict(keys=keys,fit_keys=[],members=members,top=top,labels=labels)
    summary,_=continuation_probe(ROOT,reqs,reports(reqs),d,archive)
    row=summary['seeds']['17']
    assert row['actual_generated_primary_noncollapsed']==[48,48]
    assert row['instruction_generated_primary_noncollapsed']==[24,48]
    assert row['actual_generated_primary_collapsed']==[0,64]
    assert row['actual_oracle_primary_collapsed']==[0,64]


def test_cells_share_initial_bit_coordinates_and_use_only_two_or_34_parameters():
    short,full=RetentionCell(41),RetentionCell(41,True)
    assert torch.equal(short.reference,full.reference)
    assert sum(p.numel() for p in short.parameters())==2
    assert sum(p.numel() for p in full.parameters())==34
    with pytest.raises(ValueError):
        short(torch.zeros(1,32),torch.tensor([2]))


def test_reference_recurrence_uses_the_previous_emission_without_source_refresh():
    _,keys,members,reader,bank,labels,top=fixture()
    ref=source_reference(members); masks=[6,9]
    identity=RetentionCell(17); flip=RetentionCell(17)
    with torch.no_grad():
        identity.reference[:]=torch.tensor([5.,0.])
        flip.reference[:]=torch.tensor([-5.,0.])
    copy=rollout(reader,None,members,ref,masks,bank)
    kept=rollout(reader,identity,members,ref,masks,bank)
    opposed=rollout(reader,identity,members,ref,masks,bank,'opposed')
    toggled=rollout(reader,flip,members,ref,masks,bank)
    assert all(np.array_equal(copy[k],kept[k]) for k in copy)
    target=np.asarray(masks)[kept['paths']]
    assert (kept['depths']==target[None]).all()
    assert (opposed['depths'][:,:,1:]==(15-target[None,:,1:])).all()
    assert (opposed['references']==(1-ref.numpy())[:,None,None]).all()
    for t in range(3):
        assert (toggled['references'][:,:,t]==(ref.numpy()^((t+1)%2))[:,None]).all()
    cfg=dict(fit_topics=[0],fit_masks=[],test_masks=masks)
    assert summarize(kept,keys,cfg,top,labels)['primary_fit_menus']['all_four_correct']==[32,32]
    assert summarize(opposed,keys,cfg,top,labels)['primary_fit_menus']['all_four_correct']==[0,32]


def test_native_reward_supplies_a_copy_gradient_without_updating_the_reader():
    _,_,members,reader,bank,_,_=fixture()
    cell=RetentionCell(17)
    with torch.no_grad(): cell.reference.zero_()
    cases=torch.tensor([0,1]*4)
    ms=torch.tensor([[6,9]]*8)
    q=source_reference(members)[cases]
    uniform=torch.tensor([.1,.1,.9,.9]*2)
    def reward(descriptor,mask):
        target=(mask[:,None]>>torch.arange(4))&1
        return (descriptor==target).float()[...,None].expand(-1,-1,2)
    loss,value=training_loss(cell,reader,members[cases],q,cases,ms,uniform,bank,reward)
    loss.backward()
    assert value==.5 and cell.reference.grad[0]<0
    assert all(p.grad is None for p in reader.parameters())
