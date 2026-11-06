"""Small full-reduction fixtures, including the pre-review tensor layout."""
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from selffold.message_boundary import BoundaryMemory
from selffold.mixture_boundary import rollout_w, twins_w
from selffold.role_trials import donor_indices

PATH = Path(__file__).resolve().parents[1] / 'provenance/message_boundary_20261106/summarize_mixture.py'
SPEC = importlib.util.spec_from_file_location('lm_writer_reduction', PATH)
reducer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reducer)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_legacy_twin_layout_inverse():
    correct = np.arange(2*3*4*2*2*4).reshape(2,3,4,2,2,4)
    legacy = np.moveaxis(correct, -1, -3).reshape(correct.shape)
    np.testing.assert_array_equal(reducer.align_twins(legacy, 'legacy_sampled_writer'), correct)
    np.testing.assert_array_equal(reducer.align_twins(correct, 'carried_message_bits_v2'), correct)


def test_ambiguous_native_labels_rejected(tmp_path):
    dump(tmp_path/'fit_native_hard/native.json', {'labels':[65,66]})
    dump(tmp_path/'fit_native_soft/native.json', {'labels':[66,65]})
    with pytest.raises(ValueError, match='disagree'):
        reducer.native_labels(tmp_path)


def test_matching_native_labels_accepted(tmp_path):
    for form in ('hard','soft'):
        dump(tmp_path/f'fit_native_{form}/native.json', {'labels':[65,66]})
    np.testing.assert_array_equal(reducer.native_labels(tmp_path), [65,66])


@pytest.mark.parametrize('values', [[.4], [float('nan')], [-1.], [2.]])
def test_hard_descriptor_corruption_rejected(values):
    with pytest.raises(ValueError):
        reducer.w_index([0.,1.], np.array(values))


@pytest.mark.parametrize('legacy', [False, True])
def test_complete_reduction_with_exact_synthetic_native_choices(tmp_path, legacy):
    actor = BoundaryMemory(17)
    with torch.no_grad():
        actor.address.copy_(torch.tensor([[-1.,1.], [1.,-1.]]))
        actor.message.zero_()
    keys = [(0,f,0,p,o) for f,p,o in itertools.product(range(2), repeat=3)]
    members = torch.eye(16)[:2].unsqueeze(0).repeat(len(keys),1,1)
    masks = [6,9]
    result = rollout_w(actor, 'hard', 'supplied', members, keys, masks)
    result['twins'] = twins_w(actor, 'hard', 'supplied', members, masks)
    if legacy:
        result['twins'] = np.moveaxis(result['twins'], -1, -3).reshape(result['twins'].shape)
        result.pop('written_semantics')
        result['written'][:] = 0  # Historical unused controller output.
    arrays = {'eval_paths': result.pop('paths')}
    arrays.update({'hard_supplied_w1_17_' + k: v for k,v in result.items()})
    np.savez_compressed(tmp_path/'trajectories.npz', **arrays)
    np.savez_compressed(tmp_path/'join.npz', donors=donor_indices(keys), eval_masks=masks)
    dump(tmp_path/'test_keys.json', {'keys':keys})
    dump(tmp_path/'fit_closed.json', {'fits':[dict(arm='hard_supplied_w1', seed=17, form='hard', beta=1.)]})
    dump(tmp_path/'fit_native_hard/native.json', {'labels':[65,66]})
    labels = np.array([65,66])
    top = np.array([[[labels[k[3]^member^order] for order in (0,1)]
                     for member in (0,1)] for k in keys])
    (tmp_path/'test_variants_hard_b100').mkdir()
    np.savez_compressed(tmp_path/'test_variants_hard_b100/variants.npz', w=[0.,1.], top=top)
    summary = reducer.summarize(tmp_path)
    arm = summary['arms']['hard_supplied_w1']['17']
    assert arm['all_four_correct'] == [128,128]
    assert arm['twins_both_histories']['intact'] == [128,128]
    assert arm['twins_both_histories']['erased'] == [0,128]
    assert arm['legacy_twin_axis_repair_applied'] is legacy
    if legacy:
        assert 'written_matches_offoff_bit' not in arm
        assert 'legacy_sampled_writer_matches_offoff_bit' in arm
    else:
        assert arm['written_matches_offoff_bit'] == {'bit0':1.}
