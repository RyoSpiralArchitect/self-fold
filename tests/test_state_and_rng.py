import random
import numpy as np
import pytest
import torch
from selffold.records import State, RunStore, canonical, digest, file_sha
from selffold.rng import isolated_rng

@pytest.mark.parametrize("kwargs", [{},{"text":""},{"text":"x","slots":torch.zeros(2,3)},
 {"slots":torch.zeros(3)},{"slots":torch.zeros(0,3)},{"slots":torch.ones(2,3,dtype=torch.int64)},
 {"slots":torch.tensor([[float('nan')]])}])
def test_invalid_states(kwargs):
 with pytest.raises(ValueError):State(**kwargs)

def test_snapshot_owns_tensor():
 x=torch.ones(2,3); s=State(slots=x); sha=s.sha();x.zero_()
 assert s.sha()==sha
 c=s.clone();c.slots.zero_();assert s.sha()==sha

def test_fingerprint():
 assert State(text="a").sha()==State(text="a").sha()
 assert State(text="a").sha()!=State(text="b").sha()
 assert State(slots=torch.ones(2,3)).sha()!=State(slots=torch.ones(3,2)).sha()

def test_rng_isolated_on_failure():
 random.seed(1);np.random.seed(1);torch.manual_seed(1)
 py=random.getstate();np_s=np.random.get_state();t=torch.random.get_rng_state().clone()
 with pytest.raises(RuntimeError):
  with isolated_rng():
   random.random();np.random.rand();torch.rand(5);raise RuntimeError()
 assert py==random.getstate()
 assert np.array_equal(np_s[1],np.random.get_state()[1])
 assert torch.equal(t,torch.random.get_rng_state())

def test_store_never_overwrites(tmp_path):
 s=RunStore(tmp_path/"run");s.write("x.json",{"x":1})
 with pytest.raises(FileExistsError):s.write("x.json",{"x":2})
 with pytest.raises(FileExistsError):RunStore(tmp_path/"run")

def test_finite_json():
 with pytest.raises(ValueError):canonical({"x":float("nan")})

def test_tensor_storage(tmp_path):
 s=RunStore(tmp_path/"run");state=State(slots=torch.ones(3,4,dtype=torch.bfloat16))
 s.save_state("states/L_000",state)
 with np.load(tmp_path/"run/states/L_000.npz",allow_pickle=False) as z:
  assert np.array_equal(z["slots"],np.ones((3,4)))
