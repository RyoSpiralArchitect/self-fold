from pathlib import Path
import json
import subprocess
import sys
import torch
import pytest
from selffold.records import State, RunStore, load_state
from selffold.host import SlotHost
from selffold.tiny import TinyCausalLM, ByteTokenizer

@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32,torch.float64])
def test_safe_exact_state_restore(tmp_path,dtype):
    store=RunStore(tmp_path/"run")
    s=State(slots=torch.tensor([[0.3141592653589793,2.],[-7.25,0.001]],dtype=dtype))
    store.save_state("state",s)
    r=load_state(store.path/"state")
    assert r.sha()==s.sha()
    assert r.slots.data_ptr()!=s.slots.data_ptr()

def test_text_restore_and_tamper(tmp_path):
    store=RunStore(tmp_path/"run");s=State(text="I value evidence.")
    store.save_state("state",s)
    assert load_state(store.path/"state").sha()==s.sha()
    p=store.path/"state.json";m=json.loads(p.read_text());m['text']='altered';p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='fingerprint'):load_state(store.path/"state")

def test_tensor_file_tamper(tmp_path):
    store=RunStore(tmp_path/"run");store.save_state("state",State(slots=torch.ones(2,3)))
    p=store.path/"state.npz";p.write_bytes(p.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='SHA'):load_state(store.path/"state")

def test_new_process_replay(tmp_path):
    store=RunStore(tmp_path/"run")
    h=SlotHost(TinyCausalLM(seed=47),ByteTokenizer(),slots=3,max_new_tokens=2)
    s=h.encode("I value evidence.");store.save_state("state",s)
    expected=h.update_latent(s,"plain").sha()
    code='import sys,torch\nfrom pathlib import Path\nfrom selffold.records import load_state\nfrom selffold.host import SlotHost\nfrom selffold.tiny import TinyCausalLM,ByteTokenizer\ntorch.set_num_threads(1)\nh=SlotHost(TinyCausalLM(seed=47),ByteTokenizer(),slots=3,max_new_tokens=2)\nprint(h.update_latent(load_state(Path(sys.argv[1])),"plain").sha())\n'
    actual=subprocess.check_output([sys.executable,"-c",code,str(store.path/"state")],text=True).strip()
    assert actual==expected
