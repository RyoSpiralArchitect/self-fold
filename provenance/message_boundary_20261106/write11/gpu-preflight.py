import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from selffold.cli import make_host
from selffold.lm_writer_reports import generate_kv

torch.set_default_device('cpu')
out = Path('/home/ryospiralarchitect/scratch/self-fold-source-selection-20261111/gpu-preflight.json')
assert not out.exists()
args = SimpleNamespace(backend='hf', profile='configs/models/gemma2b.yaml',
    checkpoint='/home/ryospiralarchitect/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8',
    device='cuda', dtype='bfloat16', cpu_threads=1, slots=8, cut=None,
    writer=None, max_context=1024, max_new_tokens=8, wall_seconds=300,
    reviewed_local_code=None, transport=None, encoder_bridge=None)
start = time.monotonic()
host, identity = make_host(args)
assert identity['checkpoint_sha256'] == 'f3503907cc4d2ace62abccf30393a05919827c1c8969b2f2755f15627b9756ad'
loaded = time.monotonic()
parity = host.native_parity()
assert parity['exact']
rows = []
for prompt in ['Answer with the single word ready.',
               'This is a timing check. ' * 40 + 'Answer with the single word ready.']:
    t = time.monotonic()
    result = generate_kv(host, prompt)
    rows.append(dict(seconds=time.monotonic()-t, **result.receipt()))
receipt = dict(purpose='unscored GPU resource preflight; no policy-bank prompts',
    load_seconds=loaded-start, dtype='bfloat16', device='cuda', cpu_threads=1,
    checkpoint_sha256=identity['checkpoint_sha256'], parity=parity,
    generations=rows, elapsed_seconds=time.monotonic()-start)
out.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt), flush=True)
