"""Wait for a genuinely free GPU, then execute the already sealed study once."""
import json
from pathlib import Path
import subprocess
import sys
import time

base=Path('/home/ryospiralarchitect/scratch/self-fold-source-selection-20261111')
log=(base/'resource-wait.jsonl').open('x')
start=time.monotonic(); empty=0
while time.monotonic()-start < 900:
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
    processes=result.stdout.strip()
    row=dict(elapsed_seconds=time.monotonic()-start,compute_processes=processes)
    log.write(json.dumps(row)+'\n'); log.flush()
    empty=empty+1 if not processes else 0
    if empty >= 2:
        print(json.dumps(dict(event='gpu_admitted',wait_seconds=time.monotonic()-start)),flush=True)
        log.close()
        command=['timeout','3900',sys.executable,'scripts/measure_source_selection.py',
            '--out',str(base/'write11/produce'),'--checkpoint',
            '/home/ryospiralarchitect/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8']
        raise SystemExit(subprocess.run(command).returncode)
    time.sleep(10)
log.close()
print(json.dumps(dict(event='resource_wait_closed',scientific_generation_calls=0,wait_seconds=time.monotonic()-start)),flush=True)
raise SystemExit(75)
