#!/usr/bin/env python3
"""Create a PENDING code-review inventory. Does not import or execute model Python."""
import argparse,json
from pathlib import Path
from selffold.loading import local_identity
from selffold.records import file_sha
p=argparse.ArgumentParser();p.add_argument("checkpoint",type=Path);p.add_argument("out",type=Path)
a=p.parse_args();identity=local_identity(a.checkpoint)
record={"status":"PENDING_REVIEW_DO_NOT_LOAD","reviewer":None,
        "checkpoint_sha256":identity["checkpoint_sha256"],
        "python_files":{str(x.relative_to(a.checkpoint)):file_sha(x) for x in sorted(a.checkpoint.rglob("*.py"))
                        if x.is_file() and ".cache" not in x.parts},
        "review_notes":"Review all imports, initialization and forward paths, compatibility and source identity. This is NOT an approval."}
with a.out.open("x") as f:json.dump(record,f,indent=2);f.write("\n")
