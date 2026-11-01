#!/usr/bin/env python3
"""Create a PENDING code-review inventory. Does not import or execute model Python."""
import argparse,json
from pathlib import Path
from selffold.loading import local_identity
p=argparse.ArgumentParser();p.add_argument("checkpoint",type=Path);p.add_argument("out",type=Path)
a=p.parse_args();identity=local_identity(a.checkpoint)
record={"status":"PENDING_REVIEW_DO_NOT_LOAD","reviewer":None,
        "checkpoint_sha256":identity["checkpoint_sha256"],
        "python_files":{name:sha for name,sha in identity["files"].items() if name.endswith(".py")},
        "review_notes":"Review all imports, initialization and forward paths, compatibility and source identity. This is NOT an approval."}
with a.out.open("x") as f:json.dump(record,f,indent=2);f.write("\n")
