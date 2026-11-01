#!/usr/bin/env python3
"""Make opaque review rows. Keep the key private and assign whole groups across raters."""
import argparse, json, random
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("transitions", nargs="+")
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=20261101)
a=p.parse_args()
out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
rows=[]
for filename in a.transitions:
    for line in Path(filename).read_text().splitlines():
        r=json.loads(line)
        obs=r.get("observation",{})
        rows.append((filename,r,obs))
random.Random(a.seed).shuffle(rows)
with (out/"review.jsonl").open("w") as f, (out/"private_key.jsonl").open("w") as k:
    for i,(source,r,o) in enumerate(rows):
        identity=f"review-{i:05d}"
        f.write(json.dumps({"id":identity,"current_description":o.get("text"),"censored":o.get("censored"),
                            "supported_distinction":None,"event_recovery":None,"unsupported_leap":None,
                            "human_note":""},ensure_ascii=False)+"\n")
        k.write(json.dumps({"id":identity,"source":source,"arm":r["arm"],"step":r["step"],
                            "regime":r["regime"],"state_sha256":r["state_sha256"]})+"\n")
print("Single-state review only. Transition support requires paired before/after readouts; see rubric.")
