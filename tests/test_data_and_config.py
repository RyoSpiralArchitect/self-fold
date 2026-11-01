from pathlib import Path
import json,subprocess,sys,yaml
import pytest
ROOT=Path(__file__).resolve().parents[1]

def test_seed_matrix():
 rows=[json.loads(x) for x in (ROOT/"data/seeds.dev.jsonl").read_text().splitlines()]
 assert len(rows)==48 and len({x["id"] for x in rows})==48
 assert len({x["group"] for x in rows})==6
 assert all(x["split"]=="development" for x in rows)

def test_task_oracles():
 rows=[json.loads(x) for x in (ROOT/"data/tasks.dev.jsonl").read_text().splitlines()]
 assert len(rows)==20
 for r in rows:
  assert r["answer"] in r["labels"]
  assert "oracle" not in r["question"]
  if r["tier"]=="near":assert r["oracle"]["now"]!=r["oracle"]["wait"]

def test_data_rebuild_identical():
 paths=[ROOT/"data/seeds.dev.jsonl",ROOT/"data/tasks.dev.jsonl"]
 before=[p.read_bytes() for p in paths]
 subprocess.run([sys.executable,str(ROOT/"scripts/build_dev_data.py")],check=True)
 assert before==[p.read_bytes() for p in paths]

@pytest.mark.parametrize("name",["mistral7b","llama3b","gemma2b","phi_mini_moe","phi_tiny_moe"])
def test_profiles_not_overclaim(name):
 p=yaml.safe_load((ROOT/f"configs/models/{name}.yaml").read_text())
 assert p["local_only"] and not p["trust_remote_code"]
 assert p["pinned_revision"] is None and p["checkpoint_path"] is None
 assert "NOT_PRETRAINED_TESTED" in p["status"]
 assert p["approx_active_parameters_billions"]<=p["approx_total_parameters_billions"]

def test_example_gate_not_passing():
 g=json.loads((ROOT/"configs/qualification.example.json").read_text())
 assert g["status"]=="NOT_QUALIFIED"
