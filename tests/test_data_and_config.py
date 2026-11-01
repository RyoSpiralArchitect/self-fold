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
    profile = yaml.safe_load((ROOT / f"configs/models/{name}.yaml").read_text())
    assert profile["local_only"] and not profile["trust_remote_code"]
    assert profile["pinned_revision"] is None and profile["checkpoint_path"] is None
    checked = {
        "mistral7b": ("PRETRAINED_ENGINEERING_CANARY_RECORDED", "mistral_eos_20261101/trials/caution_plus_p0_self-eos"),
        "gemma2b": ("PRETRAINED_CONTENT_PROBES_RECORDED", "slot_content_20261101/trials/contrasts-gemma"),
    }
    if name in checked:
        status, trial = checked[name]
        receipt = ROOT / "provenance" / trial / "checkpoint_identity.json"
        assert profile["status"] == status
        assert profile["tested_checkpoint_sha256"] == json.loads(receipt.read_text())["checkpoint_sha256"]
    else:
        assert "NOT_PRETRAINED_TESTED" in profile["status"]
    assert profile["approx_active_parameters_billions"] <= profile["approx_total_parameters_billions"]

def test_example_gate_not_passing():
 g=json.loads((ROOT/"configs/qualification.example.json").read_text())
 assert g["status"]=="NOT_QUALIFIED"
