import json
import numpy as np
import pytest
import torch
from selffold.transport import Transport,fit_ridge
from selffold.metrics import transition,router_summary
from selffold.records import State,digest,file_sha
from selffold.gates import verify_gate,REQUIRED

def test_transport_roundtrip_file(tmp_path):
 p=tmp_path/"transport.npz"
 np.savez(p,weight=np.eye(3,dtype=np.float32),bias=np.zeros(3,dtype=np.float32))
 t=Transport.load(p);x=torch.randn(2,3)
 assert torch.equal(t(x),x) and t.artifact_sha256==file_sha(p)

def test_transport_ridge():
 x=torch.randn(100,3);y=x*2+1
 t=fit_ridge(x,y,ridge=1e-6)
 assert torch.allclose(t(x),y,atol=1e-4)
 assert t.artifact_sha256=="fitted_unqualified"

def test_bad_transport(tmp_path):
 p=tmp_path/"bad.npz";np.savez(p,weight=np.ones((3,4)),bias=np.zeros(3))
 with pytest.raises(ValueError):Transport.load(p)

def test_metrics_are_proxies():
 r=transition(State(text="one two"),State(text="one three"))
 assert "lexical_jaccard_proxy_not_semantic" in r
 assert "semantic_distance" not in r

def test_fixed_point_not_failure():
 s=State(slots=torch.ones(3,4));m=transition(s,s)
 assert m["exact_repeat"] and m["rms_delta"]==0
 assert "failure" not in m

def test_router_metric():
 r=router_summary(torch.zeros(1,3,4),2)
 assert abs(r["entropy"]-np.log(4))<1e-6
 assert "topk_logit_candidates_not_dispatch_counts" in r

def gate_file(tmp_path):
 e=tmp_path/"evidence.json";e.write_text('{"result":"unit fixture only"}')
 h={"checkpoint":"fixture"}
 g={"status":"SEMANTIC_GATE_REVIEWED_PASS","reviewer":"unit-test-fixture","split":"calibration_only",
    "host_identity_sha256":digest(h),"source_sha256":"source",
    "checks":{k:{"verdict":"pass","evidence_path":e.name,"evidence_sha256":file_sha(e)} for k in REQUIRED}}
 p=tmp_path/"gate.json";p.write_text(json.dumps(g));return p,h,g

def test_gate_fixture(tmp_path):
 p,h,g=gate_file(tmp_path);assert verify_gate(p,host_identity=h,source_sha256="source")==g

@pytest.mark.parametrize("mode",["model","source","status","missing_evidence","missing_check"])
def test_gate_fail_closed(tmp_path,mode):
 p,h,g=gate_file(tmp_path)
 if mode=="model":h={"checkpoint":"other"}
 if mode=="status":g["status"]="ENGINEERING_PASS_NOT_SEMANTICS"
 if mode=="missing_evidence":(tmp_path/"evidence.json").unlink()
 if mode=="missing_check":g["checks"].pop(next(iter(REQUIRED)))
 p.write_text(json.dumps(g))
 with pytest.raises(ValueError):verify_gate(p,host_identity=h,source_sha256="other" if mode=="source" else "source")
