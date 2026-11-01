from copy import deepcopy
import torch
import pytest
from selffold.engine import run_trajectory, observe, branch_once, task_readout
from selffold.records import State
from selffold.prompts import operation,text_prompt

@pytest.mark.parametrize("arm",["T","L","B"])
@pytest.mark.parametrize("regime",["plain","expand"])
def test_three_arms_two_regimes(host,arm,regime):
 hist,rows=run_trajectory(host,"I value evidence.",arm=arm,regime=regime,steps=2,observer=False)
 assert len(hist)==3 and len(rows)==2
 assert rows[0]["operator"]==("plain" if regime=="plain" else "decompose")
 assert all(r["observer_compute"]["forwards"]==0 for r in rows)
 if arm=="B":assert "roundtrip_text" in rows[0]["action"]
 if arm=="L":assert "roundtrip_text" not in rows[0]["action"]

def test_observer_does_not_change_trajectory(host):
 s=host.encode("I value revisable decisions.")
 a,_=run_trajectory(host,"unused",arm="L",steps=2,initial=s,observer=False)
 b,_=run_trajectory(host,"unused",arm="L",steps=2,initial=s,observer=True)
 assert [x.sha() for x in a]==[x.sha() for x in b]

def test_L_never_decodes_without_observer(host,monkeypatch):
 def bad(*a):raise AssertionError("decode was called")
 monkeypatch.setattr(host,"decode",bad)
 run_trajectory(host,"I value evidence.",arm="L",steps=2,observer=False)

def test_T_has_no_latent_calls(host,monkeypatch):
 def bad(*a):raise AssertionError("latent operation called")
 monkeypatch.setattr(host,"encode",bad);monkeypatch.setattr(host,"update_latent",bad)
 run_trajectory(host,"I value evidence.",arm="T",steps=1,observer=False)

def test_B_uses_G_D_E_order(host,monkeypatch):
 s=host.encode("I value evidence.");calls=[]
 for name in ["update_latent","decode","encode"]:
  orig=getattr(host,name)
  def wrap(*a,_name=name,_orig=orig,**kw):
   calls.append(_name);return _orig(*a,**kw)
  monkeypatch.setattr(host,name,wrap)
 run_trajectory(host,"unused",arm="B",steps=1,initial=s,observer=False)
 assert calls==["update_latent","decode","encode"]

def test_matched_initial_latent(host):
 s=host.encode("I value evidence.")
 l,_=run_trajectory(host,"unused",arm="L",steps=1,initial=s,observer=False)
 b,_=run_trajectory(host,"unused",arm="B",steps=1,initial=s,observer=False)
 assert l[0].sha()==b[0].sha()==s.sha()

def test_one_time_branch(host):
 s=host.encode("I value evidence.");h=s.sha()
 b=branch_once(host,s,continuation_steps=1)
 assert len(b["identity"]["states"])==2
 assert b["identity"]["states"][0].sha()==h
 assert b["language_roundtrip"]["readout"] is not None
 assert s.sha()==h

def test_task_does_not_change_parent(host):
 s=host.encode("I value evidence.");h=s.sha()
 task={"id":"t1","group":"t","question":"A or B?","labels":["A","B"],"answer":"A"}
 r=task_readout(host,s,task)
 assert r["state_sha256"]==h==s.sha()

def test_fixed_schedule():
 assert [operation(i,"expand") for i in range(4)]==["decompose","boundary","reintegrate","decompose"]
 with pytest.raises(ValueError):operation(0,"adaptive")

def test_initial_seed_not_reintroduced(host,monkeypatch):
 seen=[];orig=host.update_text
 def spy(text,op):seen.append(text);return orig(text,op)
 monkeypatch.setattr(host,"update_text",spy)
 h,_=run_trajectory(host,"UNIQUE_INITIAL_DESCRIPTION",arm="T",steps=2,observer=False)
 assert seen[0]=="UNIQUE_INITIAL_DESCRIPTION" and seen[1]==h[1].text
 assert "step 1" not in text_prompt("current","plain")
