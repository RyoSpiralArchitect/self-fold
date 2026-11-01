#!/usr/bin/env python3
"""Deterministic, human-authored DEVELOPMENT fixtures; not confirmatory or training data."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
PAIRS={
 "caution": (["prioritize keeping decisions revisable", "value the ability to revise a decision"],
             ["prioritize making decisions promptly", "value reaching a decision without unnecessary delay"]),
 "evidence": (["prefer broad supporting evidence before committing", "seek support from several observations before committing"],
              ["prefer a decisive diagnostic observation", "value evidence that sharply separates competing explanations"]),
 "consistency": (["value consistency across judgments", "prefer judgments that fit a coherent set of principles"],
                 ["value revising judgments when conditions change", "prefer adapting a judgment to a changed situation"]),
 "exploration": (["keep multiple plausible explanations available", "prefer preserving several credible explanations"],
                 ["focus effort on the best-supported explanation", "prefer concentrating on an explanation with the strongest support"]),
 "precision": (["prefer precise distinctions", "value clearly separating closely related concepts"],
               ["prefer useful broad patterns", "value patterns that connect several different situations"]),
 "planning": (["prefer planning before acting", "value anticipating the consequences of an action"],
              ["prefer learning through reversible actions", "value small actions that provide information and can be undone"])
}
def seeds():
 for group,(plus,minus) in PAIRS.items():
  for sign,variants in [("plus",plus),("minus",minus)]:
   for i,text in enumerate(variants):
    for subject in ("self","agent_a"):
     # Direct third-person predication, not a report of an earlier speech event.
     full="I "+text+"."
     if subject=="agent_a":
      verb,rest=text.split(' ',1)
      third=verb+('es' if verb.endswith('s') else 's')
      full='Agent A '+third+' '+rest+'.'
     yield {"id":f"{group}_{sign}_p{i}_{subject}","group":group,"pole":sign,"paraphrase":i,
            "subject":subject,"text":full,"split":"development"}
def reachable(edges,start,target):
 seen={start}
 while True:
  more=seen|{b for a,b in edges if a in seen}
  if more==seen:return target in seen
  seen=more

def tasks():
 for i in range(6):
  now=10+i
  for delay in [0,12]:
   wait=now-4+delay
   flip=(i%2==1)
   meaning={"A":"WAIT" if flip else "NOW","B":"NOW" if flip else "WAIT"}
   correct="WAIT" if wait<now else "NOW"
   yield {"id":f"decision-{i}-{delay}","group":f"decision-{i}","tier":"near","split":"development",
          "question":f"Minimize total cost. Acting now costs {now}. Waiting yields information that reduces this cost by 4, but adds a delay cost of {delay}. No other effects exist. A = {meaning['A']}; B = {meaning['B']}. Which option has smaller total cost?",
          "labels":["A","B"],"answer":next(k for k,v in meaning.items() if v==correct),
          "oracle":{"now":now,"wait":wait,"scope":"synthetic_cost_minimization"}}
 for i in range(8):
  edges=[("a","b"),("b","c")]
  if i%2==0:edges.append(("c","d"))
  if i%3==0:edges.append(("d","a"))
  start,target=("a","d") if i<4 else ("d","b")
  yes=reachable(edges,start,target)
  flip=i%2==1
  meanings={"A":not flip,"B":flip}
  yield {"id":f"reach-{i}","group":f"reach-{i//2}","tier":"far_fixture_not_benchmark","split":"development",
         "question":f"A directed graph has exactly these edges: {', '.join(a+' -> '+b for a,b in edges)}. Is there a directed path from {start} to {target}? A = {'YES' if meanings['A'] else 'NO'}; B = {'YES' if meanings['B'] else 'NO'}. Return A or B.",
         "labels":["A","B"],"answer":next(k for k,v in meanings.items() if v==yes),
         "oracle":{"edges":edges,"start":start,"target":target,"reachable":yes}}
if __name__=="__main__":
 for name,values in [("seeds.dev.jsonl",seeds()),("tasks.dev.jsonl",tasks())]:
  (ROOT/"data"/name).write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in values))
