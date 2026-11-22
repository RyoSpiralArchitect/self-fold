# What one evaluation unit contains

The November 22 manuscript uses the registered denominators. During review we also counted distinct displayed prompts and the native action inputs shared across trajectories. The counts below can be recovered from the saved records without loading Gemma.

## Four updates and three report boundaries

The four fitted menus have two possible assignments of the source roles. Four source realizations share each assignment. Crossing 32 realizations with all $14^4$ instruction strings gives 1,229,312 histories. The first update uses the start address. Reports enter updates two, three and four. The native judgment bank for these menus contains sixteen distinct inputs (four menus, two actions and two answer orders), all used in fitting and evaluation. The earlier role-bit experiment in Section 7 used a different bank of 196 native inputs. [Report sharing](../src/selffold/lm_writer_reports.py) · [Saved input keys](../provenance/message_boundary_20261106/write7/produce/test_native_hard/calls.json) · [Four-update results](../provenance/message_boundary_20261106/write9/RESULTS.md)

The directed prompt asks about the instruction. In the earlier saved bank, the instruction and the observed table differed somewhere on 9,856 entries. They differed at the **queried off/off row** on 6,496. On this narrower set the saved lexical decoder read the observed action on 5,054 directed reports and the instructed action on 1,442. The open-ended reports gave 2,510 observed actions, 936 instructed actions and 3,050 missing mentions. This retrospective count describes the lexical decoder on that bank. [Original report audit](../provenance/message_boundary_20261106/write9/produce/report_audit.json)

## Generations and distinct prompts

The rendered records can coincide under different assignments of source roles and instruction masks. The original schedules count every transition key. For the descriptive last column, each distinct set of displayed questions counts once. Repeated identical prompts produced identical responses within each study.

| Study and condition | Generation calls / distinct prompt strings | Registered conflicting units, all required answers correct | Per distinct set of questions |
| --- | ---: | ---: | ---: |
| Instruction versus observation (write10), both questions | 448 / 288 | 21/112 | 15/72 |
| Source-cue comparison (write11), control, both questions and both orders | 896 / 576 | 50/112 | 32/72 |
| Same comparison, explicit cue | 896 / 576 | 57/112 | 37/72 |

Each of the 224 instruction calls in the first study reused prompt text from the earlier directed bank. The 224 observation calls used fresh question text. The single-record comparison and its four full-context controls (write12) together used 3,840 distinct prompts. [Instruction and observation scores](../provenance/message_boundary_20261106/write10/reports/report_summary.json) · [Record-cue comparison](../provenance/message_boundary_20261106/write11/produce/report_summary.json) · [Single-record prompt verification](../provenance/message_boundary_20261106/write12/visible_rows.json)

From the repository root, run the following with Python 3.11 or later.

```python
from collections import Counter, defaultdict
from pathlib import Path
import json

base = Path("provenance/message_boundary_20261106")

def read(path):
    return json.loads(path.read_text())

def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

bank = base / "write7/produce"
fit = {tuple(x) for x in read(bank / "fit_native_hard/calls.json")}
test = {tuple(x) for x in read(bank / "test_native_hard/calls.json")}
menus = {x[1] for x in fit}
assert fit == {x for x in test if x[1] in menus}
print("native inputs, fitted menus / all menus:", len(fit), len(test))

for style in ["rt_directed", "rt_natural"]:
    rows = lines(bank / f"reports_{style}.jsonl")
    counts = Counter()
    for r in rows:
        if (r["mask"] ^ r["out"]) & 1:
            if r["missing"]:
                counts["missing"] += 1
            elif r["bit"] == (r["out"] & 1):
                counts["observation"] += 1
            else:
                counts["instruction"] += 1
    print(style, sum(counts.values()), dict(counts))

old_prompts = {q["prompt"] for q in read(bank / "reports_rt_directed.binding.json")["requests"]}
for study, path in [("write10", base / "write10/reports"), ("write11", base / "write11/produce")]:
    requests = read(path / "requests.json")
    scores = {r["id"]: r for r in read(path / "scored_reports.json")}
    outputs = {r["id"]: r["text"] for r in lines(path / "reports.jsonl")}
    rendered = {}
    groups = defaultdict(dict)
    for q in requests:
        answer = outputs[q["id"]]
        assert rendered.setdefault(q["prompt"], answer) == answer
        scored = scores[q["id"]]
        if scored["coordinate_conflict"]:
            arm = q.get("arm", "original")
            key = q.get("transition", q["pair"])
            fields = (q.get("layout", "single order"), q["question"])
            groups[arm, key][fields] = (q["prompt"], scored["requested_correct"])
    print(study, len(requests), len(rendered))
    if study == "write10":
        for kind in ["instruction", "actual"]:
            print(kind, "old prompt matches:", sum(q["prompt"] in old_prompts
                  for q in requests if q["question"] == kind))
    else:
        for arm in ["copy_only", "source_cue"]:
            prompts = [q["prompt"] for q in requests if q["arm"] == arm]
            print(arm, "calls and distinct prompts:", len(prompts), len(set(prompts)))
    unique, registered = defaultdict(dict), defaultdict(list)
    for (arm, _), fields in groups.items():
        assert len(fields) == (2 if study == "write10" else 4)
        prompts, correct = zip(*(fields[k] for k in sorted(fields)))
        passed = all(correct)
        registered[arm].append(passed)
        assert unique[arm].setdefault(prompts, passed) == passed
    for arm, values in registered.items():
        distinct = unique[arm].values()
        print(arm, "registered:", sum(values), len(values), "distinct:", sum(distinct), len(distinct))
```
