"""Recompute native endpoints from complete, paired, de-duplicated decisions."""
from collections import defaultdict
import json
from pathlib import Path
import sys

GROUPS = {"train": ("both_if-01", "either_if-01", "unless-01", "exclusive-01"),
          "dev": ("requirement-00", "requirement-01")}
FORMS = ("rule", "forward", "reverse")


def summarize(folder):
    if json.loads((folder / "terminal.json").read_text())["status"] != "COMPLETED_POLICY_FORM_DEVELOPMENT":
        raise ValueError("Incomplete run")
    cells = defaultdict(dict)
    for line in (folder / "decisions.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = r["condition"], r["endpoint"], r["form"], r["split"]
        unit = r["group"], r["pole"], r["world"], r["label_order"], r["source_order"]
        if unit in cells[key]:
            raise ValueError("Duplicate observation")
        cells[key][unit] = tuple(r[k] for k in ("gold","greedy_token","greedy_label","A","B","source_sha256","question_sha256"))
    out = {}
    for condition in ("literal","latent"):
        out[condition] = {}
        for endpoint in ("initial","rule_fit","case_fit"):
            results = {}
            for split, groups in GROUPS.items():
                units = [(g,p,w,l) for g in groups for p in ("plus","minus") for w in range(4) for l in range(2)]
                pairs = [(g,w,l) for g in groups for w in range(4) for l in range(2)]
                correct = {}
                result = {}
                for form in FORMS:
                    panel = cells[condition,endpoint,form,split]
                    if set(panel) != {u+(i,) for u in units for i in (0,1)}:
                        raise ValueError("Wrong panel denominator")
                    if any(panel[u+(0,)] != panel[u+(1,)] for u in units):
                        raise ValueError("Nominal duplicate tags have different evidence")
                    values = {u:panel[u+(0,)] for u in units}
                    ok = {u:v[0] == v[2] for u,v in values.items()}
                    correct[form] = ok
                    saved = result[form] = {"native_correct":sum(ok.values()),"n":len(units),
                        "opposed_both_correct":sum(ok[g,"plus",w,l] and ok[g,"minus",w,l] for g,w,l in pairs),
                        "pairs":len(pairs),"forced_ties":sum(v[3] == v[4] for v in values.values()),
                        "forced_correct":sum((v[3]>v[4] if v[0]=="A" else v[4]>v[3]) for v in values.values()),
                        "native_label_coverage":sum(v[2] in ("A","B") for v in values.values())}
                    saved["by_group"] = {g:sum(v for k,v in ok.items() if k[0]==g) for g in groups}
                result["both_case_directions"] = {"correct_pairs":sum(all(correct[f][g,p,w,l] for f in ("forward","reverse")
                                                        for p in ("plus","minus")) for g,w,l in pairs),"pairs":len(pairs)}
                results[split] = result
            out[condition][endpoint] = results
    if len(cells) != 36:
        raise ValueError("Unexpected conditions")
    screens = {}
    for condition in out:
        train = out[condition]["case_fit"]["train"]
        f = train["forward"]
        r = train["both_case_directions"]
        screens[condition] = {"trainability": f["native_correct"]/f["n"] >= .95 and f["opposed_both_correct"]/f["pairs"] >= .90,
                              "order_robustness": r["correct_pairs"]/r["pairs"] >= .90}
    return {"cells":out, "screens":screens, "logical_rows":sum(len(p) for p in cells.values())}


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/"run"),
                     indent=2, sort_keys=True))
