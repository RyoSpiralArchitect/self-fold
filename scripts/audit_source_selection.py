"""Post-result description of naming and agreement controls; never changes gates."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return re.sub(r'\W+', ' ', text.casefold()).strip()


def audit(run):
    if read(run/'terminal.json')['status'] != 'SOURCE_SELECTION_CLOSED':
        raise ValueError('A closed scientific run is required')
    if sha(run/'reports.jsonl') != read(run/'records_closed.json')['sha256']:
        raise ValueError('Report bytes changed')
    reqs = read(run/'requests.json')
    rows = {r['id']:r for r in map(json.loads,(run/'reports.jsonl').read_text().splitlines())}
    scored = {r['id']:r for r in read(run/'scored_reports.json')}
    if len(reqs) != len(rows) or set(rows) != set(scored) or set(rows) != {q['id'] for q in reqs}:
        raise ValueError('Mismatched records')
    groups, wrong = {}, []
    for q in reqs:
        r, s = rows[q['id']], scored[q['id']]
        key = q['arm']+'/'+q['question']
        counts = groups.setdefault(key,dict(all=0,correct=0,canonical_name=0,
            unanimous=0,unanimous_correct=0,conflicting_records=0,conflicting_records_correct=0))
        canonical = sum(normalize(r['text']) == normalize(o) for o in q['options']) == 1
        expected = q['options'][((q['mask'] if q['question']=='instruction' else q['out'])&1)^q['pole']]
        if s['strict_requested_correct'] != (normalize(r['text']) == normalize(expected)):
            raise ValueError('Canonical-name normalization differs from the registered scorer')
        unanimous = (q['prior']&1) == (q['mask']&1) == (q['out']&1)
        counts['all'] += 1; counts['correct'] += s['requested_correct']
        counts['canonical_name'] += canonical
        label = 'unanimous' if unanimous else 'conflicting_records'
        counts[label] += 1; counts[label+'_correct'] += s['requested_correct']
        if unanimous and not s['requested_correct']:
            wrong.append(dict(id=q['id'],response=r['text'],shared_offoff_action=expected,
                arm=q['arm'],question=q['question'],layout=q['layout']))
    return dict(posthoc=True,changes_registered_gates=False,
        method='All three displayed records have the same off/off action when the low bits of prior, mask and out agree. Count requested-action correctness on that subset.',
        inputs={name:sha(run/name) for name in ('requests.json','reports.jsonl','scored_reports.json','terminal.json')},
        script_sha256=sha(Path(__file__)),groups=groups,unanimous_errors=wrong,
        generation_calls=0,new_native_scoring_calls=0)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args(); value=audit(args.run)
    if args.verify:
        assert value==read(args.out)
        print(json.dumps(dict(exact=True,posthoc=True,groups=len(value['groups']))))
    else:
        with args.out.open('x') as f: json.dump(value,f,indent=2,sort_keys=True); f.write('\n')
        print(json.dumps(value['groups'],sort_keys=True))
