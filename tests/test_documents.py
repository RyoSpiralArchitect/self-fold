from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1]

def test_bibliography_resolves_and_has_audit():
    bib=(ROOT/'paper/references.bib').read_text()
    keys=set(re.findall(r'@\w+\{([^,]+),',bib))
    assert len(keys)==13
    tex=(ROOT/'paper/main.tex').read_text()
    citations={k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',tex) for k in group.split(',')}
    assert citations <= keys
    audit=json.loads((ROOT/'paper/sources.json').read_text())
    assert {x['key'] for x in audit}==keys
    assert all(x['source'].startswith('https://') for x in audit)
    assert bib.count('{')==bib.count('}')

def test_readme_local_links_exist():
    text=(ROOT/'README.md').read_text()
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if not link.startswith(('https://','http://','#')):
            assert (ROOT/link.split('#')[0]).exists(),link

def test_other_frame_not_an_utterance_report():
    rows=[json.loads(x) for x in (ROOT/'data/seeds.dev.jsonl').read_text().splitlines()]
    for row in rows:
        if row['subject']=='agent_a':
            assert row['text'].startswith('Agent A ')
            assert 'I ' not in row['text'] and 'stated' not in row['text'] and 'focuss' not in row['text']

def test_explicit_no_empirical_claim():
    status=(ROOT/'docs/IMPLEMENTATION_STATUS.md').read_text()
    assert 'NOT RUN' in status and 'NOT ESTABLISHED' in status
    paper=(ROOT/'paper/main.md').read_text()
    assert '**Not available.**' in paper
