from pathlib import Path
import json,re
import pytest
ROOT=Path(__file__).resolve().parents[1]

def test_bibliography_resolves_and_has_audit():
    bib=(ROOT/'paper/references.bib').read_text()
    keys=set(re.findall(r'@\w+\{([^,]+),',bib))
    assert len(keys)==15
    assert {"babu2026quine", "feng2024propositional", "frasertaliente2026nla"} <= keys
    tex=(ROOT/'paper/main.tex').read_text()
    citations={k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',tex) for k in group.split(',')}
    assert citations <= keys
    audit=json.loads((ROOT/'paper/sources.json').read_text())
    assert {x['key'] for x in audit}==keys
    assert all(x['source'].startswith('https://') for x in audit)
    assert bib.count('{')==bib.count('}')

@pytest.mark.parametrize("filename", ["README.md", "README.ja.md"])
def test_readme_local_links_exist(filename):
    text=(ROOT/filename).read_text()
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if not link.startswith(('https://','http://','#')):
            assert (ROOT/link.split('#')[0]).exists(),link

def test_other_frame_not_an_utterance_report():
    rows=[json.loads(x) for x in (ROOT/'data/seeds.dev.jsonl').read_text().splitlines()]
    for row in rows:
        if row['subject']=='agent_a':
            assert row['text'].startswith('Agent A ')
            assert 'I ' not in row['text'] and 'stated' not in row['text'] and 'focuss' not in row['text']

def test_paper_distinguishes_development_from_confirmation():
    for name in ("main.md", "main.tex"):
        paper = (ROOT / "paper" / name).read_text().lower()
        assert "mistral" in paper and "confirmation" in paper and "uncalibrated" in paper
        assert "no pretrained-model or furnace result is reported" not in paper
