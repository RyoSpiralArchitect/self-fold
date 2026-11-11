"""A decoder miss must not turn a resolved report into a missing report."""
import importlib.util
import json
from pathlib import Path


def test_unique_forms_count_reports_once_under_consensus_category(tmp_path, monkeypatch, capsys):
    path = (Path(__file__).resolve().parents[1]
            / "provenance/message_boundary_20261106/audit_roundtrip.py")
    spec = importlib.util.spec_from_file_location("roundtrip_audit_test", path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    # Importing the audit must not execute an unrelated archived run.
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "GROUPS", [{"options": ["river path", "ridge path"]}])
    prod = tmp_path / "provenance/message_boundary_20261106/fixture/produce"
    prod.mkdir(parents=True)
    (prod / "fit_keys.json").write_text(json.dumps({"keys": [[0, 0, 0, 0, 0]]}))
    (prod / "test_keys.json").write_text(json.dumps({"keys": []}))
    texts = ["off/off -> river path.", "off/off -> river path.",
             "off/off -> ridge path.", "No selection given."]
    (prod / "reports_rt_directed.jsonl").write_text("".join(
        json.dumps(dict(source=0, prior=8, mask=8, out=0, text=text)) + "\n"
        for text in texts))
    audit.audit("fixture")
    output = capsys.readouterr().out
    assert "n=4 unresolved=1" in output
    assert "unresolved_all_missing: 1 unique forms / 1 reports" in output
    assert "consensus_correct: 1 unique forms / 2 reports" in output
    assert "consensus_wrong: 1 unique forms / 1 reports" in output
    assert "disagreement: 0 unique forms / 0 reports" in output
