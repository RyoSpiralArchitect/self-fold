"""CPU-only checks of stage attribution and joins; never load qualification data."""

import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "provenance/g_stage_diagnostics_20261104"
sys.path.insert(0, str(AUDIT))
import audit
import audit_native
import audit_structured


class StageDiagnosticsTests(unittest.TestCase):
    def test_frozen_results_and_field_diagnostics_reproduce(self):
        receipt = audit.verify_archive(ROOT)
        self.assertTrue(all(receipt["original_summary_reductions_exact"].values()))
        self.assertTrue(all(receipt["original_field_reductions_exact"].values()))
        self.assertFalse(receipt["qualification_opened"])
        self.assertEqual(receipt["qualification_status"], "NOT_QUALIFIED")

    def test_physical_comparison_requires_all_conditions(self):
        conditions = dict(source_digest="source", question_digest="question",
                          reader="adapter", geometry="cut13", scoring="strict")
        self.assertTrue(audit.exact_evaluation_match(conditions, dict(conditions)))
        for field in conditions:
            different = {**conditions, field: "different"}
            self.assertFalse(audit.exact_evaluation_match(conditions, different))
        self.assertFalse(audit.exact_evaluation_match({}, {}))

    def test_semantic_overlap_keeps_missingness_and_physical_difference(self):
        rows = list(audit.overlap_rows(ROOT))
        self.assertEqual(len(rows), 1024)
        self.assertTrue(all(r["same_authored_source"] and r["same_semantic_query"] for r in rows))
        self.assertFalse(any(r["same_physical_evaluation"] for r in rows))
        present = [r for r in rows if r["native_D_updated_present"]]
        absent = [r for r in rows if not r["native_D_updated_present"]]
        self.assertEqual((len(present), len(absent)), (128, 896))
        self.assertTrue(all(r["native_D_updated_correct"] == "" for r in absent))
        self.assertEqual(sum(r["structured_updated_correct"] for r in rows), 1024)
        self.assertEqual(sum(r["structured_D_updated_correct"] for r in present), 92)

    def test_native_registered_gain_is_tie_status_not_action_change(self):
        path = ROOT / audit.STUDY / "raw-bound/decisions.jsonl.gz"
        with gzip.open(path, "rt") as stream:
            records = [json.loads(line) for line in stream]
        by_tag = {}
        for r in records:
            if r[6] in ("identity", "boundary"):
                by_tag.setdefault(r[6], {})[(*r[:6], r[7], r[8])] = r
        gains = []
        for key, before in by_tag["identity"].items():
            after = by_tag["boundary"][key]
            self.assertEqual((before[10], after[10]), ("A", "A"))
            old_ok = before[10] == before[9] and before[12] != before[13]
            new_ok = after[10] == after[9] and after[12] != after[13]
            if new_ok and not old_ok:
                self.assertEqual(before[12], before[13])
                self.assertFalse(audit_native.ledger(key[:6], key[6])[1])
                gains.append(key)
            self.assertFalse(old_ok and not new_ok)
        self.assertEqual(len(gains), 4)

    def test_hard_action_content_and_binary_success_are_distinct(self):
        p = audit_structured.load_panel(ROOT, "readout")
        hard = p["states"]["updated"].argmax(-1)
        hard_correct = hard == p["gold_action"]
        native_correct = p["routes"]["updated"]["correct"]
        off_menu = (hard // 2) != p["keys"][:, 0, None]
        self.assertEqual(int(hard_correct.sum()), 4624)
        self.assertEqual(int(off_menu.sum()), 160)
        self.assertEqual(int((native_correct & ~hard_correct[:, :, None]).sum()), 224)
        np.testing.assert_array_equal(p["dispatch"]["analytic"], p["dispatch"]["updated"])
        np.testing.assert_array_equal(p["routes"]["updated"]["top"], p["routes"]["ED_updated"]["top"])
        self.assertFalse(np.array_equal(p["routes"]["updated"]["ab"], p["routes"]["ED_updated"]["ab"]))

    def test_reused_style_ids_do_not_mean_reused_source_or_question(self):
        p = audit_structured.load_panel(ROOT, "structured")
        cfg = audit_native.read(ROOT / "data/supported_g_20261104.json")
        new = audit_native.read(ROOT / "data/supported_g_readout_20261104.json")
        for key in p["keys"]:
            old_text = audit_structured.source_text(key, p["menus"], cfg["fresh"])
            new_text = audit_structured.source_text(key, p["menus"], new["fresh"])
            self.assertNotEqual(audit_native.digest(old_text), audit_native.digest(new_text))
        key = p["keys"][0]
        q0 = audit_structured.literal_question("structured", p["menus"], key, 0, 0, new)
        q1 = audit_structured.literal_question("readout", p["menus"], key, 0, 0, new)
        self.assertNotEqual(audit_native.digest(q0), audit_native.digest(q1))

    def test_output_cannot_replace_existing_evidence(self):
        with tempfile.TemporaryDirectory() as existing:
            with self.assertRaises(FileExistsError):
                audit.run(ROOT, Path(existing))
        with self.assertRaises(ValueError):
            audit.run(ROOT, ROOT / audit.STUDY / "forbidden_new_result")


if __name__ == "__main__":
    unittest.main()
