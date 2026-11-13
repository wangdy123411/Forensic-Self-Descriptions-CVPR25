from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_tables import build_report, export_files, read_prediction_snapshot


def manifest_row(key, source, label):
    return dict(image_id=key, relative_path=key + ".png", source=source,
                label=str(label), split="test", sha256="a" * 64)


def prediction(row, fake_score):
    return dict(row, status="ok", fake_score=str(fake_score), z_score=str(-fake_score),
                raw_score="0", predicted_fake=str(int(fake_score > 2)), threshold="-2", error="")


class ExportSafetyTests(unittest.TestCase):
    def setUp(self):
        self.manifest = [manifest_row("r1", "COCO2017", 0), manifest_row("r2", "COCO2017", 0),
                         manifest_row("f1", "BigGAN", 1), manifest_row("f2", "BigGAN", 1),
                         manifest_row("f3", "GLIDE", 1)]

        self.predictions = [prediction(row, score) for row, score in zip(self.manifest, [0, 1, 1, 3, -1])]

    def test_incomplete_source_is_not_scored(self):
        report = build_report(self.manifest, self.predictions[:3], {})
        self.assertEqual(report["run_status"], "PARTIAL")
        self.assertEqual(report["measured_complete_pairs"], 0)
        self.assertIsNone(report["macro_auc_available_finished_pairs"])
        appendix = report["appendix_tables"][0]
        self.assertEqual(next(r for r in appendix["rows"] if r["generator"] == "BigGAN")["status"], "pending")

    def test_complete_available_data_is_not_complete_paper(self):
        report = build_report(self.manifest, self.predictions, {})
        self.assertEqual(report["run_status"], "COMPLETE_AVAILABLE_DATA")
        self.assertEqual(report["measured_complete_pairs"], 2)
        self.assertEqual(report["macro_auc_available_finished_pairs"], (0.875 + 0) / 2)
        self.assertFalse(report["exact_paper_reproduction"])
        self.assertIsNone(report["paper_96_pair_macro_auc"])
        self.assertEqual(report["main_table_2"][0]["measured_generators"], "GLIDE")
        self.assertIsNone(report["main_table_1"][-1]["MIDB"])

    def test_error_invalidates_whole_source(self):
        rows = [dict(r) for r in self.predictions]
        rows[2].update(status="error", error="CUDA OOM")
        report = build_report(self.manifest, rows, {})
        self.assertEqual(report["run_status"], "COMPLETED_WITH_ERRORS")
        self.assertEqual(report["measured_complete_pairs"], 1)
        self.assertEqual(report["macro_auc_available_finished_pairs"], 0)
        self.assertEqual(report["failed_count"], 1)

    def test_reversed_or_tampered_scores_rejected(self):
        rows = [dict(r) for r in self.predictions]
        rows[1]["fake_score"] = "-1"
        with self.assertRaisesRegex(ValueError, "oriented"):
            build_report(self.manifest, rows, {})
        rows = [dict(r) for r in self.predictions]
        rows[1]["source"] = "MIDB"
        with self.assertRaisesRegex(ValueError, "metadata"):
            build_report(self.manifest, rows, {})

    def test_unterminated_live_csv_tail_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            path.write_bytes(b"image_id,status\r\na,ok\r\nb,er")
            rows, ignored = read_prediction_snapshot(path)
            self.assertTrue(ignored)
            self.assertEqual(rows, [{"image_id": "a", "status": "ok"}])

    def test_output_preserves_na_and_explicit_partial_state(self):
        report = build_report(self.manifest, self.predictions[:4], {})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            export_files(report, output)
            data = json.loads((output / "table_report.json").read_text(encoding="utf-8"))
            self.assertEqual(data["run_status"], "PARTIAL")
            self.assertIn("PARTIAL", (output / "paper_tables.tex").read_text(encoding="utf-8"))
            self.assertIn("N/A", (output / "report.html").read_text(encoding="utf-8"))
            self.assertEqual(len(list(output.glob("appendix_table_*.csv"))), 4)


if __name__ == "__main__":
    unittest.main()
