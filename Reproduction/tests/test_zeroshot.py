import copy
import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_zeroshot import main as evaluate_main, read_previous_predictions
from prepare_zeroshot_manifest import build_manifest
from zeroshot_common import (
    MANIFEST_FIELDS, PREDICTION_FIELDS, binary_metrics, check_resume_metadata,
    file_hash, load_manifest, summarize, write_csv,
)


class ProtocolTests(unittest.TestCase):
    def test_score_direction_ties_and_single_class(self):

        self.assertEqual(binary_metrics([0, 1], [-1, 4], [0, 1])["roc_auc"], 1)
        tied = binary_metrics([0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1])
        self.assertEqual(tied["roc_auc"], 0.5)
        self.assertEqual(tied["average_precision"], 0.5)
        single = binary_metrics([0, 0], [-1, 3], [0, 1])
        self.assertIsNone(single["roc_auc"])
        self.assertIsNone(single["average_precision"])
        self.assertEqual(single["false_positive_rate"], 0.5)

    def test_unweighted_pair_macro_not_pooled_auc(self):
        rows = [
            {"source": "COCO2017", "label": "0", "status": "ok", "fake_score": "1", "predicted_fake": "0"},
            {"source": "BigGAN", "label": "1", "status": "ok", "fake_score": "2", "predicted_fake": "1"},
            *[{"source": "StyleGAN", "label": "1", "status": "ok", "fake_score": "0", "predicted_fake": "0"} for _ in range(3)],
        ]
        report = summarize(rows, rows)
        self.assertEqual(report["macro_auc_available_paper_pairs"], 0.5)
        self.assertEqual(report["available_expected_pair_count"], 2)
        self.assertEqual(report["coverage_status"], "partial")
        self.assertIsNone(report["paper_96_pair_macro_auc"])
        self.assertFalse(report["exact_paper_reproduction"])
        pooled = binary_metrics([0, 1, 1, 1, 1], [1, 2, 0, 0, 0], [0, 1, 0, 0, 0])["roc_auc"]
        self.assertEqual(pooled, 0.25)
        extra = {"source": "Extra-Generator", "label": "1", "status": "ok", "fake_score": "2", "predicted_fake": "1"}
        expanded = summarize([*rows, extra], [*rows, extra])
        real_summary = expanded["per_real_source"]["COCO2017"]
        self.assertEqual(real_summary["paper_generator_macro_auc"], 0.5)
        self.assertEqual(real_summary["paper_generator_count"], 2)
        self.assertAlmostEqual(real_summary["macro_auc"], 2 / 3)

    def test_resume_identity_changes_rejected(self):
        original = {"manifest_sha256": "a", "weights_sha256": {"fre.pt": "x"}, "threshold": -2}
        check_resume_metadata(copy.deepcopy(original), original)
        for key, value in (("manifest_sha256", "b"), ("weights_sha256", {"fre.pt": "y"}), ("threshold", -3)):
            changed = copy.deepcopy(original)
            changed[key] = value
            with self.assertRaises(ValueError):
                check_resume_metadata(original, changed)

    def test_manifest_sampling_hashes_and_bad_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "real"
            directory.mkdir()
            for i in range(8):
                (directory / f"{i}.jpg").write_bytes(f"Original bytes {i}".encode())
            config = {"sources": [{"source": "COCO2017", "label": 0, "directory": "real", "split": "test"}]}
            rows, _ = build_manifest(config, root, per_source=4, seed=17)
            repeated, _ = build_manifest(config, root, per_source=4, seed=17)
            self.assertEqual(rows, repeated)
            manifest = root / "manifest.csv"
            write_csv(manifest, MANIFEST_FIELDS, rows)
            self.assertEqual(load_manifest(manifest, root), rows)
            broken = copy.deepcopy(rows)
            broken[0]["split"] = "train"
            write_csv(manifest, MANIFEST_FIELDS, broken)
            with self.assertRaises(ValueError):
                load_manifest(manifest, root)
            broken[0]["split"] = "test"
            broken[0]["relative_path"] = "../outside.jpg"
            write_csv(manifest, MANIFEST_FIELDS, broken)
            with self.assertRaises(ValueError):
                load_manifest(manifest, root)
            write_csv(manifest, MANIFEST_FIELDS, rows)
            (root / rows[0]["relative_path"]).write_bytes(b"changed bytes")
            with self.assertRaises(ValueError):
                load_manifest(manifest, root)

    def test_cross_label_duplicate_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for i, (source, label) in enumerate((("COCO2017", "0"), ("BigGAN", "1"))):
                path = root / f"{i}.jpg"
                path.write_bytes(b"same original bytes")
                rows.append(dict(image_id=str(i), relative_path=path.name, label=label, source=source, split="test", sha256=file_hash(path)))
            manifest = root / "manifest.csv"
            write_csv(manifest, MANIFEST_FIELDS, rows)
            with self.assertRaises(ValueError):
                load_manifest(manifest, root)

    def test_multiple_directories_and_source_count_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("SB", "DMID"):
                (root / name).mkdir()
                for i in range(3):
                    (root / name / f"{i}.jpg").write_bytes(f"{name}-{i}".encode())
            config = {"sources": [{"source": "DALLE 2", "label": 1, "directories": ["SB", "DMID"], "split": "test", "sample_count": 5}]}
            rows, report = build_manifest(config, root, per_source=1, paper_counts=True)
            self.assertEqual(len(rows), 5)
            self.assertEqual({r["relative_path"].split("/")[0] for r in rows}, {"SB", "DMID"})
            self.assertEqual(report["sources"][0]["requested"], 5)

    def test_resume_prediction_decision_boundary(self):
        item = dict(image_id="1", relative_path="a.jpg", label="0", source="COCO2017", split="test", sha256="a" * 64)
        prediction = dict(item, status="ok", z_score="-2.0", raw_score="0.0", fake_score="2.0", predicted_fake="0", threshold="-2.0", elapsed_seconds="0.1", error="")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.csv"
            write_csv(path, PREDICTION_FIELDS, [prediction])
            self.assertEqual(len(read_previous_predictions(path, [item], -2.0)), 1)
            prediction["predicted_fake"] = "1"
            write_csv(path, PREDICTION_FIELDS, [prediction])
            with self.assertRaises(ValueError):
                read_previous_predictions(path, [item], -2.0)

    def test_cli_resume_avoids_rescoring_and_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for i, (source, label) in enumerate((("COCO2017", "0"), ("BigGAN", "1"))):
                path = root / f"{i}.jpg"
                path.write_bytes(f"fixture bytes {i}".encode())
                rows.append(dict(image_id=str(i), relative_path=path.name, label=label, source=source, split="test", sha256=file_hash(path)))
            manifest = root / "manifest.csv"
            write_csv(manifest, MANIFEST_FIELDS, rows)
            weights = root / "weights"
            weights.mkdir()
            config = {key: {"weights_file": key + ".pt"} for key in ("fre", "gmm", "transforms")}
            config["scoring"] = {"default_threshold": -2.0}
            (weights / "config.json").write_text(json.dumps(config), encoding="utf-8")
            for key in ("fre", "gmm", "transforms"):
                (weights / f"{key}.pt").write_bytes(key.encode())
            package = root / "fsd"
            package.mkdir()
            (package / "__init__.py").write_text("pass\n", encoding="utf-8")
            calls = []

            class FixtureDetector:
                @classmethod
                def load(cls, **kwargs):
                    return cls()

                def score(self, path):
                    calls.append(path.name)
                    z = -1.0 if path.name == "0.jpg" else -3.0
                    return SimpleNamespace(z_score=z, raw_score=10.0, threshold=-2.0, is_fake=z < -2.0)

            fake_torch = SimpleNamespace(__version__="test-fixture", get_num_threads=lambda: 1, cuda=SimpleNamespace(is_available=lambda: False))
            fake_fsd = SimpleNamespace(__file__=str(package / "__init__.py"), FSDDetector=FixtureDetector)
            output = root / "results"
            arguments = ["evaluate_zeroshot.py", "--manifest", str(manifest), "--data-root", str(root), "--output-dir", str(output), "--weights-dir", str(weights), "--device", "cpu"]
            with patch.dict(sys.modules, {"torch": fake_torch, "fsd": fake_fsd}), redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", arguments):
                    self.assertEqual(evaluate_main(), 0)
                prediction_path = output / "predictions.csv"
                prediction_path.write_bytes(prediction_path.read_bytes().rstrip(b"\r\n"))
                with patch.object(sys, "argv", [*arguments, "--resume"]):
                    self.assertEqual(evaluate_main(), 0)
                self.assertEqual(calls, ["0.jpg", "1.jpg"])
                self.assertTrue(prediction_path.read_bytes().endswith(b"\n"))


                lines = prediction_path.read_bytes().splitlines()
                prediction_path.write_bytes(b"\n".join(lines[:2]))
                with patch.object(sys, "argv", [*arguments, "--resume"]):
                    self.assertEqual(evaluate_main(), 0)
                self.assertEqual(calls, ["0.jpg", "1.jpg", "1.jpg"])
                self.assertEqual(len(read_previous_predictions(prediction_path, rows, -2.0)), 2)
                summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["scored_count"], 2)
                self.assertEqual(summary["macro_auc_available_paper_pairs"], 1.0)
                with patch.object(sys, "argv", [*arguments, "--resume", "--threshold", "-3"]):
                    with self.assertRaises(SystemExit) as error:
                        evaluate_main()
                    self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
