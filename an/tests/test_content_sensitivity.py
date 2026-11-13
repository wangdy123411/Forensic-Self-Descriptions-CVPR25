"""Checks of content-proxy bootstrap logic using fabricated scores only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

AN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AN))
from robustness.analysis import _cohort_metrics
from robustness.content_sensitivity import (
    _cluster_weights,
    _condition_metrics,
    _proxy_cluster_ids,
    analyze_content_sensitivity,
)


class ContentSensitivityTests(unittest.TestCase):
    def test_shared_stems_use_shared_draws_and_real_source_sizes_stay_fixed(self):
        frame = pd.DataFrame(
            {
                "image_id": ["r1", "r2", "r3", "a1", "a2", "b1", "b2"],
                "label": [0, 0, 0, 1, 1, 1, 1],
                "source": ["real_a", "real_a", "real_b", "A", "A", "B", "B"],
                "dataset": ["COCO"] * 3 + ["Synthbuster"] * 4,
                "archive_member": [""] * 3
                + [
                    "synth/A/shared.png",
                    "synth/A/unique_a.png",
                    "synth/B/shared.png",
                    "synth/B/unique_b.png",
                ],
            }
        )
        clusters = _proxy_cluster_ids(frame)
        self.assertEqual(clusters[3], clusters[5])
        self.assertNotEqual(clusters[4], clusters[6])
        weights = _cluster_weights(
            frame.label.to_numpy(), frame.source.to_numpy(), clusters, 300, np.random.default_rng(8)
        )
        np.testing.assert_array_equal(weights[:, 3], weights[:, 5])
        np.testing.assert_array_equal(weights[:, :2].sum(axis=1), 2)
        np.testing.assert_array_equal(weights[:, 2], 1)
        # Fake cluster counts are fixed, but source image counts deliberately vary.
        np.testing.assert_array_equal(weights[:, [3, 4, 6]].sum(axis=1), 3)
        self.assertTrue(np.any(weights[:, 3:5].sum(axis=1) != 2))
        frame.loc[5, "dataset"] = "different_dataset"
        self.assertNotEqual(_proxy_cluster_ids(frame)[3], _proxy_cluster_ids(frame)[5])
        frame.loc[5, "dataset"] = "Synthbuster"
        frame.loc[5, "archive_member"] = ""
        self.assertNotEqual(_proxy_cluster_ids(frame)[3], _proxy_cluster_ids(frame)[5])

    def test_macro_discards_missing_source_draws_and_pairs_identity_exactly(self):
        frame = pd.DataFrame(
            {
                "image_id": ["r1", "r2", "a1", "b1"],
                "label": [0, 0, 1, 1],
                "source": ["real", "real", "A", "B"],
                "dataset": ["COCO", "COCO", "Synthbuster", "Synthbuster"],
                "archive_member": ["", "", "synth/A/a.png", "synth/B/b.png"],
                "anomaly_score": [0.0, 1.0, 0.5, 2.0],
                "clean_anomaly_score": [0.0, 1.0, 0.5, 2.0],
            }
        )
        draws, seed = 600, 29
        weights = _cluster_weights(
            frame.label.to_numpy(),
            frame.source.to_numpy(),
            _proxy_cluster_ids(frame),
            draws,
            np.random.default_rng(seed),
        )
        expected_valid = int(((weights[:, 2] > 0) & (weights[:, 3] > 0)).sum())
        results = pd.DataFrame(
            _condition_metrics(frame, "identity", draws, np.random.default_rng(seed))
        )
        macro = results.loc[results.group == "macro"].iloc[0]
        self.assertEqual(macro.bootstrap_draws_valid, expected_valid)
        self.assertGreater(expected_valid, 0)
        self.assertLess(expected_valid, draws)
        self.assertEqual(macro.auroc, 0.75)
        np.testing.assert_array_equal(results.auroc_delta, 0.0)
        np.testing.assert_array_equal(results.auroc_delta_ci_low, 0.0)
        np.testing.assert_array_equal(results.auroc_delta_ci_high, 0.0)
        # Dropping a missing generator would silently make all draws valid.
        self.assertEqual(macro.bootstrap_draws_missing_class_or_source, draws - expected_valid)

    def test_file_analysis_keeps_main_results_and_reports_missing_expected_pairs(self):
        cache = AN / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="content-proxy-tests-", dir=cache) as temporary:
            run = Path(temporary).resolve()
            self.assertTrue(run.is_relative_to(AN))
            originals, records = [], []
            specs = [
                ("r1", 0, "real", 0.0),
                ("r2", 0, "real", 1.0),
                ("a1", 1, "A", 0.5),
                ("a2", 1, "A", 2.0),
                ("b1", 1, "B", 0.7),
                ("b2", 1, "B", 2.5),
                ("cal", 0, "real", 1000.0),
            ]
            for identifier, label, source, score in specs:
                original = dict(
                    image_id=identifier,
                    label=label,
                    source=source,
                    split="calibration" if identifier == "cal" else "test",
                    dataset="Synthbuster" if label else "COCO",
                    archive_member=(
                        f"synth/{source}/{'shared' if identifier in {'a1', 'b1'} else identifier}.png"
                        if label
                        else ""
                    ),
                )
                originals.append(original)
                for condition in ["clean", "jpeg_q50"]:
                    if identifier == "cal" and condition != "clean":
                        continue
                    if identifier == "b2" and condition == "jpeg_q50":
                        continue
                    records.append(
                        {
                            **original,
                            "condition": condition,
                            "anomaly_score": score if condition == "clean" else -score,
                        }
                    )
            metadata = dict(
                selected_manifest=originals,
                expected_conditions=["clean", "jpeg_q50"],
                expected_image_conditions={
                    row["image_id"]: (
                        ["clean"] if row["split"] == "calibration" else ["clean", "jpeg_q50"]
                    )
                    for row in originals
                },
            )
            (run / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (run / "records.jsonl").write_text(
                "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8"
            )
            frame = pd.DataFrame(records)
            clean = frame.loc[
                (frame.split == "test") & (frame.condition == "clean"),
                ["image_id", "anomaly_score"],
            ]
            clean = clean.rename(columns={"anomaly_score": "clean_anomaly_score"})
            main = []
            for condition in metadata["expected_conditions"]:
                paired = frame.loc[(frame.split == "test") & (frame.condition == condition)].merge(
                    clean, on="image_id", validate="one_to_one"
                )
                main.extend(
                    _cohort_metrics(
                        paired, condition, None, 200, np.random.default_rng(3), "condition_paired"
                    )
                )
            pd.DataFrame(main).to_csv(run / "metrics.csv", index=False)
            before = {
                name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                for name in ("metrics.csv", "run_metadata.json", "records.jsonl")
            }
            summary = analyze_content_sensitivity(run, bootstrap=400, seed=11)
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(summary["n_fake_originals"], 4)
            self.assertEqual(summary["n_fake_proxy_clusters"], 3)
            self.assertEqual(summary["n_cross_generator_shared_proxy_clusters"], 1)
            self.assertEqual(summary["n_fake_in_cross_generator_shared_proxy_clusters"], 2)
            self.assertEqual(summary["test_real_source_counts"], {"real": 2})
            self.assertEqual(
                summary["main_point_estimates_verified"], ["auroc", "clean_auroc", "auroc_delta"]
            )
            self.assertFalse(summary["cluster_identity_confirmed"])
            self.assertFalse(summary["fake_source_counts_fixed"])
            self.assertEqual(summary["coverage"][1]["missing_expected_test"], 1)
            for name, previous in before.items():
                self.assertEqual(hashlib.sha256((run / name).read_bytes()).hexdigest(), previous)
            self.assertTrue((run / "content_sensitivity.md").is_file())
            metrics = pd.read_csv(run / "content_sensitivity_metrics.csv")
            np.testing.assert_allclose(metrics.auroc, metrics.main_auroc, rtol=0, atol=1e-12)
            np.testing.assert_allclose(
                metrics.auroc_delta, metrics.main_auroc_delta, rtol=0, atol=1e-12
            )
            metadata.pop("expected_image_conditions")
            (run / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            no_plan = analyze_content_sensitivity(run, bootstrap=40, seed=11)
            self.assertEqual(no_plan["status"], "complete_for_observed_test_pairs_only")
            self.assertFalse(no_plan["real_fake_content_dependence_modeled"])
            self.assertTrue(
                any(
                    "No expected-image condition plan" in warning for warning in no_plan["warnings"]
                )
            )


if __name__ == "__main__":
    unittest.main()
