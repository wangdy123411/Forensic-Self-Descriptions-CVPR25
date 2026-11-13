"""Statistical correctness checks using fabricated scores, not benchmark results."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robustness.analysis import (
    _cohort_metrics,
    _draw_pointwise_intervals,
    _real_domain_metrics,
    _resample_weights,
    _weighted_auroc,
    analyze,
)


class BootstrapTests(unittest.TestCase):
    def test_percentile_interval_plot_preserves_endpoints_outside_the_point_estimate(self):
        os.environ.setdefault(
            "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib")
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        try:
            # A percentile interval need not contain the observed estimate.
            _draw_pointwise_intervals(
                ax,
                np.array([0.4, 0.8]),
                np.array([0.5, 0.7]),
                np.array([0.7, 0.9]),
                np.array([0.0, 1.0]),
                "Example",
                "black",
            )
            segments = ax.collections[0].get_segments()
            np.testing.assert_allclose(segments[0], [[0.5, 0.0], [0.7, 0.0]])
            np.testing.assert_allclose(segments[1], [[0.7, 1.0], [0.9, 1.0]])
            np.testing.assert_allclose(ax.lines[0].get_xdata(), [0.4, 0.8])
        finally:
            plt.close(fig)

    def test_real_domains_are_separated_with_the_same_frozen_threshold(self):
        labels = np.array([0] * 10 + [0] * 10 + [1] * 10)
        sources = np.array(["real_a"] * 10 + ["real_b"] * 10 + ["fake"] * 10)
        scores = np.array([0.0] * 10 + [3.0] * 10 + [1.0] * 10)
        frame = pd.DataFrame(
            dict(label=labels, source=sources, anomaly_score=scores, clean_anomaly_score=scores)
        )
        results = _real_domain_metrics([frame], ["clean"], ["real_a", "real_b"], 0.95, 40, 42)
        macro = results.loc[results.group == "macro"].set_index("real_domain")
        self.assertEqual(macro.loc["real_a", "auroc"], 1.0)
        self.assertEqual(macro.loc["real_b", "auroc"], 0.0)
        self.assertEqual(macro.loc["real_a", "released_fpr"], 0.0)
        self.assertEqual(macro.loc["real_b", "released_fpr"], 1.0)
        self.assertEqual(macro.loc["real_a", "calibrated_fpr"], 0.0)
        self.assertEqual(macro.loc["real_b", "calibrated_fpr"], 1.0)
        np.testing.assert_array_equal(macro.n_real, 10)
        np.testing.assert_array_equal(macro.n_fake, 10)

    def test_weighted_auc_matches_sklearn_with_ties_and_reversed_scores(self):
        rng = np.random.default_rng(13)
        labels = np.array([0] * 17 + [1] * 19)
        scores = rng.integers(0, 5, len(labels)).astype(float)
        sources = np.array(["real"] * 17 + ["fake"] * 19)
        weights = _resample_weights(labels, sources, 40, rng)
        actual = _weighted_auroc(labels, scores, weights)
        expected = [roc_auc_score(labels, scores, sample_weight=row) for row in weights]
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        np.testing.assert_allclose(_weighted_auroc(labels, -scores, weights), 1 - actual)
        np.testing.assert_array_equal(weights[:, :17].sum(1), 17)
        np.testing.assert_array_equal(weights[:, 17:].sum(1), 19)

    def test_macro_weights_generators_equally_and_paired_identity_has_zero_ci(self):
        labels = np.array([0] * 10 + [1] * 4 + [1] * 12)
        sources = np.array(["real"] * 10 + ["small"] * 4 + ["large"] * 12)
        # Small source is perfectly ranked; the larger source is fully reversed.
        scores = np.array([1.0] * 10 + [2.0] * 4 + [0.0] * 12)
        frame = pd.DataFrame(
            dict(label=labels, source=sources, anomaly_score=scores, clean_anomaly_score=scores)
        )
        rows = _cohort_metrics(frame, "clean", 1.5, 100, np.random.default_rng(1), "paired")
        values = {row["group"]: row for row in rows}
        self.assertEqual(values["macro"]["auroc"], 0.5)
        self.assertEqual(values["pooled"]["auroc"], 0.25)
        for row in rows:
            self.assertEqual(row["auroc_delta"], 0)
            self.assertEqual(row["auroc_delta_ci_low"], 0)
            self.assertEqual(row["auroc_delta_ci_high"], 0)


class AnalysisIntegrationTests(unittest.TestCase):
    def make_fixture(self, target: Path):
        rows, manifest, expected = [], [], {}
        for index in range(100):
            image_id = f"cal_{index}"
            rows.append(
                dict(
                    image_id=image_id,
                    label=0,
                    source="real",
                    split="calibration",
                    condition="clean",
                    anomaly_score=index / 100,
                    family="clean",
                    severity=0,
                    seconds=99.0,
                    peak_cuda_mb=999.0,
                )
            )
            expected[image_id] = ["clean"]
        # A corrupted calibration image must not change the clean-only threshold.
        rows.append(
            dict(
                image_id="cal_0",
                label=0,
                source="real",
                split="calibration",
                condition="jpeg_q50",
                anomaly_score=1e9,
                family="jpeg",
                severity=3,
                seconds=77.0,
                peak_cuda_mb=777.0,
            )
        )
        for label, source in ((0, "real"), (1, "fake_a"), (1, "fake_b")):
            for index in range(20):
                image_id = f"{source}_{index}"
                manifest.append(dict(image_id=image_id, label=label, source=source, split="test"))
                expected[image_id] = ["clean", "jpeg_q50"]
                for condition in ("clean", "jpeg_q50"):
                    if condition != "clean" and source == "fake_b" and index < 2:
                        continue
                    score = label if condition == "clean" else 1 - label
                    rows.append(
                        dict(
                            image_id=image_id,
                            label=label,
                            source=source,
                            split="test",
                            condition=condition,
                            family="clean" if condition == "clean" else "jpeg",
                            severity=0 if condition == "clean" else 3,
                            anomaly_score=score,
                            attribution_pred=source,
                            attribution_confidence=0.8,
                            attribution_loglik={"fake_a": 0.0, "fake_b": 1.0},
                            fsd_l2_relative=0 if condition == "clean" else 0.4,
                            fsd_cosine_distance=0.1,
                            projected_l2_relative=0.3,
                            projected_cosine_distance=0.1,
                            residual_rms=1.0,
                            residual_rms_ratio=1.0,
                            seconds=0.3,
                            peak_cuda_mb=45.0,
                            diagnostics=dict(
                                raw_residual_rms=2.0,
                                native_residual_patch_hf_fraction=0.4,
                                raw_residual_rms_channels=[2.0] * 8,
                            ),
                        )
                    )
        (target / "records.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
        )
        metadata = dict(
            expected_conditions=["clean", "jpeg_q50"],
            selected_manifest=manifest,
            expected_image_conditions=expected,
        )
        (target / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    @patch("robustness.analysis._plot_results", return_value=[])
    def test_clean_calibration_missing_pairs_and_complete_common_cohort(self, _plot):
        with tempfile.TemporaryDirectory(prefix="an-analysis-tests-") as dirname:
            target = Path(dirname)
            self.make_fixture(target)
            result = analyze(
                target, bootstrap=80, source_mapping={"fake_a": "fake_a", "fake_b": "wrong_name"}
            )
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["n_common_test"], 58)
            self.assertEqual(result["missing_expected_image_conditions"], 2)
            self.assertEqual(result["calibrated_anomaly_threshold"], 0.95)
            self.assertEqual(result["calibration_empirical_fpr"], 0.04)
            self.assertEqual(result["source_mapping"], {"fake_a": "fake_a"})
            metrics = pd.read_csv(target / "metrics.csv")
            row = metrics[(metrics.group == "macro") & (metrics.condition == "jpeg_q50")].iloc[0]
            self.assertEqual(row.auroc, 0.0)
            self.assertEqual(row.clean_auroc, 1.0)
            self.assertEqual(row.auroc_delta, -1.0)
            self.assertEqual(row.auroc_delta_ci_low, -1.0)
            self.assertEqual(row.auroc_delta_ci_high, -1.0)
            self.assertEqual(row.calibrated_prediction_flip_rate, 1.0)
            self.assertEqual(row.calibrated_fpr, 1.0)
            self.assertEqual(row.calibrated_tpr, 0.0)
            common = pd.read_csv(target / "metrics_common_cohort.csv")
            np.testing.assert_array_equal(common.loc[common.group == "pooled", "n_real"], [20, 20])
            np.testing.assert_array_equal(common.loc[common.group == "pooled", "n_fake"], [38, 38])
            drift = pd.read_csv(target / "feature_drift.csv")
            np.testing.assert_allclose(drift.raw_residual_rms_ratio_median, 1.0)
            np.testing.assert_allclose(drift.native_hf_fraction_delta_median, 0.0)
            runtime = pd.read_csv(target / "runtime.csv").set_index("condition")
            self.assertEqual(runtime.loc["clean", "n"], 60)
            self.assertEqual(runtime.loc["jpeg_q50", "n"], 58)
            np.testing.assert_allclose(runtime.mean_seconds, 0.3)
            np.testing.assert_allclose(runtime.median_seconds, 0.3)
            np.testing.assert_allclose(runtime.max_cuda_mb, 45.0)
            plot_cohort = _plot.call_args.args[2]
            self.assertTrue((plot_cohort["split"] == "test").all())
            pd.testing.assert_series_equal(
                runtime["n"], plot_cohort.groupby("condition").size(), check_names=False
            )
            calibration_runtime = pd.read_csv(target / "runtime_calibration.csv").set_index(
                "condition"
            )
            self.assertEqual(calibration_runtime.loc["clean", "n"], 100)
            self.assertEqual(calibration_runtime.loc["clean", "mean_seconds"], 99.0)
            self.assertEqual(calibration_runtime.loc["jpeg_q50", "n"], 1)
            self.assertEqual(calibration_runtime.loc["jpeg_q50", "mean_seconds"], 77.0)
            self.assertIn("calibration excluded", result["runtime_cohort"])
            self.assertIn("not a reproduction", (target / "report.md").read_text(encoding="utf-8"))

    @patch("robustness.analysis._plot_results", return_value=[])
    def test_wholly_absent_condition_is_not_silently_complete(self, _plot):
        with tempfile.TemporaryDirectory(prefix="an-analysis-tests-") as dirname:
            target = Path(dirname)
            self.make_fixture(target)
            path = target / "run_metadata.json"
            metadata = json.loads(path.read_text())
            metadata["expected_conditions"].append("resize_050")
            for row in metadata["selected_manifest"]:
                metadata["expected_image_conditions"][row["image_id"]].append("resize_050")
            path.write_text(json.dumps(metadata), encoding="utf-8")
            result = analyze(target, bootstrap=0)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["n_common_test"], 0)
            self.assertEqual(result["missing_expected_image_conditions"], 62)
            coverage = pd.read_csv(target / "coverage.csv").set_index("condition")
            self.assertEqual(coverage.loc["resize_050", "paired_test"], 0)
            self.assertTrue(pd.read_csv(target / "metrics_common_cohort.csv").empty)


if __name__ == "__main__":
    unittest.main()
