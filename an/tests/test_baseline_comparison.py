"""CPU tests that prevent mismatched-cohort claims in implementation comparison."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline_comparison"))
from analyze_comparison import (
    audit_public_api_records,
    overlaps_interval,
    power_runtime_strata,
    recompute_task_metrics,
    validate_matching,
)


def original(identifier, source, label):
    return dict(
        image_id=identifier, source=source, label=label, split="test", sha256=identifier * 64
    )


def record(item, score, condition="clean", prediction=None):
    value = {
        **item,
        "condition": condition,
        "anomaly_score": score,
        "z_score": -score,
        "attribution_pred": prediction or ("Real" if not item["label"] else item["source"]),
    }
    return value


def comparison(item, score, condition="clean", prediction=None):
    return {
        **item,
        "condition": condition,
        "image_sha256": item["sha256"],
        "upstream": record(item, score, condition, prediction),
    }


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.real = original("r", "COCO2017", 0)
        self.fake = original("f", "DALL-E 3", 1)
        self.originals = [self.real, self.fake]
        self.main = [record(self.real, 0.0), record(self.fake, 3.0)]
        self.rows = [comparison(self.real, 0.0), comparison(self.fake, 3.0)]

    def test_missing_cases_are_explicit_and_duplicate_or_mislabeled_cases_fail(self):
        _, _, missing = validate_matching(self.rows[:1], self.originals, self.main, ["clean"])
        self.assertEqual(missing, {("f", "clean")})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_matching(self.rows + self.rows[:1], self.originals, self.main, ["clean"])
        changed = copy.deepcopy(self.rows)
        changed[0]["source"] = "wrong"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            validate_matching(changed, self.originals, self.main, ["clean"])

    def test_changed_native_hash_and_unexpected_condition_fail(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["image_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_matching(changed, self.originals, self.main, ["clean"])
        changed[0]["condition"] = "jpeg_q50"
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            validate_matching(changed, self.originals, self.main, ["clean"])

    def test_metrics_exclude_unmatched_main_originals(self):
        extra_real = original("x", "COCO2017", 0)
        extra_fake = original("y", "DALL-E 3", 1)
        main_rows = self.main + [record(extra_real, 100.0), record(extra_fake, -100.0)]
        _, main, _ = validate_matching(self.rows, self.originals, main_rows, ["clean"])
        metrics, _ = recompute_task_metrics(
            self.rows, main, ["clean"], {"COCO2017": "Real", "DALL-E 3": "DALL-E 3"}, -2.0, 1.0
        )
        pooled = metrics[metrics.group == "pooled"]
        self.assertEqual(set(pooled.implementation), {"upstream", "main"})
        self.assertTrue((pooled.n_real == 1).all())
        self.assertTrue((pooled.n_fake == 1).all())
        self.assertTrue((pooled.auroc == 1).all())
        self.assertTrue((pooled.released_tpr == 1).all())
        self.assertTrue((pooled.released_fpr == 0).all())

    def test_unknown_source_contributes_to_detection_but_not_attribution_accuracy(self):
        unknown = original("g", "GLIDE", 1)
        rows = self.rows + [comparison(unknown, 4.0, prediction="DALL-E 3")]
        main_rows = self.main + [record(unknown, 4.0, prediction="DALL-E 3")]
        _, main, _ = validate_matching(rows, self.originals + [unknown], main_rows, ["clean"])
        metrics, attribution = recompute_task_metrics(
            rows, main, ["clean"], {"COCO2017": "Real", "DALL-E 3": "DALL-E 3"}, -2.0, 1.0
        )
        self.assertTrue((metrics[metrics.group == "pooled"].n_fake == 2).all())
        self.assertTrue((attribution[attribution.cohort == "mapped_sources"].n == 2).all())
        self.assertTrue(
            (attribution[attribution.cohort == "mapped_synthetic_sources"].n == 1).all()
        )
        self.assertTrue((attribution.accuracy == 1).all())


class PowerIntervalTests(unittest.TestCase):
    def test_closed_window_boundaries_are_conservatively_inclusive(self):
        start, end = "2026-11-08T10:00:00Z", "2026-11-08T10:10:00Z"
        self.assertTrue(overlaps_interval("2026-11-08T09:59:00Z", start, start, end))
        self.assertTrue(overlaps_interval(end, "2026-11-08T10:11:00Z", start, end))
        self.assertTrue(
            overlaps_interval("2026-11-08T09:59:00Z", "2026-11-08T10:11:00Z", start, end)
        )
        self.assertFalse(
            overlaps_interval("2026-11-08T09:58:00Z", "2026-11-08T09:59:59.999999Z", start, end)
        )
        self.assertFalse(
            overlaps_interval("2026-11-08T10:10:00.000001Z", "2026-11-08T10:11:00Z", start, end)
        )

    def test_open_window_uses_analysis_time_and_normalizes_offsets(self):
        self.assertTrue(
            overlaps_interval(
                "2026-11-08T18:05:00+08:00",
                "2026-11-08T18:06:00+08:00",
                "2026-11-08T10:00:00Z",
                analysis_time="2026-11-08T10:10:00Z",
            )
        )
        self.assertFalse(
            overlaps_interval(
                "2026-11-08T10:11:00Z",
                "2026-11-08T10:12:00Z",
                "2026-11-08T10:00:00Z",
                analysis_time="2026-11-08T10:10:00Z",
            )
        )
        with self.assertRaisesRegex(ValueError, "timezone"):
            overlaps_interval(
                "2026-11-08T10:05:00",
                "2026-11-08T10:06:00Z",
                "2026-11-08T10:00:00Z",
                analysis_time="2026-11-08T10:10:00Z",
            )

    def test_stratification_preserves_all_cases_and_overall_timings(self):
        rows = [
            dict(
                image_id=str(i),
                condition="clean",
                started_utc=f"2026-11-08T10:0{i}:00Z",
                finished_utc=f"2026-11-08T10:0{i}:30Z",
            )
            for i in range(3)
        ]
        pairs = pd.DataFrame(
            [
                dict(
                    image_id=str(i),
                    condition="clean",
                    upstream_seconds=float(2**i),
                    an_seconds=0.5,
                    paired_speedup=float(2 ** (i + 1)),
                )
                for i in range(3)
            ]
        )
        incident = {
            "timing_window_start_utc": "2026-11-08T10:01:00Z",
            "timing_window_end_utc": "2026-11-08T10:01:30Z",
        }
        annotated, strata, summary = power_runtime_strata(
            pairs, rows, incident, "2026-11-08T10:03:00Z"
        )
        self.assertEqual(len(annotated), len(pairs))
        self.assertEqual(annotated.power_incident_overlap.tolist(), [False, True, False])
        overall = strata[(strata.stratum == "all_cases") & (strata.condition == "all")].iloc[0]
        self.assertEqual(overall.n, 3)
        self.assertEqual(overall.upstream_median_seconds, pairs.upstream_seconds.median())
        self.assertEqual(summary["cases_excluded_from_numeric_or_task_analysis"], 0)
        self.assertEqual(summary["overlapping_cases"], 1)


class PublicAPITests(unittest.TestCase):
    def test_exact_score_api_checks_for_both_labels_and_reject_bad_decision(self):
        records = []
        for label, z in [(0, 0.5), (1, -3.0)]:
            is_fake = z < -2.0
            upstream = dict(raw_score=z * 10, z_score=z, is_fake=is_fake)
            check = dict(
                raw_score=z * 10,
                z_score=z,
                is_fake=is_fake,
                threshold=-2.0,
                raw_score_abs=0.0,
                z_score_abs=0.0,
                default_decision_match=True,
            )
            records.append(
                dict(
                    image_id=str(label),
                    label=label,
                    condition="clean",
                    upstream=upstream,
                    public_score_api_check=check,
                )
            )
        result = audit_public_api_records(records, -2.0, {"z_score_absolute": 1e-3}, True)
        self.assertEqual(result["score"]["count"], 2)
        self.assertEqual(result["score"]["max_raw_score_abs"], 0)
        self.assertEqual(result["score"]["max_z_score_abs"], 0)
        self.assertTrue(result["score"]["all_decisions_match"])
        records[0]["public_score_api_check"]["is_fake"] = True
        with self.assertRaisesRegex(ValueError, "threshold/decision mismatch"):
            audit_public_api_records(records, -2.0, {"z_score_absolute": 1e-3}, True)


if __name__ == "__main__":
    unittest.main()
