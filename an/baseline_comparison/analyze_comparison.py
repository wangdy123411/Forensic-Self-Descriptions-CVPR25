"""CPU analysis of literal-upstream and FFT runs on an exactly matched cohort.

This never imports torch or performs extraction. Main-study headline metrics are
not used: each upstream case is matched to its own frozen main-study record.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "an"
os.environ.setdefault("MPLCONFIGDIR", str(AN / ".cache" / "matplotlib"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    if not Path(path).exists():
        return []
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve(path, base=ROOT):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def clean_json(value):
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def number(value, digits=3):
    return f"{float(value):.{digits}f}" if math.isfinite(float(value)) else "n/a"


def save_json(path, value):
    Path(path).write_text(
        json.dumps(clean_json(value), indent=2, allow_nan=False), encoding="utf-8"
    )


def key(row):
    return str(row["image_id"]), str(row["condition"])


def utc_timestamp(value):
    value = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Runtime interval timestamps must include a timezone")
    return value.astimezone(timezone.utc)


def overlaps_interval(case_start, case_end, window_start, window_end=None, analysis_time=None):
    """Conservative closed-interval overlap; an open end stops at analysis time."""
    start, end = utc_timestamp(case_start), utc_timestamp(case_end)
    lower = utc_timestamp(window_start)
    upper = utc_timestamp(
        window_end
        if window_end is not None
        else (analysis_time if analysis_time is not None else datetime.now(timezone.utc))
    )
    if end < start or upper < lower:
        raise ValueError("Runtime interval end precedes its start")
    return start <= upper and end >= lower


def power_runtime_strata(pairs, records, incident=None, analysis_time=None):
    """Add supplemental timing groups without filtering any comparison case."""
    analysis_time = utc_timestamp(analysis_time or datetime.now(timezone.utc))
    annotated = pairs.copy()
    record_by_key = index_unique(records, "power-timing")
    starts, finishes, flags = [], [], []
    for row in annotated.to_dict("records"):
        source = record_by_key[key(row)]
        start, finish = source["started_utc"], source["finished_utc"]
        # Validate all case timestamps even when no incident has been recorded.
        if utc_timestamp(finish) < utc_timestamp(start):
            raise ValueError("Case finish precedes case start")
        starts.append(start)
        finishes.append(finish)
        flags.append(
            bool(
                incident
                and overlaps_interval(
                    start,
                    finish,
                    incident["timing_window_start_utc"],
                    incident.get("timing_window_end_utc"),
                    analysis_time,
                )
            )
        )
    annotated["case_started_utc"] = starts
    annotated["case_finished_utc"] = finishes
    annotated["power_incident_overlap"] = flags
    annotated["power_timing_stratum"] = [
        (
            "overlaps_incident_window"
            if flag
            else ("outside_incident_window" if incident else "no_incident_recorded")
        )
        for flag in flags
    ]
    effective_end = (
        utc_timestamp(incident.get("timing_window_end_utc") or analysis_time).isoformat()
        if incident
        else None
    )
    summary = {
        "incident_recorded": incident is not None,
        "analysis_time_utc": analysis_time.isoformat(),
        "timing_window_start_utc": incident.get("timing_window_start_utc") if incident else None,
        "timing_window_end_utc": incident.get("timing_window_end_utc") if incident else None,
        "effective_window_end_utc": effective_end,
        "window_open_at_analysis": bool(incident and incident.get("timing_window_end_utc") is None),
        "case_interval_policy": "Inclusive overlap of the entire case started_utc/finished_utc interval with the conservative incident window; equality at either boundary is flagged",
        "all_cases": len(annotated),
        "overlapping_cases": sum(flags),
        "nonoverlapping_cases": len(flags) - sum(flags),
        "cases_excluded_from_numeric_or_task_analysis": 0,
        "all_case_runtime_preserved": True,
        "limitation": "Case-level overlap cannot identify which method's timed section was power-limited. Nonoverlap means outside the recorded window, not continuous proof of AC power. Strata are descriptive and not a randomized estimate of battery-power effects.",
    }
    strata = {"all_cases": annotated}
    if incident:
        strata.update(
            overlaps_incident_window=annotated[annotated.power_incident_overlap],
            outside_incident_window=annotated[~annotated.power_incident_overlap],
        )
    else:
        strata["no_incident_recorded"] = annotated
    conditions = ["all"] + list(dict.fromkeys(annotated.condition))
    output = []
    for name, group in strata.items():
        for condition in conditions:
            selected = group if condition == "all" else group[group.condition == condition]
            output.append(
                {
                    "stratum": name,
                    "condition": condition,
                    "n": len(selected),
                    "upstream_median_seconds": (
                        float(selected.upstream_seconds.median()) if len(selected) else math.nan
                    ),
                    "an_median_seconds": (
                        float(selected.an_seconds.median()) if len(selected) else math.nan
                    ),
                    "median_paired_speedup": (
                        float(selected.paired_speedup.median()) if len(selected) else math.nan
                    ),
                    "upstream_total_seconds": float(selected.upstream_seconds.sum()),
                    "an_total_seconds": float(selected.an_seconds.sum()),
                    "timing_window_start_utc": summary["timing_window_start_utc"],
                    "timing_window_end_utc": summary["timing_window_end_utc"],
                    "effective_window_end_utc": effective_end,
                    "window_open_at_analysis": summary["window_open_at_analysis"],
                    "analysis_time_utc": summary["analysis_time_utc"],
                }
            )
    return annotated, pd.DataFrame(output), summary


def index_unique(rows, description):
    result = {}
    for row in rows:
        pair = key(row)
        if pair in result:
            raise ValueError(f"Duplicate {description} case: {pair}")
        result[pair] = row
    return result


def validate_matching(records, originals, main_records, conditions):
    """Reject mismatched or unexpected cases; explicitly return missing pairs."""
    original_map = {row["image_id"]: row for row in originals}
    if len(original_map) != len(originals):
        raise ValueError("Duplicate selected original IDs")
    if any(row["split"] != "test" for row in originals):
        raise ValueError("Comparison cohort must contain test originals only")
    expected = {(image_id, condition) for image_id in original_map for condition in conditions}
    observed = index_unique(records, "comparison")
    main = index_unique(main_records, "main")
    unexpected = set(observed) - expected
    if unexpected:
        raise ValueError(f"Unexpected comparison cases: {sorted(unexpected)[:5]}")
    for pair, row in observed.items():
        original = original_map[pair[0]]
        if pair not in main:
            raise ValueError(f"Missing matching main case: {pair}")
        reference = main[pair]
        for field in ["source", "split", "label"]:
            if str(row[field]) != str(original[field]) or str(reference[field]) != str(
                original[field]
            ):
                raise ValueError(f"Cohort identity mismatch in {field}: {pair}")
        if row["image_sha256"] != original["sha256"] or reference["sha256"] != original["sha256"]:
            raise ValueError(f"Original image hash mismatch: {pair}")
    return observed, main, expected - set(observed)


def binary_metrics(rows, method, main, released_z, calibrated_anomaly):
    labels = np.asarray([int(row["label"]) for row in rows])
    scores = np.asarray(
        [
            float((main[key(row)] if method == "main" else row[method])["anomaly_score"])
            for row in rows
        ]
    )
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite detection score")
    real, fake = labels == 0, labels == 1
    released = scores > -released_z
    calibrated = scores > calibrated_anomaly
    tpr = float(released[fake].mean()) if fake.any() else math.nan
    fpr = float(released[real].mean()) if real.any() else math.nan
    return {
        "n_real": int(real.sum()),
        "n_fake": int(fake.sum()),
        "auroc": float(roc_auc_score(labels, scores)) if real.any() and fake.any() else math.nan,
        "released_tpr": tpr,
        "released_fpr": fpr,
        "released_balanced_accuracy": (tpr + 1 - fpr) / 2,
        "calibrated_tpr": float(calibrated[fake].mean()) if fake.any() else math.nan,
        "calibrated_fpr": float(calibrated[real].mean()) if real.any() else math.nan,
    }


def recompute_task_metrics(
    records, main, conditions, source_mapping, released_z, calibrated_anomaly
):
    """Use identical original-condition pairs for upstream and cached AN metrics."""
    metrics, attribution = [], []
    fake_sources = sorted({row["source"] for row in records if int(row["label"]) == 1})
    for condition in conditions:
        selected = [row for row in records if row["condition"] == condition]
        if not selected:
            continue
        for method in ["upstream", "main"]:
            per_source = []
            for source in fake_sources:
                group = [
                    row for row in selected if int(row["label"]) == 0 or row["source"] == source
                ]
                values = binary_metrics(group, method, main, released_z, calibrated_anomaly)
                values.update(condition=condition, implementation=method, group=source)
                metrics.append(values)
                if values["n_fake"] and values["n_real"]:
                    per_source.append(values)
            pooled = binary_metrics(selected, method, main, released_z, calibrated_anomaly)
            metrics.append(
                {**pooled, "condition": condition, "implementation": method, "group": "pooled"}
            )
            if per_source:
                macro = {
                    column: float(np.mean([row[column] for row in per_source]))
                    for column in [
                        "auroc",
                        "released_tpr",
                        "released_fpr",
                        "released_balanced_accuracy",
                        "calibrated_tpr",
                        "calibrated_fpr",
                    ]
                }
                metrics.append(
                    {
                        **macro,
                        "n_real": pooled["n_real"],
                        "n_fake": pooled["n_fake"],
                        "condition": condition,
                        "implementation": method,
                        "group": "macro",
                    }
                )
            for cohort, predicate in [
                ("mapped_sources", lambda row: row["source"] in source_mapping),
                (
                    "mapped_synthetic_sources",
                    lambda row: int(row["label"]) == 1 and row["source"] in source_mapping,
                ),
            ]:
                known = [row for row in selected if predicate(row)]
                correct = [
                    str((main[key(row)] if method == "main" else row[method])["attribution_pred"])
                    == source_mapping[row["source"]]
                    for row in known
                ]
                attribution.append(
                    {
                        "condition": condition,
                        "implementation": method,
                        "cohort": cohort,
                        "n": len(known),
                        "accuracy": float(np.mean(correct)) if correct else math.nan,
                    }
                )
    return pd.DataFrame(metrics), pd.DataFrame(attribution)


def array_difference(left, right):
    if left.shape != right.shape or left.shape != (960,):
        raise ValueError(f"Unexpected feature dimensions: {left.shape} vs {right.shape}")
    if (
        left.dtype != np.float64
        or right.dtype != np.float64
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("Features must be finite float64 vectors")
    difference = left - right
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(right), 1e-30)),
    }


def source_probabilities(score):
    names = list(score["attribution_loglik"])
    values = np.asarray([float(score["attribution_loglik"][name]) for name in names])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite source likelihood")
    probabilities = np.exp((values - values.max()) / 750.0)
    probabilities /= probabilities.sum()
    return dict(zip(names, probabilities))


def audit_public_api_records(records, released_z, tolerances, require_both_score_labels=False):
    """Verify recorded literal score()/attribute() checks independently on CPU."""
    score_results, attribute_results = [], []
    for row in records:
        upstream = row["upstream"]
        if "public_score_api_check" in row:
            check = row["public_score_api_check"]
            if check["threshold"] != released_z or check["is_fake"] != (
                check["z_score"] < released_z
            ):
                raise ValueError("Literal score() threshold/decision mismatch")
            expected = {
                "raw_score_abs": abs(check["raw_score"] - upstream["raw_score"]),
                "z_score_abs": abs(check["z_score"] - upstream["z_score"]),
                "default_decision_match": check["is_fake"] == upstream["is_fake"],
            }
            for name in ["raw_score_abs", "z_score_abs"]:
                if not math.isclose(expected[name], check[name], rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(f"Recorded literal score() difference is incorrect: {name}")
            if expected["default_decision_match"] != check["default_decision_match"]:
                raise ValueError("Recorded literal score() decision agreement is incorrect")
            score_results.append(
                {
                    **expected,
                    "image_id": row["image_id"],
                    "condition": row["condition"],
                    "label": int(row["label"]),
                    "numeric_pass": expected["z_score_abs"] <= tolerances["z_score_absolute"],
                }
            )
        for check in row.get("public_attribute_api_checks", []):
            probe = read_json(check["probe_path"])
            if (
                probe["image_sha256"] != row["image_sha256"]
                or probe["condition"] != row["condition"]
            ):
                raise ValueError("Literal attribute() probe identity differs from comparison case")
            if probe.get("status") != "success" or probe.get("requested_max_size") != 1024:
                raise ValueError("Expected a successful default-1024 attribute() probe")
            result = probe["result"]
            if set(result["scores"]) != set(upstream["attribution_probabilities"]):
                raise ValueError("Literal attribute() probability source labels differ")
            expected = {
                "z_score_abs": abs(result["z_score"] - upstream["z_score"]),
                "source_match": result["source"] == upstream["attribution_pred"],
                "default_decision_match": result["is_fake"] == upstream["is_fake"],
                "confidence_abs": abs(result["confidence"] - upstream["attribution_confidence"]),
                "probabilities_max_abs": max(
                    abs(result["scores"][name] - probability)
                    for name, probability in upstream["attribution_probabilities"].items()
                ),
            }
            for name, value in expected.items():
                if isinstance(value, bool):
                    if value != check[name]:
                        raise ValueError(
                            f"Recorded literal attribute() agreement is incorrect: {name}"
                        )
                elif not math.isclose(value, check[name], rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(
                        f"Recorded literal attribute() difference is incorrect: {name}"
                    )
            attribute_results.append(
                {
                    **expected,
                    "image_id": row["image_id"],
                    "condition": row["condition"],
                    "probe_path": check["probe_path"],
                    "numeric_pass": expected["z_score_abs"] <= tolerances["z_score_absolute"]
                    and expected["probabilities_max_abs"] <= tolerances["probabilities_absolute"],
                }
            )
    if require_both_score_labels and (
        len(score_results) != 2 or {item["label"] for item in score_results} != {0, 1}
    ):
        raise ValueError("Expected one completed literal score() check for each binary label")
    return {
        "score": {
            "count": len(score_results),
            "labels": sorted({item["label"] for item in score_results}),
            "max_raw_score_abs": max(
                (item["raw_score_abs"] for item in score_results), default=None
            ),
            "max_z_score_abs": max((item["z_score_abs"] for item in score_results), default=None),
            "all_decisions_match": all(item["default_decision_match"] for item in score_results),
            "all_numeric_pass": all(item["numeric_pass"] for item in score_results),
            "checks": score_results,
        },
        "attribute": {
            "count": len(attribute_results),
            "max_z_score_abs": max(
                (item["z_score_abs"] for item in attribute_results), default=None
            ),
            "max_probabilities_abs": max(
                (item["probabilities_max_abs"] for item in attribute_results), default=None
            ),
            "all_sources_match": all(item["source_match"] for item in attribute_results),
            "all_decisions_match": all(
                item["default_decision_match"] for item in attribute_results
            ),
            "all_numeric_pass": all(item["numeric_pass"] for item in attribute_results),
            "checks": attribute_results,
        },
    }


def paired_differences(
    records, main, run_dir, main_dir, released_z, calibrated_anomaly, tolerances
):
    output = []
    for row in records:
        reference = main[key(row)]
        features = {}
        for method, path in [
            ("upstream", resolve(row["upstream"]["feature_path"], run_dir)),
            ("an", resolve(row["an"]["feature_path"], run_dir)),
            ("main", resolve(reference["feature_path"], main_dir)),
        ]:
            if method != "main" and digest(path) != row[method]["feature_sha256"]:
                raise ValueError(f"Changed {method} feature cache: {path}")
            with np.load(path, allow_pickle=False) as archive:
                features[method] = {
                    name: archive[name].copy() for name in ["raw", "projected", "attribution"]
                }
        values = {
            "image_id": row["image_id"],
            "source": row["source"],
            "label": int(row["label"]),
            "condition": row["condition"],
        }
        scores = {"upstream": row["upstream"], "an": row["an"], "main": reference}
        for method, score in scores.items():
            if not math.isclose(
                float(score["anomaly_score"]), -float(score["z_score"]), abs_tol=1e-12
            ):
                raise ValueError(f"Wrong anomaly-score sign for {method}/{key(row)}")
            probability = source_probabilities(score)
            if score["attribution_pred"] != max(
                score["attribution_loglik"], key=score["attribution_loglik"].get
            ):
                raise ValueError(
                    f"Attribution prediction is not the maximum likelihood source: {method}/{key(row)}"
                )
            if (
                abs(float(score["attribution_confidence"]) - probability[score["attribution_pred"]])
                > 1e-8
            ):
                raise ValueError(
                    f"Attribution confidence does not match the released temperature: {method}/{key(row)}"
                )
            if method != "main":
                if score["is_fake"] != (float(score["z_score"]) < released_z):
                    raise ValueError("Stored released decision differs from its score")
                if score["calibrated_is_fake"] != (
                    float(score["anomaly_score"]) > calibrated_anomaly
                ):
                    raise ValueError("Stored calibrated decision differs from its score")
                if set(score["attribution_probabilities"]) != set(probability):
                    raise ValueError("Probability/source-label mismatch")
                if (
                    max(
                        abs(float(score["attribution_probabilities"][name]) - value)
                        for name, value in probability.items()
                    )
                    > 1e-8
                ):
                    raise ValueError("Stored source probabilities do not match raw likelihoods")
        for left, right in [("upstream", "an"), ("upstream", "main"), ("an", "main")]:
            prefix = f"{left}_vs_{right}"
            for name in ["raw", "projected", "attribution"]:
                for metric, value in array_difference(
                    features[left][name], features[right][name]
                ).items():
                    values[f"{prefix}_{name}_{metric}"] = value
            a, b = scores[left], scores[right]
            for name in ["z_score", "raw_score", "anomaly_score", "attribution_confidence"]:
                values[f"{prefix}_{name}_abs"] = abs(float(a[name]) - float(b[name]))
            values[f"{prefix}_released_decision_match"] = (float(a["z_score"]) < released_z) == (
                float(b["z_score"]) < released_z
            )
            values[f"{prefix}_calibrated_decision_match"] = (
                float(a["anomaly_score"]) > calibrated_anomaly
            ) == (float(b["anomaly_score"]) > calibrated_anomaly)
            values[f"{prefix}_source_match"] = str(a["attribution_pred"]) == str(
                b["attribution_pred"]
            )
            if set(a["attribution_loglik"]) != set(b["attribution_loglik"]):
                raise ValueError("Source GMM labels differ")
            values[f"{prefix}_loglik_max_abs"] = max(
                abs(float(a["attribution_loglik"][name]) - float(b["attribution_loglik"][name]))
                for name in a["attribution_loglik"]
            )
            probabilities_a, probabilities_b = source_probabilities(a), source_probabilities(b)
            values[f"{prefix}_probabilities_max_abs"] = max(
                abs(probabilities_a[name] - probabilities_b[name]) for name in probabilities_a
            )
            values[f"{prefix}_numeric_pass"] = bool(
                all(
                    values[f"{prefix}_{name}_relative_l2"] <= tolerances["feature_relative_l2"]
                    for name in ["raw", "projected", "attribution"]
                )
                and values[f"{prefix}_z_score_abs"] <= tolerances["z_score_absolute"]
                and values[f"{prefix}_probabilities_max_abs"]
                <= tolerances["probabilities_absolute"]
            )
            recorded = row["differences"][prefix]
            for name in ["raw", "projected", "attribution"]:
                for metric in ["max_abs", "relative_l2"]:
                    if not math.isclose(
                        values[f"{prefix}_{name}_{metric}"],
                        recorded[name][metric],
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(
                            f"Recorded feature difference does not match cached vectors: {prefix}/{name}/{metric}"
                        )
            if values[f"{prefix}_numeric_pass"] != recorded["numeric_pass"]:
                raise ValueError(
                    f"Independent numeric verdict differs from recorded verdict: {prefix}"
                )
        for method in ["upstream", "an"]:
            for name in ["seconds", "peak_cuda_allocated_mib", "peak_cuda_reserved_mib"]:
                values[f"{method}_{name}"] = float(row[method][name])
                if not math.isfinite(values[f"{method}_{name}"]) or values[f"{method}_{name}"] <= 0:
                    raise ValueError(f"Invalid timing/allocation measurement: {method}/{name}")
            values[f"{method}_z_score"] = float(row[method]["z_score"])
        values["main_z_score"] = float(reference["z_score"])
        values["paired_speedup"] = values["upstream_seconds"] / max(values["an_seconds"], 1e-30)
        output.append(values)
    return pd.DataFrame(output)


def audit_inputs(run_dir, metadata):
    signature = metadata["signature"]
    main_dir = resolve(metadata["main_run_dir"])
    if main_dir == Path(run_dir).resolve():
        raise ValueError("Analysis output must not overwrite the frozen main study")
    manifest_path = resolve(metadata["manifest_path"])
    checks = []

    def check(path, expected, label):
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"{label} hash changed: {path}")
        checks.append({"path": str(Path(path).relative_to(ROOT)), "sha256": actual, "check": label})

    check(manifest_path, signature["manifest_sha256"], "frozen comparison manifest")
    check(main_dir / "records.jsonl", signature["main_records_sha256"], "main records")
    check(main_dir / "run_metadata.json", signature["main_metadata_sha256"], "main metadata")
    check(
        main_dir / "analysis_summary.json",
        signature["main_analysis_summary_sha256"],
        "main analysis summary",
    )
    for path, expected in signature["code_sha256"].items():
        check(resolve(path), expected, "scoring code")
    weights_dir = resolve(metadata.get("weights_dir", "an/weights/v1.2.0"))
    for filename, expected in signature["weights"].items():
        check(weights_dir / filename, expected, "released weights")
    originals = pd.read_csv(manifest_path, keep_default_na=False).to_dict("records")
    if len({row["sha256"] for row in originals}) != len(originals):
        raise ValueError("Duplicate native image bytes in comparison cohort")
    selection_path = manifest_path.with_name("selection.json")
    selection = read_json(selection_path)
    if selection["manifest"]["sha256"] != signature["manifest_sha256"]:
        raise ValueError("Comparison manifest differs from the independently frozen selection")
    if signature["selected_image_ids"] != [row["image_id"] for row in originals]:
        raise ValueError("Selected ID order differs from frozen manifest")
    main_meta = read_json(main_dir / "run_metadata.json")
    if signature["main_records_sha256"] != main_meta.get("merged_records_sha256"):
        raise ValueError("Main records do not match the merge's independently recorded digest")
    if signature["fsd_config"] != main_meta["signature"]["fsd_config"]:
        raise ValueError(
            "Comparison settings differ from cached main settings; cached main cannot be an equivalence reference"
        )
    main_conditions = {item["name"]: item for item in main_meta["signature"]["conditions"]}
    for condition in signature["conditions"]:
        if condition != main_conditions.get(condition["name"]):
            raise ValueError(f"Processing settings differ for {condition['name']}")
    if signature["device"] != "cuda" or signature["dtype"] != "float64":
        raise ValueError("Expected paired CUDA float64 experiment")
    if signature["weights"] != main_meta["signature"]["weights"]:
        raise ValueError("Literal-upstream and main checkpoint hashes differ")
    main = index_unique(read_jsonl(main_dir / "records.jsonl"), "main")
    for original in originals:
        expected = signature["original_sha256"][original["image_id"]]
        if expected != original["sha256"]:
            raise ValueError("Original hash does not match comparison signature")
        check(resolve(original["path"]), expected, "native original")
    for case, expected in signature["reference_feature_sha256"].items():
        image_id, condition = case.split("|", 1)
        check(
            resolve(main[(image_id, condition)]["feature_path"], main_dir),
            expected,
            "main feature cache",
        )
    for probe in metadata.get("public_attribute_probes", []):
        check(resolve(probe["path"]), probe["sha256"], "literal public API probe")
    return originals, list(main.values()), main_dir, main_meta, checks


def verify_processed_pixels(records, originals):
    """Reconstruct recorded inputs on CPU, without any FSD/model operation."""
    from PIL import Image

    sys.path.insert(0, str(ROOT))
    from an.robustness.transforms import CONDITION_MAP, apply_condition

    grouped = defaultdict(list)
    for row in records:
        grouped[row["image_id"]].append(row)
    for original in originals:
        rows = grouped.get(original["image_id"], [])
        if not rows:
            continue
        with Image.open(resolve(original["path"])) as opened:
            image = opened.convert("RGB")
        for row in rows:
            processed = apply_condition(image, CONDITION_MAP[row["condition"]])
            pixel_digest = hashlib.sha256(
                f"RGB:{processed.width}:{processed.height}:".encode() + processed.tobytes()
            ).hexdigest()
            if (
                pixel_digest != row["processed_pixel_sha256"]
                or list(processed.size) != row["processed_size"]
            ):
                raise ValueError(
                    f"Processed pixels differ from the recorded shared input: {key(row)}"
                )
    return len(records)


def make_plots(pairs, metrics, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for condition, group in pairs.groupby("condition", sort=False):
        axis.scatter(group.main_z_score, group.upstream_z_score, s=24, alpha=0.75, label=condition)
    low = min(pairs.main_z_score.min(), pairs.upstream_z_score.min())
    high = max(pairs.main_z_score.max(), pairs.upstream_z_score.max())
    padding = max((high - low) * 0.05, max(abs(low), abs(high)) * 0.05, 0.25)
    limits = [low - padding, high + padding]
    axis.plot(limits, limits, color="black", linewidth=1, linestyle="--")
    scale = "symlog" if max(abs(low), abs(high)) > 100 else "linear"
    axis.set(
        xscale=scale,
        yscale=scale,
        xlim=limits,
        ylim=limits,
        xlabel="AN main z-score (same original and condition)",
        ylabel="Literal upstream z-score",
        title="Paired score agreement",
    )
    axis.legend(fontsize=7, loc="best")
    figure.savefig(plots / "score_agreement.png", dpi=160)
    plt.close(figure)
    macro = metrics[metrics.group == "macro"]
    conditions = list(dict.fromkeys(macro.condition))
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    if conditions:
        for method, marker in [("upstream", "o"), ("main", "x")]:
            values = (
                macro[macro.implementation == method].set_index("condition").reindex(conditions)
            )
            axis.plot(
                range(len(conditions)),
                values.auroc,
                marker=marker,
                linewidth=1.5,
                label="Literal upstream" if method == "upstream" else "AN main, matched cohort",
            )
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "AUROC unavailable until both real and synthetic\ncomparison cases are scored.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set(
        xticks=range(len(conditions)),
        xticklabels=conditions,
        ylim=(0, 1.05),
        ylabel="Macro AUROC",
        title="Task metrics recomputed on identical comparison cases",
    )
    axis.tick_params(axis="x", labelrotation=25)
    figure.savefig(plots / "matched_cohort_auroc.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    labels = ["Literal upstream", "AN fresh FFT"]
    axes[0].bar(
        labels,
        [pairs.upstream_seconds.median(), pairs.an_seconds.median()],
        color=["#ce6b25", "#2473ab"],
    )
    axes[0].set(ylabel="Median seconds per paired case", title="Same timer scope")
    axes[1].bar(
        labels,
        [
            pairs.upstream_peak_cuda_allocated_mib.max() / 1024,
            pairs.an_peak_cuda_allocated_mib.max() / 1024,
        ],
        color=["#ce6b25", "#2473ab"],
    )
    axes[1].axhline(6, linestyle="--", color="black", linewidth=1, label="Nominal 6 GiB VRAM")
    axes[1].set(ylabel="Peak PyTorch allocated GiB", title="Allocation is not physical residency")
    axes[1].legend(fontsize=8)
    figure.savefig(plots / "runtime_memory.png", dpi=160)
    plt.close(figure)
    return [
        f"plots/{name}.png"
        for name in ["score_agreement", "matched_cohort_auroc", "runtime_memory"]
    ]


def analyze(run_dir, require_complete=False):
    run_dir = Path(run_dir).resolve()
    metadata = read_json(run_dir / "run_metadata.json")
    originals, main_records, main_dir, main_meta, hash_checks = audit_inputs(run_dir, metadata)
    records = read_jsonl(run_dir / "records.jsonl")
    conditions = [item["name"] for item in metadata["signature"]["conditions"]]
    observed, main, missing = validate_matching(records, originals, main_records, conditions)
    if require_complete and missing:
        raise ValueError(f"Comparison incomplete: {len(missing)} expected pairs missing")
    if not records:
        raise ValueError("No completed comparison pairs to analyze")
    source_counts = Counter(row["source"] for row in originals)
    if len(originals) != 25 or len(source_counts) != 5 or set(source_counts.values()) != {5}:
        raise ValueError(f"Expected frozen balanced25-original cohort; got {dict(source_counts)}")
    main_summary = read_json(main_dir / "analysis_summary.json")
    released_z = float(main_summary["released_z_threshold"])
    calibrated_anomaly = float(main_summary["calibrated_anomaly_threshold"])
    if (
        released_z != metadata["released_z_threshold"]
        or calibrated_anomaly != metadata["calibrated_anomaly_threshold"]
    ):
        raise ValueError("Comparison and main decision thresholds differ")
    source_mapping = main_summary["source_mapping"]
    scoring = main_meta["environment"]["model_config"]["scoring"]
    for row in records:
        for values in [row["upstream"], row["an"], main[key(row)]]:
            expected_z = (values["raw_score"] - scoring["train_mean"]) / scoring["train_std"]
            if not math.isclose(expected_z, values["z_score"], rel_tol=1e-12, abs_tol=1e-10):
                raise ValueError(f"Raw-score normalization mismatch: {key(row)}")
    processed_pixel_checks = verify_processed_pixels(records, originals)
    pairs = paired_differences(
        records, main, run_dir, main_dir, released_z, calibrated_anomaly, metadata["tolerances"]
    )
    incident_path = Path(__file__).resolve().parent / "power_incident.json"
    incident = None
    incident_provenance = None
    if incident_path.exists():
        payload = incident_path.read_bytes()
        incident = json.loads(payload)
        incident_provenance = {
            "path": incident_path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "incident": incident,
        }
        status_path = incident_path.with_name("power_status_log.jsonl")
        if status_path.exists():
            incident_provenance["status_log"] = {
                "path": status_path.relative_to(ROOT).as_posix(),
                "sha256_at_analysis": digest(status_path),
            }
    pairs, runtime_power, power_summary = power_runtime_strata(pairs, records, incident)
    save_json(run_dir / "runtime_power_incident.json", power_summary)
    public_api_checks = audit_public_api_records(
        records,
        released_z,
        metadata["tolerances"],
        require_both_score_labels=not missing
        and not metadata["signature"].get("skip_score_api_checks", False),
    )
    metrics, attribution = recompute_task_metrics(
        records, main, conditions, source_mapping, released_z, calibrated_anomaly
    )
    coverage = pd.DataFrame(
        [
            {
                "condition": condition,
                "source": source,
                "expected": count,
                "completed": sum(
                    row["condition"] == condition and row["source"] == source for row in records
                ),
            }
            for condition in conditions
            for source, count in source_counts.items()
        ]
    )
    coverage["missing"] = coverage.expected - coverage.completed
    runtime = []
    for condition in ["all"] + conditions:
        group = pairs if condition == "all" else pairs[pairs.condition == condition]
        if group.empty:
            continue
        runtime.append(
            {
                "condition": condition,
                "n": len(group),
                "upstream_median_seconds": group.upstream_seconds.median(),
                "an_median_seconds": group.an_seconds.median(),
                "median_paired_speedup": group.paired_speedup.median(),
                "upstream_peak_allocated_mib": group.upstream_peak_cuda_allocated_mib.max(),
                "an_peak_allocated_mib": group.an_peak_cuda_allocated_mib.max(),
                "upstream_peak_reserved_mib": group.upstream_peak_cuda_reserved_mib.max(),
                "an_peak_reserved_mib": group.an_peak_cuda_reserved_mib.max(),
            }
        )
    runtime = pd.DataFrame(runtime)
    metric_differences = {}
    for name in [
        "auroc",
        "released_tpr",
        "released_fpr",
        "released_balanced_accuracy",
        "calibrated_tpr",
        "calibrated_fpr",
    ]:
        table = metrics.pivot(index=["condition", "group"], columns="implementation", values=name)
        differences = (table["upstream"] - table["main"]).abs().dropna()
        metric_differences[name] = float(differences.max()) if len(differences) else None
    accuracy = attribution.pivot(
        index=["condition", "cohort"], columns="implementation", values="accuracy"
    )
    accuracy_difference = (accuracy["upstream"] - accuracy["main"]).abs().dropna()
    metric_differences["mapped_attribution_accuracy"] = (
        float(accuracy_difference.max()) if len(accuracy_difference) else None
    )
    for name, frame in [
        ("comparison", pairs),
        ("metrics", metrics),
        ("attribution_metrics", attribution),
        ("coverage", coverage),
        ("runtime", runtime),
        ("runtime_power_strata", runtime_power),
    ]:
        frame.to_csv(run_dir / f"{name}.csv", index=False)
    figures = make_plots(pairs, metrics, run_dir)
    match_columns = [name for name in pairs if name.endswith("_match")]
    difference_columns = [
        name for name in pairs if name.endswith("_abs") or name.endswith("_relative_l2")
    ]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if not missing else "partial",
        "expected_cases": len(originals) * len(conditions),
        "completed_cases": len(records),
        "missing_cases": [list(pair) for pair in sorted(missing)],
        "originals": len(originals),
        "source_counts": dict(source_counts),
        "conditions": conditions,
        "max_differences": {name: float(pairs[name].max()) for name in difference_columns},
        "agreement": {
            name: {
                "matched": int(pairs[name].sum()),
                "total": len(pairs),
                "all_match": bool(pairs[name].all()),
            }
            for name in match_columns
        },
        "independently_recomputed_numeric_pass": {
            name: bool(pairs[name].all()) for name in pairs if name.endswith("_numeric_pass")
        },
        "max_matched_task_metric_differences": metric_differences,
        "released_z_threshold": released_z,
        "calibrated_anomaly_threshold": calibrated_anomaly,
        "calibration_notice": "Frozen threshold inherited from 100 separate COCO calibration originals in main; no comparison test score fits a threshold",
        "metric_cohort_notice": "Both implementations use exactly the same completed original-condition cases; main 300-test-original headline metrics are not compared to this 25-original cohort",
        "timer_notes": metadata.get("timer_notes", {}),
        "tolerances": metadata.get("tolerances", {}),
        "version_comparison": metadata.get("version_comparison", {}),
        "version_warnings": metadata.get("warnings", []),
        "processed_pixel_hashes_reconstructed_and_verified": processed_pixel_checks,
        "hash_checks": hash_checks,
        "figures": figures,
        "errors_recorded": len(read_jsonl(run_dir / "errors.jsonl")),
        "runtime_power_stratification": power_summary,
        "power_incident_provenance": incident_provenance,
        "public_api_record_checks": public_api_checks,
        "main_results_modified": False,
    }
    probe_path = Path(__file__).resolve().parent / "api_probe_audit.json"
    if probe_path.exists():
        probe_audit = read_json(probe_path)
        summary["public_api_probe_evidence"] = {
            "path": probe_path.relative_to(ROOT).as_posix(),
            "sha256": digest(probe_path),
            "passed": bool(probe_audit["passed"]),
            "count": len(probe_audit["results"]),
            "max_absolute_z_difference": max(
                item["absolute_z_difference"] for item in probe_audit["results"]
            ),
            "max_source_probability_difference": max(
                item["maximum_source_probability_difference"] for item in probe_audit["results"]
            ),
            "all_source_matches": all(item["source_match"] for item in probe_audit["results"]),
            "all_decision_matches": all(
                item["released_decision_match"] for item in probe_audit["results"]
            ),
        }
    save_json(run_dir / "comparison_summary.json", summary)
    overall = runtime.iloc[0]
    macro = metrics[metrics.group == "macro"].pivot(
        index="condition", columns="implementation", values="auroc"
    )
    lines = [
        "# Literal upstream versus AN FFT comparison",
        "",
        f"Status: **{summary['status']}**; {len(records)}/{summary['expected_cases']} paired cases completed.",
        "",
        "The frozen cohort contains 25 test originals: 5 COCO photographs and 5 from each of DALL-E 3, Firefly, SDXL, and GLIDE. "
        "Both task-metric columns below are recomputed on exactly the same original-condition cases. "
        "They are not compared with the larger main-study headline result.",
        "",
        f"Released settings: `{metadata['signature']['fsd_config']}`. Float64 features are compared directly from caches.",
        "",
        "| Condition | Literal upstream AUROC | AN main AUROC, same cases |",
        "|---|---:|---:|",
    ]
    for condition in conditions:
        if condition in macro.index:
            lines.append(
                f"| {condition} | {macro.loc[condition, 'upstream']:.6f} | {macro.loc[condition, 'main']:.6f} |"
            )
    lines += [
        "",
        "| Condition | Released TPR, upstream / AN | Released FPR, upstream / AN | Mapped synthetic-source attribution accuracy (15 originals), upstream / AN |",
        "|---|---:|---:|---:|",
    ]
    pooled = metrics[metrics.group == "pooled"].set_index(["condition", "implementation"])
    mapped = attribution[attribution.cohort == "mapped_synthetic_sources"].set_index(
        ["condition", "implementation"]
    )
    for condition in conditions:
        if (condition, "upstream") in pooled.index:
            up, fast = pooled.loc[(condition, "upstream")], pooled.loc[(condition, "main")]
            up_attr, fast_attr = (
                mapped.loc[(condition, "upstream")],
                mapped.loc[(condition, "main")],
            )
            lines.append(
                f"| {condition} | {number(up.released_tpr)} / {number(fast.released_tpr)} | "
                f"{number(up.released_fpr)} / {number(fast.released_fpr)} | {number(up_attr.accuracy)} / {number(fast_attr.accuracy)} |"
            )
    lines += [
        "",
        "AUROC is an equal-weight mean of the four real-versus-generator AUCs. "
        "[metrics.csv](metrics.csv) also reports each source, pooled AUC, TPR/FPR, and balanced accuracy. "
        "The attribution headline uses the 15 mapped synthetic originals from DALL-E 3, Firefly, and SDXL; "
        "COCO and unmapped GLIDE are excluded from that accuracy. "
        "[attribution_metrics.csv](attribution_metrics.csv) retains both the 15-synthetic and 20-mapped-source "
        "(including real) cohorts. The released classifier has no unknown rejection.",
        "",
        f"Largest upstream-versus-main absolute z difference: **{pairs.upstream_vs_main_z_score_abs.max():.3g}**. "
        f"Released decisions agree in **{int(pairs.upstream_vs_main_released_decision_match.sum())}/{len(pairs)}** cases; "
        f"attribution predictions agree in **{int(pairs.upstream_vs_main_source_match.sum())}/{len(pairs)}** cases. "
        "[comparison.csv](comparison.csv) includes raw/projected/attribution feature differences and fresh FFT rerun checks.",
        "",
        f"Fresh paired timing: literal-upstream median **{overall.upstream_median_seconds:.3f}s**, "
        f"AN FFT median **{overall.an_median_seconds:.3f}s**; median per-case speed ratio **{overall.median_paired_speedup:.2f}×**. "
        "Saved main-study timings are not used in this speed comparison.",
        "",
        "Timer scope: " + metadata.get("timer_notes", {}).get("per_method", "Not recorded."),
        "",
        "Diagnostic work: " + metadata.get("timer_notes", {}).get("an_extra", "Not recorded."),
        "",
        "Order and allocator: " + metadata.get("timer_notes", {}).get("allocator", "Not recorded."),
        "",
        "Memory values are PyTorch allocated/reserved peaks, not measurements of physical GPU residency. "
        "On Windows WDDM, allocations beyond the RTX 2060's 6 GiB can involve shared-memory paging; "
        "these measurements do not establish the same behavior on other GPU/OS configurations.",
        "",
        "Processing is applied once to produce the shared upstream/fresh-FFT input. Its recorded pixel hash "
        "does not retroactively prove the earlier main input pixels: the main link is the frozen original, "
        "processing definition/code hashes, and numerical output comparison.",
        "",
        "The purpose is implementation equivalence on this small, fixed cohort. It does not expand the "
        "main benchmark, establish population-level robustness, or remove possible checkpoint-training overlap. "
        "Repeated derivatives and semantically related Synthbuster prompts are not independent observations.",
        "",
        "[Coverage](coverage.csv) · [Runtime and allocation](runtime.csv) · [Full audit JSON](comparison_summary.json)",
        "",
    ]
    limits = metadata["tolerances"]
    lines += [
        f"Declared numerical limits are relative L2 ≤ {limits['feature_relative_l2']:.0e} for each raw/projected/attribution vector, "
        f"absolute z difference ≤ {limits['z_score_absolute']:.0e}, and maximum source-probability difference ≤ {limits['probabilities_absolute']:.0e}. "
        "Independent numeric verdicts: `"
        + json.dumps(summary["independently_recomputed_numeric_pass"])
        + "`. "
        "These numeric tolerances are separate from discrete decision and source agreement. "
        f"The frozen calibrated decisions agree in {int(pairs.upstream_vs_main_calibrated_decision_match.sum())}/{len(pairs)} upstream-versus-main cases.",
        "",
    ]
    score_api = public_api_checks["score"]
    if score_api["count"]:
        lines += [
            f"The literal public `FSDDetector.score()` endpoint was checked in **{score_api['count']} cases** "
            f"covering binary labels {score_api['labels']}. Maximum raw-score difference was "
            f"**{score_api['max_raw_score_abs']:.3g}** and maximum z difference **{score_api['max_z_score_abs']:.3g}**; "
            f"all released decisions agree: **{score_api['all_decisions_match']}**. "
            "These extra endpoint checks are excluded from paired method timing. Their recorded differences "
            "and decisions are independently rechecked by this analyzer.",
            "",
        ]
    attribute_api = public_api_checks["attribute"]
    if attribute_api["count"]:
        lines += [
            f"Within paired records, **{attribute_api['count']} literal `attribute()` probe matches** were also verified "
            f"against their frozen probe files: maximum z difference **{attribute_api['max_z_score_abs']:.3g}**, "
            f"maximum probability difference **{attribute_api['max_probabilities_abs']:.3g}**, "
            f"all sources/decisions match: **{attribute_api['all_sources_match']}/{attribute_api['all_decisions_match']}**.",
            "",
        ]
    if "public_api_probe_evidence" in summary:
        evidence = summary["public_api_probe_evidence"]
        lines += [
            "The separate literal public `FSDDetector.attribute()` probes provide an additional API check: "
            f"**{evidence['count']} probes**, pass={evidence['passed']}, maximum absolute z difference "
            f"**{evidence['max_absolute_z_difference']:.3g}**, and maximum source-probability difference "
            f"**{evidence['max_source_probability_difference']:.3g}**. Their full evidence is recorded in "
            "`an/baseline_comparison/api_probe_audit.json`; these probes are separate from the paired timing table.",
            "",
        ]
    if power_summary["incident_recorded"]:
        end = (
            power_summary["timing_window_end_utc"]
            or f"open through analysis time {power_summary['effective_window_end_utc']}"
        )
        lines += [
            "A recorded battery-power interruption affects timing interpretation. The conservative window is "
            f"**{power_summary['timing_window_start_utc']} to {end}**; "
            f"**{power_summary['overlapping_cases']}/{power_summary['all_cases']} cases** overlap it. "
            "The incident log reports a 10 W GPU cap and 300 MHz SM clock while on battery. "
            "No software or driver power settings were changed.",
            "",
            "All cases remain in numerical comparisons, task metrics, and the overall timing results above. "
            "[runtime_power_strata.csv](runtime_power_strata.csv) supplements those results with whole-case "
            "overlapping/nonoverlapping timing groups. Boundary equality is conservatively flagged. "
            "An open window ends at the recorded analysis time; a closed window uses the recorded AC-restoration "
            "plus 60-second recovery boundary, without adding another recovery period.",
            "",
            "| Timing stratum | Cases | Upstream median seconds | AN median seconds | Median paired speed ratio |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in runtime_power[runtime_power.condition == "all"].to_dict("records"):
            lines.append(
                f"| {row['stratum']} | {row['n']} | {number(row['upstream_median_seconds'])} | "
                f"{number(row['an_median_seconds'])} | {number(row['median_paired_speedup'])} |"
            )
        lines += ["", power_summary["limitation"], ""]
    if missing:
        lines.insert(
            4,
            "**Incomplete coverage:** metrics use the observed matched pairs only; do not treat this as the full frozen-cohort result.\n",
        )
    for figure in figures:
        lines.append(f"![{Path(figure).stem.replace('_', ' ')}]({figure})\n")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    summary = analyze(args.run, args.require_complete)
    print(
        json.dumps(
            {
                name: summary[name]
                for name in [
                    "status",
                    "expected_cases",
                    "completed_cases",
                    "independently_recomputed_numeric_pass",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
