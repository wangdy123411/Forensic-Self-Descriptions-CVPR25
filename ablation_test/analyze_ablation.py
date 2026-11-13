"""Analyze saved FSD ablation predictions without rerunning inference.

Produces scale-specific threshold calibration, open-set attribution metrics,
and stratified paired bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ABLATION_ROOT = (
    SCRIPT_DIR
    if SCRIPT_DIR.name == "ablation_test"
    else SCRIPT_DIR / "ablation_test"
)

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ABLATION_ROOT / "ablation_results" / "predictions.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ABLATION_ROOT / "ablation_analysis"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path):
    required = {
        "L", "image_path", "true_binary_label", "source_folder", "true_source",
        "z_score", "predicted_source", "source_confidence",
    }
    if not path.is_file():
        raise SystemExit(f"Predictions file not found: {path}")
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing columns: {sorted(missing)}")
        for raw in reader:
            try:
                confidence = float(raw["source_confidence"])
            except ValueError:
                confidence = math.nan
            z_score = float(raw["z_score"])
            rows.append({
                "L": int(raw["L"]),
                "path": raw["image_path"],
                "label": int(raw["true_binary_label"]),
                "folder": raw["source_folder"],
                "true_source": raw["true_source"],
                "z_score": z_score,
                "fake_score": -z_score,
                "predicted_source": raw["predicted_source"],
                "confidence": confidence,
            })
    return rows


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stratum(row):
    return "real" if row["label"] == 0 else f"fake::{row['folder']}"


def rates(labels, predictions):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "real_accuracy": tn / (tn + fp),
        "fake_accuracy": tp / (tp + fn),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def optimal_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid = np.flatnonzero(np.isfinite(thresholds))
    return float(thresholds[valid[np.argmax((tpr - fpr)[valid])]])


def calibration(rows_by_l, folds, seed):
    fold_output, summary = [], []
    for scale, rows in sorted(rows_by_l.items()):
        labels = np.asarray([r["label"] for r in rows], dtype=int)
        scores = np.asarray([r["fake_score"] for r in rows], dtype=float)
        strata = np.asarray([stratum(r) for r in rows])
        smallest = min(np.unique(strata, return_counts=True)[1])
        if smallest < folds:
            raise SystemExit(f"Smallest source group ({smallest}) is smaller than --folds={folds}.")

        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
        out_of_fold = np.zeros_like(labels)
        thresholds = []
        for fold, (train, test) in enumerate(splitter.split(scores, strata), 1):
            threshold = optimal_threshold(labels[train], scores[train])
            predictions = (scores[test] >= threshold).astype(int)
            out_of_fold[test] = predictions
            thresholds.append(threshold)
            fold_output.append({
                "L": scale, "fold": fold, "num_calibration": len(train),
                "num_test": len(test), "fake_score_threshold": threshold,
                "equivalent_z_threshold": -threshold,
                **rates(labels[test], predictions),
            })

        default = rates(labels, (scores > 2.0).astype(int))
        calibrated = rates(labels, out_of_fold)
        summary.append({
            "L": scale,
            "detection_auroc": roc_auc_score(labels, scores),
            "default_balanced_accuracy": default["balanced_accuracy"],
            "calibrated_oof_balanced_accuracy": calibrated["balanced_accuracy"],
            "calibrated_oof_accuracy": calibrated["accuracy"],
            "calibrated_real_accuracy": calibrated["real_accuracy"],
            "calibrated_fake_accuracy": calibrated["fake_accuracy"],
            "mean_fake_score_threshold": np.mean(thresholds),
            "std_fake_score_threshold": np.std(thresholds, ddof=1),
            "mean_equivalent_z_threshold": -np.mean(thresholds),
        })
    return fold_output, summary


def fpr95(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.flatnonzero(tpr >= 0.95)
    return float(np.min(fpr[valid])) if len(valid) else math.nan


def oscr(known, unknown):
    known_conf = np.asarray([r["confidence"] for r in known])
    unknown_conf = np.asarray([r["confidence"] for r in unknown])
    correct = np.asarray([r["predicted_source"] == r["true_source"] for r in known])
    thresholds = np.r_[np.inf, np.unique(np.r_[known_conf, unknown_conf])[::-1], -np.inf]
    points = []
    for threshold in thresholds:
        points.append((np.mean(unknown_conf >= threshold), np.mean(correct & (known_conf >= threshold))))
    points.sort()
    x, y = map(np.asarray, zip(*points))
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def open_set(rows_by_l):
    output = []
    for scale, rows in sorted(rows_by_l.items()):
        fake = [r for r in rows if r["label"] == 1 and not math.isnan(r["confidence"])]
        known = [r for r in fake if r["true_source"]]
        unknown = [r for r in fake if not r["true_source"]]
        if not known or not unknown:
            raise SystemExit("Open-set analysis requires known and unknown synthetic sources.")
        combined = known + unknown
        labels = np.asarray([0] * len(known) + [1] * len(unknown))
        unknown_scores = np.asarray([1.0 - r["confidence"] for r in combined])
        output.append({
            "L": scale,
            "num_known": len(known), "num_unknown": len(unknown),
            "known_top1_accuracy": np.mean([r["predicted_source"] == r["true_source"] for r in known]),
            "mean_known_confidence": np.mean([r["confidence"] for r in known]),
            "mean_unknown_confidence": np.mean([r["confidence"] for r in unknown]),
            "unknown_rejection_auroc": roc_auc_score(labels, unknown_scores),
            "unknown_fpr_at_95_tpr": fpr95(labels, unknown_scores),
            "au_oscr": oscr(known, unknown),
        })
    return output


METRICS = (
    "detection_auroc", "default_balanced_accuracy",
    "known_attribution_accuracy", "unknown_rejection_auroc",
)


def metric(rows, name):
    if name == "detection_auroc":
        return roc_auc_score([r["label"] for r in rows], [r["fake_score"] for r in rows])
    if name == "default_balanced_accuracy":
        return balanced_accuracy_score([r["label"] for r in rows], [r["fake_score"] > 2 for r in rows])
    if name == "known_attribution_accuracy":
        subset = [r for r in rows if r["label"] == 1 and r["true_source"]]
        return np.mean([r["predicted_source"] == r["true_source"] for r in subset])
    if name == "unknown_rejection_auroc":
        subset = [r for r in rows if r["label"] == 1 and not math.isnan(r["confidence"])]
        return roc_auc_score(
            [0 if r["true_source"] else 1 for r in subset],
            [1 - r["confidence"] for r in subset],
        )
    raise ValueError(name)


def interval(values):
    return np.mean(values), np.percentile(values, 2.5), np.percentile(values, 97.5)


def bootstrap(rows_by_l, repetitions, seed):
    scales = sorted(rows_by_l)
    reference = 3 if 3 in scales else scales[-1]
    lookup = {(scale, r["path"]): r for scale, rows in rows_by_l.items() for r in rows}
    groups = defaultdict(list)
    for row in rows_by_l[scales[0]]:
        groups[stratum(row)].append(row["path"])

    distributions = {(scale, name): [] for scale in scales for name in METRICS}
    differences = {(other, name): [] for other in scales if other != reference for name in METRICS}
    rng = np.random.default_rng(seed)
    for repetition in range(1, repetitions + 1):
        sampled = []
        for group in sorted(groups):
            paths = groups[group]
            sampled.extend(rng.choice(paths, len(paths), replace=True))
        values = {}
        for scale in scales:
            sample_rows = [lookup[(scale, path)] for path in sampled]
            for name in METRICS:
                values[scale, name] = metric(sample_rows, name)
                distributions[scale, name].append(values[scale, name])
        for other in scales:
            if other != reference:
                for name in METRICS:
                    differences[other, name].append(values[reference, name] - values[other, name])
        if repetition % max(1, repetitions // 10) == 0:
            print(f"Bootstrap {repetition}/{repetitions}", end="\r", flush=True)
    print(" " * 50, end="\r")

    ci_rows = []
    for (scale, name), values in sorted(distributions.items()):
        mean, low, high = interval(values)
        ci_rows.append({"L": scale, "metric": name, "bootstrap_mean": mean,
                        "ci95_lower": low, "ci95_upper": high, "repetitions": repetitions})
    comparison_rows = []
    for (other, name), values in sorted(differences.items()):
        mean, low, high = interval(values)
        comparison_rows.append({
            "reference_L": reference, "comparison_L": other, "metric": name,
            "mean_paired_difference": mean, "ci95_lower": low, "ci95_upper": high,
            "reference_better_at_95pct": low > 0, "repetitions": repetitions,
        })
    return ci_rows, comparison_rows


def main():
    args = arguments()
    if args.folds < 2 or args.bootstrap < 1:
        raise SystemExit("--folds must be >=2 and --bootstrap must be >=1.")
    rows_by_l = defaultdict(list)
    for row in load_rows(args.predictions):
        rows_by_l[row["L"]].append(row)
    for rows in rows_by_l.values():
        rows.sort(key=lambda r: r["path"].lower())
    path_sets = [{r["path"] for r in rows} for rows in rows_by_l.values()]
    if any(paths != path_sets[0] for paths in path_sets[1:]):
        raise SystemExit("Each L must contain the same image paths.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(path_sets[0])} images for L={sorted(rows_by_l)}; no inference will run.")
    fold_rows, calibrated = calibration(rows_by_l, args.folds, args.seed)
    open_rows = open_set(rows_by_l)

    # Save the deterministic analyses before the potentially long bootstrap.
    save_csv(args.output_dir / "calibration_folds.csv", fold_rows, list(fold_rows[0]))
    save_csv(args.output_dir / "calibrated_detection_metrics.csv", calibrated, list(calibrated[0]))
    save_csv(args.output_dir / "open_set_attribution_metrics.csv", open_rows, list(open_rows[0]))

    print(f"Running {args.bootstrap} paired bootstrap repetitions...")
    ci_rows, comparisons = bootstrap(rows_by_l, args.bootstrap, args.seed)

    save_csv(args.output_dir / "bootstrap_confidence_intervals.csv", ci_rows, list(ci_rows[0]))
    save_csv(args.output_dir / "paired_scale_comparisons.csv", comparisons, list(comparisons[0]))

    print("\nCalibration summary:")
    for row in calibrated:
        print(f"L={row['L']}: AUROC={row['detection_auroc']:.4f}, "
              f"default BA={row['default_balanced_accuracy']:.4f}, "
              f"calibrated BA={row['calibrated_oof_balanced_accuracy']:.4f}, "
              f"mean z threshold={row['mean_equivalent_z_threshold']:.3f}")
    print("\nOpen-set summary:")
    for row in open_rows:
        print(f"L={row['L']}: known acc={row['known_top1_accuracy']:.4f}, "
              f"unknown AUROC={row['unknown_rejection_auroc']:.4f}, "
              f"AU-OSCR={row['au_oscr']:.4f}")
    print(f"\nSaved analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
