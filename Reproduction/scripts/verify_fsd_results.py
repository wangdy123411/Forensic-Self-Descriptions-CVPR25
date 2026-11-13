import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def close(actual, expected, name):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool), f"{name}: not numeric")
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), f"{name}: expected {expected}, found {actual}")


def verify(results, data):
    results, data = results.resolve(), data.resolve()
    manifest_path = data / "manifest.csv"
    manifest_rows = read_csv(manifest_path)
    rows = read_csv(results / "predictions.csv")
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8-sig"))
    require(len(manifest_rows) == 200, "Manifest must contain exactly 200 rows")
    require(len(rows) == 200, "Predictions must contain exactly 200 rows")
    manifest = {row["relative_path"]: row for row in manifest_rows}
    require(len(manifest) == 200, "Manifest contains duplicate paths")
    require(len({row["relative_path"] for row in rows}) == 200, "Predictions contain duplicate paths")
    require({row["relative_path"] for row in rows} == set(manifest), "Prediction paths differ from manifest")
    require(summary["manifest_sha256"] == digest(manifest_path), "Summary manifest SHA-256 mismatch")

    confusion = [[0, 0], [0, 0]]
    scores = {0: [], 1: []}
    labels = Counter()
    by_generator = {}
    image_hashes = set()
    for row in rows:
        relative = row["relative_path"]
        expected = manifest[relative]
        require(row["status"] == "ok", f"{relative}: inference status is not ok")
        require(not row["error"].strip(), f"{relative}: nonempty error field")
        label = int(row["label"])
        require(label in (0, 1) and label == int(expected["label"]), f"{relative}: invalid label")
        require(expected["label_name"] == ("real" if label == 0 else "fake"), f"{relative}: manifest label_name mismatch")
        require(row["generator"] == expected["generator"], f"{relative}: generator mismatch")
        require((row["generator"] == "Real") == (label == 0), f"{relative}: generator/label inconsistency")
        path = (data / relative).resolve()
        require(path.is_relative_to(data), f"{relative}: path escapes the data directory")
        require(path.is_file(), f"{relative}: image is missing")
        actual_hash = digest(path)
        require(row["sha256"] == expected["sha256"] == actual_hash, f"{relative}: image SHA-256 mismatch")
        require(path.stat().st_size == int(expected["bytes"]), f"{relative}: image size mismatch")
        require(actual_hash not in image_hashes, f"{relative}: duplicate image content")
        image_hashes.add(actual_hash)
        z, raw, threshold = (float(row[name]) for name in ("z_score", "raw_score", "threshold"))
        require(all(math.isfinite(value) for value in (z, raw, threshold)), f"{relative}: non-finite score")
        close(threshold, -2.0, f"{relative}: threshold")
        predicted, correct = int(row["pred_label"]), int(row["correct"])
        require(predicted in (0, 1) and predicted == int(z < threshold), f"{relative}: pred_label disagrees with z_score")
        require(correct in (0, 1) and correct == int(predicted == label), f"{relative}: incorrect correctness flag")
        labels[label] += 1
        scores[label].append(z)
        confusion[label][predicted] += 1
        if label == 1:
            counts = by_generator.setdefault(row["generator"], {"selected": 0, "detected_fake": 0})
            counts["selected"] += 1
            counts["detected_fake"] += predicted

    require(labels == {0: 100, 1: 100}, "Expected 100 real and 100 fake predictions")
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0


    auc = sum(1.0 if fake < real else 0.5 if fake == real else 0.0
              for fake in scores[1] for real in scores[0]) / (len(scores[1]) * len(scores[0]))
    independent = {"accuracy": (tn + tp) / 200, "precision_fake": precision, "recall_fake": recall, "f1_fake": f1, "roc_auc": auc}
    require(summary["selected_images"] == 200 and summary["successful_images"] == 200 and summary["failed_images"] == 0, "Summary success/selection counts mismatch")
    require(summary["successful_label_counts"] == {"real": 100, "fake": 100}, "Summary label counts mismatch")
    require(summary["selected_label_counts"] == {"real": 100, "fake": 100}, "Summary selected label counts mismatch")
    require(summary["errors"] == [], "Summary contains errors")
    close(summary["threshold"], -2.0, "Summary threshold")
    close(summary["model_config"]["scoring"]["default_threshold"], -2.0, "Model default threshold")
    metrics = summary["metrics"]
    for name, value in independent.items():
        close(metrics[name], value, f"Summary {name}")
    require(metrics["confusion_matrix"] == confusion, "Summary confusion matrix mismatch")
    require(metrics["confusion_matrix_rows"] == ["true_real", "true_fake"], "Confusion matrix row order mismatch")
    require(metrics["confusion_matrix_columns"] == ["pred_real", "pred_fake"], "Confusion matrix column order mismatch")
    require(set(summary["per_generator"]) == set(by_generator), "Summary generator set mismatch")
    for generator, counts in by_generator.items():
        reported = summary["per_generator"][generator]
        require(reported["selected"] == reported["successful"] == counts["selected"] and reported["failed"] == 0, f"{generator}: summary counts mismatch")
        require(reported["detected_fake"] == counts["detected_fake"], f"{generator}: detected count mismatch")
        close(reported["recall_fake"], counts["detected_fake"] / counts["selected"], f"{generator}: recall")
    return {"status": "passed", "images_verified": 200, "counts": {"real": 100, "fake": 100}, "sha256_verified": len(image_hashes), "threshold": -2.0, "independent_metrics": independent, "confusion_matrix": confusion, "auc_pair_comparisons": 10000}


def main():
    parser = argparse.ArgumentParser(description='verify fsd results')
    parser.add_argument("results_directory", type=Path)
    parser.add_argument("data_directory", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.results_directory, args.data_directory)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, allow_nan=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
