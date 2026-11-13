from __future__ import annotations

from project_paths import ARCHIVE_ROOT, WEIGHTS_ROOT


import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

from zeroshot_common import (
    FAKE_SOURCES, MANIFEST_FIELDS, PREDICTION_FIELDS, REAL_SOURCES,
    check_resume_metadata, child_path, file_hash, load_manifest,
    summarize, write_csv, write_json,
)


def select_rows(rows: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return rows
    if limit <= 0:
        raise ValueError("--limit-per-source must be positive")
    counts, selected = Counter(), []
    for row in rows:
        if counts[row["source"]] < limit:
            selected.append(row)
            counts[row["source"]] += 1
    return selected


def weights_identity(directory: Path) -> tuple[dict, dict]:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    paths = {"config.json": directory / "config.json"}
    for section in ("fre", "gmm", "transforms"):
        name = config[section]["weights_file"]
        paths[name] = child_path(directory, name)
    return {name: file_hash(path) for name, path in sorted(paths.items())}, config


def read_previous_predictions(path: Path, selected: list[dict], threshold: float) -> list[dict]:
    expected = {r["image_id"]: r for r in selected}
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(PREDICTION_FIELDS):
            raise ValueError("Resume predictions have an unexpected CSV schema")
        previous = list(reader)
    ids = set()
    for row in previous:
        item_id = row.get("image_id")
        if item_id in ids or item_id not in expected:
            raise ValueError(f"Unknown or duplicate resumed prediction: {item_id}")
        ids.add(item_id)
        if any(row.get(field) != expected[item_id][field] for field in MANIFEST_FIELDS):
            raise ValueError(f"Resumed prediction no longer matches manifest: {item_id}")
        if row.get("status") not in ("ok", "error") or row.get("error") is None:
            raise ValueError(f"Incomplete or invalid prediction row: {item_id}")
        if float(row["threshold"]) != threshold:
            raise ValueError(f"Threshold differs in saved prediction: {item_id}")
        if row["status"] == "ok":
            z, raw, fake = (float(row[k]) for k in ("z_score", "raw_score", "fake_score"))
            if not all(math.isfinite(v) for v in (z, raw, fake)) or fake != -z:
                raise ValueError(f"Invalid score in resumed prediction: {item_id}")
            if row["predicted_fake"] != str(int(z < threshold)):
                raise ValueError(f"Inconsistent fixed-threshold decision: {item_id}")
        elif any(row.get(k) != "" for k in ("z_score", "raw_score", "fake_score", "predicted_fake")):
            raise ValueError(f"Failed row must not contain scores: {item_id}")
    return previous


def make_identity(args, rows, selected, weights, threshold, fsd_directory, torch) -> dict:
    code_files = sorted(fsd_directory.rglob("*.py"))
    if not code_files:
        raise ValueError(f"No FSD Python files found in {fsd_directory}")
    scripts = (Path(__file__).resolve(), Path(__file__).with_name("zeroshot_common.py").resolve())
    dependency_versions = {}
    for package in ("Pillow", "numpy", "scipy"):
        try:
            dependency_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[package] = "not-installed"
    return {
        "schema_version": 1,
        "manifest_sha256": file_hash(args.manifest),
        "manifest_count": len(rows),
        "selected_image_ids_sha256": hashlib.sha256("\n".join(r["image_id"] for r in selected).encode("utf-8")).hexdigest(),
        "selected_count": len(selected),
        "data_root": str(args.data_root.resolve()),
        "weights_sha256": weights,
        "evaluation_code_sha256": {p.name: file_hash(p) for p in scripts},
        "fsd_code_sha256": {p.relative_to(fsd_directory).as_posix(): file_hash(p) for p in code_files},
        "threshold": threshold, "decision_rule": "z_score < threshold", "positive_label": 1,
        "ranking_score": "-z_score", "device": args.device,
        "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
        "dependency_versions": dependency_versions,
        "python_version": platform.python_version(), "limit_per_source": args.limit_per_source,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='evaluate zeroshot')
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, default=WEIGHTS_ROOT)
    parser.add_argument("--fsd-code-dir", type=Path, help="Optional folder containing the fsd package, useful for staging")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--threads", type=int, help="Optional PyTorch CPU thread count")
    parser.add_argument("--threshold", type=float, help="Fixed z-score threshold; default from model config")
    parser.add_argument("--limit-per-source", type=int, help="Smoke test only; take at most N manifest rows per source")
    parser.add_argument("--resume", action="store_true", help="Resume only if manifest, data, code, weights, device and settings match")
    parser.add_argument("--retry-errors", action="store_true", help="With --resume, retry images that previously failed")
    parser.add_argument("--dry-run", action="store_true", help="Verify manifest and image hashes without loading model weights or writing output")
    args = parser.parse_args()
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be positive")
    if args.retry_errors and not args.resume:
        parser.error("--retry-errors requires --resume")
    if args.limit_per_source is not None and args.limit_per_source <= 0:
        parser.error("--limit-per-source must be positive")
    try:
        root = args.data_root.resolve()
        args.data_root = root
        rows = load_manifest(args.manifest, root)
        selected = select_rows(rows, args.limit_per_source)
        counts = dict(Counter(r["source"] for r in selected))
        print(json.dumps({"validated_manifest_rows": len(rows), "selected_rows": len(selected), "selected_sources": counts}, ensure_ascii=False), flush=True)
        if args.dry_run:
            print(json.dumps({"dry_run": "passed", "all_image_hashes_verified": True, "missing_real_sources": [s for s in REAL_SOURCES if s not in counts], "missing_fake_sources": [s for s in FAKE_SOURCES if s not in counts]}, ensure_ascii=False))
            return 0

        weights, config = weights_identity(args.weights_dir.resolve())
        threshold = float(args.threshold if args.threshold is not None else config["scoring"]["default_threshold"])
        if not math.isfinite(threshold):
            raise ValueError("Threshold must be finite")
        if args.fsd_code_dir is not None:
            code = args.fsd_code_dir.resolve()
            if not (code / "fsd" / "__init__.py").is_file():
                raise ValueError("--fsd-code-dir must contain fsd/__init__.py")
            sys.path.insert(0, str(code))
        import torch
        import fsd
        from fsd import FSDDetector

        if args.threads is not None:
            torch.set_num_threads(args.threads)
        if args.device == "auto":
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available; select --device cpu")
        fsd_directory = Path(fsd.__file__).resolve().parent
        if args.fsd_code_dir is not None and fsd_directory != args.fsd_code_dir.resolve() / "fsd":
            raise ValueError("Imported FSD package does not match --fsd-code-dir")
        identity = make_identity(args, rows, selected, weights, threshold, fsd_directory, torch)
        output = args.output_dir.resolve()
        metadata_path, predictions_path = output / "run_metadata.json", output / "predictions.csv"
        if args.resume:
            if not metadata_path.is_file():
                raise ValueError("--resume requires existing run_metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            check_resume_metadata(metadata["identity"], identity)
            previous = read_previous_predictions(predictions_path, selected, threshold)
        else:
            if output.exists() and any(output.iterdir()):
                raise ValueError("Output directory is not empty; use a new directory or --resume")
            metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "identity": identity}
            previous = []
        if args.retry_errors:
            previous = [r for r in previous if r["status"] == "ok"]
        completed = {r["image_id"] for r in previous}
        todo = [r for r in selected if r["image_id"] not in completed]
        detector = FSDDetector.load(weights_dir=args.weights_dir.resolve(), device=args.device, threshold=threshold, attribution=False) if todo else None
        output.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            write_json(metadata_path, metadata)


        if not predictions_path.exists() or args.resume:
            write_csv(predictions_path, PREDICTION_FIELDS, previous)
        all_predictions = list(previous)
        with predictions_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
            for index, item in enumerate(todo, start=len(previous) + 1):
                started = time.perf_counter()
                prediction = {**item, "status": "error", "z_score": "", "raw_score": "", "fake_score": "", "predicted_fake": "", "threshold": repr(threshold), "elapsed_seconds": "", "error": ""}
                try:
                    path = child_path(root, item["relative_path"])

                    if file_hash(path) != item["sha256"]:
                        raise ValueError("Image bytes changed after manifest verification")
                    result = detector.score(path)
                    z, raw = float(result.z_score), float(result.raw_score)
                    if not (math.isfinite(z) and math.isfinite(raw)):
                        raise ValueError("Detector returned non-finite scores")
                    if float(result.threshold) != threshold or bool(result.is_fake) != (z < threshold):
                        raise ValueError("Detector result disagrees with fixed-threshold decision rule")
                    prediction.update(status="ok", z_score=repr(z), raw_score=repr(raw), fake_score=repr(-z), predicted_fake=str(int(z < threshold)))
                except Exception as error:
                    prediction["error"] = f"{type(error).__name__}: {error}".replace("\r", " ").replace("\n", " ")
                prediction["elapsed_seconds"] = f"{time.perf_counter() - started:.6f}"
                writer.writerow(prediction)
                stream.flush()
                all_predictions.append(prediction)
                print(f"[{index}/{len(selected)}] {item['source']} {prediction['status']}: {item['relative_path']}", flush=True)

        report = summarize(all_predictions, rows, limited=args.limit_per_source is not None)
        report.update(threshold=threshold, manifest_sha256=identity["manifest_sha256"], completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "summary.json", report)
        pair_fields = ("real_source", "fake_source", "n", "n_real", "n_fake", "roc_auc", "average_precision", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "false_positive_rate", "tn", "fp", "fn", "tp")
        write_csv(output / "pairwise_metrics.csv", pair_fields, report["pairs"])
        real_rows = [{"real_source": name, "n_real": info["n_real"], "available_generator_count": info["available_generator_count"], "macro_auc": info["macro_auc"], "macro_auc_scope": info["macro_auc_scope"], "paper_generator_count": info["paper_generator_count"], "paper_generator_macro_auc": info["paper_generator_macro_auc"], "worst_auc": info["worst_auc"], "false_positive_rate": info["real_only_metrics"]["false_positive_rate"]} for name, info in report["per_real_source"].items()]
        write_csv(output / "per_real_source.csv", ("real_source", "n_real", "available_generator_count", "macro_auc", "macro_auc_scope", "paper_generator_count", "paper_generator_macro_auc", "worst_auc", "false_positive_rate"), real_rows)
        print(json.dumps({"summary": str(output / "summary.json"), "coverage": report["coverage_status"], "scored": report["scored_count"], "failed": report["failed_count"], "available_paper_pairs": report["available_expected_pair_count"], "macro_auc_available_paper_pairs": report["macro_auc_available_paper_pairs"]}, ensure_ascii=False, indent=2))
        return 2 if report["failed_count"] else 0
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        parser.exit(2, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
