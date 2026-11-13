from __future__ import annotations

from project_paths import PROJECT_ROOT, CODE_ROOT, WEIGHTS_ROOT


import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from itertools import zip_longest
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_argument(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description='run fsd eval')
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-dir", default="Data/tiny_genimage_smoke_200")
    parser.add_argument("--weights-dir", default=str(WEIGHTS_ROOT))
    parser.add_argument("--output-dir", default="Results/tiny_genimage_smoke_200")
    parser.add_argument("--device", default="cuda", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument("--limit", type=int, help="Smoke test: at most N images, alternating real/fake")
    parser.add_argument("--threads", type=int, help="Optional PyTorch CPU thread count")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be positive")

    root = args.root.expanduser().resolve()
    code = CODE_ROOT
    data = resolve_argument(root, args.data_dir)
    weights = resolve_argument(root, args.weights_dir)
    output = resolve_argument(root, args.output_dir)
    predictions_path = output / "predictions.csv"
    if predictions_path.exists():
        parser.error(f"Refusing to overwrite {predictions_path}; choose another --output-dir")
    manifest = data / "manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("The manifest is empty")
    paths = set()
    for row in rows:
        label = int(row["label"])
        if label not in (0, 1) or row["label_name"] != ("real" if label == 0 else "fake"):
            raise ValueError(f"Invalid label: {row['relative_path']}")
        if (label == 0) != (row["generator"] == "Real"):
            raise ValueError(f"Inconsistent generator label: {row['relative_path']}")
        image_path = (data / row["relative_path"]).resolve()
        if not image_path.is_relative_to(data) or image_path in paths:
            raise ValueError(f"Duplicate or unsafe image path: {row['relative_path']}")
        paths.add(image_path)
    all_rows = len(rows)
    selection = "All rows in manifest order"
    if args.limit is not None and args.limit < len(rows):
        real = [r for r in rows if int(r["label"]) == 0]
        fake = [r for r in rows if int(r["label"]) == 1]
        rows = [r for pair in zip_longest(real, fake) for r in pair if r is not None][:args.limit]
        selection = f"First {args.limit} rows after alternating real/fake, preserving within-class order"

    sys.path.insert(0, str(code))
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
    import fsd
    from fsd import FSDDetector

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    weight_names = ["config.json", "fre.pt", "gmm.pt", "fsd_transforms.pt"]
    weight_hashes = {name: sha256(weights / name) for name in weight_names}
    download_manifest = weights / "download_manifest.json"
    if download_manifest.exists():
        expected = {entry["name"]: entry["sha256"] for entry in json.loads(download_manifest.read_text(encoding="utf-8"))["assets"]}
        for name, digest in weight_hashes.items():
            if name not in expected or digest != expected[name]:
                raise RuntimeError(f"Weight checksum mismatch: {name}")
    try:
        commit = subprocess.run(["git", "-C", str(code), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision_file = code / "code_revision.txt"
        commit = revision_file.read_text(encoding="utf-8").strip() if revision_file.exists() else None
    source_hashes = {path.relative_to(code).as_posix(): sha256(path) for path in sorted((code / "fsd").glob("*.py"))}
    started = datetime.now(timezone.utc).isoformat()
    start_clock = time.perf_counter()
    print(f"Loading FSD on {device}; selected {len(rows)} of {all_rows} images", flush=True)
    detector = FSDDetector.load(weights_dir=weights, device=device, attribution=False)
    gpu = None
    if detector.fre.device.type == "cuda":
        properties = torch.cuda.get_device_properties(detector.fre.device)
        gpu = {"name": properties.name, "total_memory_bytes": properties.total_memory, "compute_capability": [properties.major, properties.minor]}
        torch.cuda.synchronize(detector.fre.device)
    model_load_seconds = time.perf_counter() - start_clock
    versions = {}
    for package in ("torch", "numpy", "scipy", "pillow", "scikit-learn"):
        versions[package] = importlib.metadata.version(package)
    run_info = {
        "started_at_utc": started, "root": str(root), "data_dir": str(data), "weights_dir": str(weights),
        "output_dir": str(output), "manifest_sha256": sha256(manifest), "source_manifest_rows": all_rows,
        "selected_images": len(rows), "selection": selection,
        "selected_label_counts": dict(Counter(row["label_name"] for row in rows)),
        "label_mapping": {"0": "real", "1": "fake"}, "positive_class": "fake",
        "decision_rule": "is_fake = z_score < threshold", "auc_score": "-z_score (larger means more likely fake)",
        "threshold": detector.threshold, "model_config": detector.config,
        "device": str(detector.fre.device), "gpu": gpu, "torch_cuda_version": torch.version.cuda,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "torch_cpu_threads": torch.get_num_threads(), "python": sys.version, "platform": platform.platform(),
        "packages": versions, "fsd_version": fsd.__version__, "code_commit": commit,
        "source_sha256": source_hashes, "weight_sha256": weight_hashes,
        "runner_sha256": sha256(Path(__file__)), "model_load_seconds": model_load_seconds,
        "note": "Community convenience subset for checking inference. Not the full FSD paper benchmark. No preprocessing or model configuration changes were made. Scores are not probabilities."
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_info.json", run_info)
    fields = ["relative_path", "label", "generator", "z_score", "raw_score", "threshold", "pred_label", "correct", "elapsed_seconds", "gpu_peak_allocated_mib", "gpu_peak_reserved_mib", "sha256", "status", "error"]
    predictions = []
    inference_start = time.perf_counter()
    with predictions_path.open("x", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        target.flush()
        for index, row in enumerate(rows, 1):
            timer = time.perf_counter()
            record = {field: "" for field in fields}
            record.update(relative_path=row["relative_path"], label=int(row["label"]), generator=row["generator"], threshold=detector.threshold)
            try:
                path = (data / row["relative_path"]).resolve()
                digest = sha256(path)
                record["sha256"] = digest
                if digest != row["sha256"]:
                    raise ValueError("Image SHA-256 mismatch")
                if gpu is not None:
                    torch.cuda.reset_peak_memory_stats(detector.fre.device)
                result = detector.score(path)
                if gpu is not None:
                    torch.cuda.synchronize(detector.fre.device)
                    record["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated(detector.fre.device) / 1048576
                    record["gpu_peak_reserved_mib"] = torch.cuda.max_memory_reserved(detector.fre.device) / 1048576
                if not (math.isfinite(result.z_score) and math.isfinite(result.raw_score)):
                    raise ValueError("Model produced a non-finite score")
                prediction = int(result.is_fake)
                record.update(z_score=float(result.z_score), raw_score=float(result.raw_score), threshold=float(result.threshold), pred_label=prediction, correct=int(prediction == int(row["label"])), status="ok")
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}".replace("\n", " "))
                if gpu is not None:
                    torch.cuda.empty_cache()
            record["elapsed_seconds"] = time.perf_counter() - timer
            writer.writerow(record)
            target.flush()
            predictions.append(record)
            print(f"[{index}/{len(rows)}] {row['relative_path']} status={record['status']} z={record['z_score']} pred={record['pred_label']} true={record['label']} seconds={record['elapsed_seconds']:.3f}" + (f" error={record['error']}" if record["error"] else ""), flush=True)

    good = [row for row in predictions if row["status"] == "ok"]
    bad = [row for row in predictions if row["status"] != "ok"]
    metrics = None
    if good:
        truth = np.array([row["label"] for row in good])
        predicted = np.array([row["pred_label"] for row in good])
        precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="binary", pos_label=1, zero_division=0)
        metrics = {
            "accuracy": float(accuracy_score(truth, predicted)), "precision_fake": float(precision),
            "recall_fake": float(recall), "f1_fake": float(f1),
            "roc_auc": float(roc_auc_score(truth, [-row["z_score"] for row in good])) if len(set(truth.tolist())) == 2 else None,
            "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
            "confusion_matrix_rows": ["true_real", "true_fake"], "confusion_matrix_columns": ["pred_real", "pred_fake"],
            "computed_on": "Successful images only; failures are excluded and reported separately"
        }
    per_generator = {}
    for generator in sorted({row["generator"] for row in predictions if row["label"] == 1}):
        selected = [row for row in predictions if row["generator"] == generator and row["label"] == 1]
        successful = [row for row in selected if row["status"] == "ok"]
        per_generator[generator] = {"selected": len(selected), "successful": len(successful), "failed": len(selected) - len(successful), "detected_fake": sum(row["pred_label"] for row in successful), "recall_fake": sum(row["pred_label"] for row in successful) / len(successful) if successful else None}
    summary = {
        **run_info, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "successful_images": len(good), "failed_images": len(bad),
        "successful_label_counts": dict(Counter("real" if row["label"] == 0 else "fake" for row in good)),
        "metrics": metrics, "per_generator": per_generator,
        "inference_seconds": time.perf_counter() - inference_start,
        "total_seconds": time.perf_counter() - start_clock,
        "mean_image_seconds": sum(row["elapsed_seconds"] for row in predictions) / len(predictions),
        "peak_gpu_allocated_mib": max((row["gpu_peak_allocated_mib"] for row in good if row["gpu_peak_allocated_mib"] != ""), default=None),
        "peak_gpu_reserved_mib": max((row["gpu_peak_reserved_mib"] for row in good if row["gpu_peak_reserved_mib"] != ""), default=None),
        "errors": [{"relative_path": row["relative_path"], "error": row["error"]} for row in bad]
    }
    write_json(output / "summary.json", summary)
    print("COMPLETE " + json.dumps({"successful": len(good), "failed": len(bad), "metrics": metrics, "output_dir": str(output)}, allow_nan=False), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
