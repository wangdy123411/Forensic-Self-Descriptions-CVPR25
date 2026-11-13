from __future__ import annotations

from project_paths import PROJECT_ROOT, CODE_ROOT, DATA_ROOT, WEIGHTS_ROOT


import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
CODE = CODE_ROOT
PROJECT = PROJECT_ROOT
sys.path.insert(0, str(CODE))

from evaluate_zeroshot import weights_identity
from zeroshot_common import child_path, file_hash, write_json
from feature_cache import FeatureCache


def compare_results(reference, actual, tolerance: float) -> dict:
    left, right = asdict(reference), asdict(actual)
    errors = {key: abs(left[key] - right[key]) for key in ("raw_score", "z_score")}
    finite = all(math.isfinite(value) for key in errors for value in (left[key], right[key]))
    same_decision = left["is_fake"] == right["is_fake"] and left["threshold"] == right["threshold"]
    return {
        "passed": finite and same_decision and all(value <= tolerance for value in errors.values()),
        "exact": left == right,
        "absolute_error": errors,
        "same_decision_and_threshold": same_decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='verify cached scoring')
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--weights-dir", type=Path, default=WEIGHTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "Results" / "zeroshot_20261106" / "feature_cache_validation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()
    if args.threads < 1 or not math.isfinite(args.atol) or args.atol < 0:
        parser.error("threads must be positive and atol must be finite and nonnegative")
    if not args.device.startswith("cuda"):
        parser.error("This validation requires the local CUDA device")

    import numpy as np
    import torch
    from fsd import FSDDetector
    from feature_pipeline import raw_dimensions, score_and_cache

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = (args.manifest or args.data_root / "manifests" / "eval_unique.csv").resolve()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    selected = []
    for label in ("0", "1"):
        try:
            selected.append(next(item for item in manifest_rows if item["label"] == label))
        except StopIteration:
            raise ValueError("Parity validation needs one real and one synthetic manifest image") from None

    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir.resolve()
    weights_hashes, config = weights_identity(args.weights_dir.resolve())
    source_hashes = {
        str(path.relative_to(CODE)).replace("\\", "/"): file_hash(path)
        for path in [*sorted((CODE / "fsd").glob("*.py")), HERE / "feature_pipeline.py", HERE / "feature_cache.py", Path(__file__).resolve()]
    }
    detector = FSDDetector.load(args.weights_dir, device=args.device)
    identity = {
        "purpose": "standalone_scoring_parity",
        "weights_sha256": weights_hashes,
        "model_config": config,
        "code_sha256": source_hashes,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "torch_threads": args.threads,
        "seed": 2026,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    report = {
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "absolute_tolerance": args.atol, "relative_tolerance": 0.0,
        "manifest_sha256": file_hash(manifest), "identity": identity,
        "cache_directory": str(output / ("cache_" + stamp)), "images": [],
    }
    report_path = output / ("parity_" + stamp + ".json")
    try:
        for item in selected:
            path = child_path(args.data_root, item["relative_path"])
            if file_hash(path) != item["sha256"]:
                raise ValueError(f"Image SHA256 differs from manifest: {item['relative_path']}")
            print(f"Comparing released, cold-cache and reloaded-cache scoring: {item['source']} / {item['image_id']}", flush=True)
            baseline = detector.score(path)
            original_compute = detector.compute_fsd
            extracted = []

            def track_extraction(image):
                raw = original_compute(image)
                extracted.append(raw.detach().cpu().numpy().copy())
                return raw

            detector.compute_fsd = track_extraction
            try:
                with FeatureCache(Path(report["cache_directory"]), identity, dimensions=raw_dimensions(detector)) as cache:
                    cold, cold_hit = score_and_cache(detector, path, item, cache)

                with FeatureCache(Path(report["cache_directory"]), identity, dimensions=raw_dimensions(detector)) as cache:
                    loaded = cache.get(item)
                    reloaded, reload_hit = score_and_cache(detector, path, item, cache)
            finally:
                detector.compute_fsd = original_compute

            bytes_preserved = len(extracted) == 1 and loaded is not None and (
                loaded.dtype == np.dtype("float64") and loaded.shape == extracted[0].shape
                and loaded.tobytes() == extracted[0].tobytes()
            )
            cold_comparison = compare_results(baseline, cold, args.atol)
            reload_comparison = compare_results(baseline, reloaded, args.atol)
            entry = {
                "image_id": item["image_id"], "source": item["source"], "label": item["label"],
                "sha256": item["sha256"], "baseline": asdict(baseline),
                "cold_cache_result": asdict(cold), "reloaded_cache_result": asdict(reloaded),
                "cold_cache_hit": cold_hit, "reloaded_cache_hit": reload_hit,
                "new_pipeline_extraction_count": len(extracted),
                "raw_feature_bytes_preserved": bytes_preserved,
                "cold_cache_comparison": cold_comparison,
                "reloaded_cache_comparison": reload_comparison,
            }
            entry["passed"] = (not cold_hit and reload_hit and bytes_preserved
                               and cold_comparison["passed"] and reload_comparison["passed"])
            report["images"].append(entry)
            write_json(report_path, report)
        report["status"] = "passed" if all(entry["passed"] for entry in report["images"]) else "failed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(report_path, report)
        write_json(output / "latest.json", report)
        print(f"Parity validation {report['status']}: {report_path}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
