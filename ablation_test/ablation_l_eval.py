"""Evaluate the released FSD model with inference scales L = 1, 2, 3, 4.

Expected dataset layout::

    ablation_test/ablation_img/
      real/
        000001.jpg
      fake/
        stable_diffusion_xl/
          000001.png
        dalle_3/
          000001.png

Images placed directly in ``fake/`` are included in detection evaluation but
excluded from attribution accuracy because they have no source label.

Examples::

    .\.venv\Scripts\python.exe ablation_test\ablation_l_eval.py --list-sources
    .\.venv\Scripts\python.exe ablation_test\ablation_l_eval.py --device cuda
    .\.venv\Scripts\python.exe ablation_test\ablation_l_eval.py --device cpu --no-attribution
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "ablation_test" else SCRIPT_DIR
ABLATION_ROOT = (
    SCRIPT_DIR
    if SCRIPT_DIR.name == "ablation_test"
    else SCRIPT_DIR / "ablation_test"
)

# Keep local package imports working if this script is later moved into
# ablation_test/ and launched directly from any working directory.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from fsd import FSDDetector


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".heif",
    ".heic",
}


@dataclass(frozen=True)
class Sample:
    path: Path
    binary_label: int  # 0 = real, 1 = AI-generated
    source_folder: str
    true_source: str


def normalize_source(name: str) -> str:
    """Convert a model source name to its expected directory name."""
    normalized = name.strip().lower().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def image_files(root: Path):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_samples(data_dir: Path, source_lookup: dict[str, str]) -> tuple[list[Sample], set[str]]:
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"
    if not real_dir.is_dir() or not fake_dir.is_dir():
        raise ValueError(
            f"Expected both '{real_dir}' and '{fake_dir}'. See this script's "
            "module docstring for the required directory layout."
        )

    samples = [Sample(path, 0, "real", "Real") for path in image_files(real_dir)]
    unknown_folders: set[str] = set()

    for path in image_files(fake_dir):
        relative = path.relative_to(fake_dir)
        if len(relative.parts) == 1:
            folder = ""
            true_source = ""
        else:
            folder = relative.parts[0]
            normalized = normalize_source(folder)
            true_source = source_lookup.get(normalized, "")
            if not true_source:
                unknown_folders.add(folder)
        samples.append(Sample(path, 1, folder, true_source))

    samples.sort(key=lambda sample: str(sample.path).lower())
    if not samples:
        raise ValueError(f"No supported images found under '{data_dir}'.")
    if not any(sample.binary_label == 0 for sample in samples):
        raise ValueError("No real images found; detection AUROC requires both labels.")
    if not any(sample.binary_label == 1 for sample in samples):
        raise ValueError("No fake images found; detection AUROC requires both labels.")
    return samples, unknown_folders


def subsample_samples(
    samples: list[Sample],
    max_real: int | None,
    max_per_source: int | None,
    seed: int,
) -> list[Sample]:
    """Select a reproducible subset, independently for real and each fake folder."""
    rng = random.Random(seed)
    real_samples = [sample for sample in samples if sample.binary_label == 0]
    fake_groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        if sample.binary_label == 1:
            # Direct children of fake/ form one unlabelled group.
            fake_groups[sample.source_folder].append(sample)

    def choose(group: list[Sample], limit: int | None) -> list[Sample]:
        if limit is None or len(group) <= limit:
            return list(group)
        return rng.sample(group, limit)

    selected = choose(real_samples, max_real)
    for folder in sorted(fake_groups):
        selected.extend(choose(fake_groups[folder], max_per_source))
    return sorted(selected, key=lambda sample: str(sample.path).lower())


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(
    output_dir: Path,
    scale: int,
    true_sources: list[str],
    predicted_sources: list[str],
) -> None:
    if not true_sources:
        return
    labels = sorted(set(true_sources) | set(predicted_sources))
    matrix = confusion_matrix(true_sources, predicted_sources, labels=labels)
    path = output_dir / f"attribution_confusion_L{scale}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def save_detection_confusion_matrix(
    output_dir: Path,
    scale: int,
    true_labels: list[int],
    predicted_labels: list[int],
) -> None:
    """Save a binary matrix with rows=true and columns=predicted."""
    matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    path = output_dir / f"detection_confusion_L{scale}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", "Real", "Fake"])
        writer.writerow(["Real", *matrix[0].tolist()])
        writer.writerow(["Fake", *matrix[1].tolist()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=ABLATION_ROOT / "ablation_img"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ABLATION_ROOT / "ablation_results"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument(
        "--max-real",
        type=int,
        default=None,
        help="Randomly select at most this many real images.",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="Randomly select at most this many fake images from each source folder.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for image subsampling (default: 42).",
    )
    parser.add_argument(
        "--no-attribution",
        action="store_true",
        help="Skip attribution weights and report detection metrics only.",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Print model source names and valid folder names, then exit.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Do not perform one unmeasured warm-up inference for each L.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scales = list(dict.fromkeys(args.scales))
    if not scales or any(scale < 1 for scale in scales):
        raise SystemExit("Every value passed to --scales must be a positive integer.")
    if args.max_real is not None and args.max_real < 1:
        raise SystemExit("--max-real must be at least 1.")
    if args.max_per_source is not None and args.max_per_source < 1:
        raise SystemExit("--max-per-source must be at least 1.")

    attribution = not args.no_attribution
    print(f"Loading FSD model (device={args.device}, attribution={attribution})...")
    detector = FSDDetector.load(
        device=args.device,
        threshold=args.threshold,
        attribution=attribution,
    )
    device = torch.device(detector.fre.device)
    print(f"Using device: {device}")

    source_names = list(detector.source_gmms.keys()) if attribution else []
    source_lookup = {normalize_source(name): name for name in source_names}

    if args.list_sources:
        if not source_names:
            print("Attribution is disabled; remove --no-attribution to list sources.")
            return
        print("\nModel source -> folder name")
        for name in source_names:
            print(f"  {name} -> {normalize_source(name)}")
        return

    try:
        samples, unknown_folders = discover_samples(args.data_dir, source_lookup)
    except ValueError as error:
        raise SystemExit(f"Dataset error: {error}") from error

    total_before_sampling = len(samples)
    samples = subsample_samples(
        samples,
        max_real=args.max_real,
        max_per_source=args.max_per_source,
        seed=args.seed,
    )

    counts = Counter(sample.binary_label for sample in samples)
    print(
        f"Selected {len(samples)} of {total_before_sampling} images with seed={args.seed}: "
        f"{counts[0]} real, {counts[1]} fake"
    )
    if unknown_folders and attribution:
        print("WARNING: these fake source folders do not match model labels and will be")
        print("excluded from attribution accuracy (but included in detection AUROC):")
        for folder in sorted(unknown_folders):
            print(f"  {folder}")
        print("Run with --list-sources to see valid folder names.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "selected_samples.csv",
        [
            {
                "image_path": str(sample.path),
                "binary_label": sample.binary_label,
                "binary_name": "fake" if sample.binary_label else "real",
                "source_folder": sample.source_folder,
                "true_source": sample.true_source,
                "sampling_seed": args.seed,
            }
            for sample in samples
        ],
        [
            "image_path",
            "binary_label",
            "binary_name",
            "source_folder",
            "true_source",
            "sampling_seed",
        ],
    )
    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    generator_auc_rows: list[dict] = []

    for scale in scales:
        detector.config["fsd"]["num_scales"] = scale
        print(f"\nEvaluating L={scale}...")

        if not args.no_warmup:
            synchronize(device)
            if attribution:
                detector.attribute(samples[0].path)
            else:
                detector.score(samples[0].path)
            synchronize(device)

        scale_rows: list[dict] = []
        for index, sample in enumerate(samples, start=1):
            if device.type == "cuda":
                baseline_memory = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            else:
                baseline_memory = 0

            synchronize(device)
            started = time.perf_counter()
            if attribution:
                result = detector.attribute(sample.path)
                predicted_source = result.source
                confidence = result.confidence
                z_score = result.z_score
                is_fake = result.is_fake
            else:
                result = detector.score(sample.path)
                predicted_source = ""
                confidence = math.nan
                z_score = result.z_score
                is_fake = result.is_fake
            synchronize(device)
            elapsed = time.perf_counter() - started

            if device.type == "cuda":
                peak_memory = torch.cuda.max_memory_allocated(device)
                incremental_peak_mb = max(0, peak_memory - baseline_memory) / (1024**2)
            else:
                incremental_peak_mb = math.nan

            row = {
                "L": scale,
                "image_path": str(sample.path),
                "true_binary_label": sample.binary_label,
                "true_binary_name": "fake" if sample.binary_label else "real",
                "source_folder": sample.source_folder,
                "true_source": sample.true_source,
                "z_score": z_score,
                "fake_score": -z_score,
                "predicted_is_fake": bool(is_fake),
                "predicted_source": predicted_source,
                "source_confidence": confidence,
                "runtime_seconds": elapsed,
                "incremental_peak_gpu_memory_mb": incremental_peak_mb,
            }
            prediction_rows.append(row)
            scale_rows.append(row)
            print(f"  [{index:>5}/{len(samples)}] {sample.path.name}", end="\r", flush=True)
        print(" " * 80, end="\r")

        labels = [row["true_binary_label"] for row in scale_rows]
        fake_scores = [row["fake_score"] for row in scale_rows]
        detection_auc = roc_auc_score(labels, fake_scores)
        predicted_labels = [int(row["predicted_is_fake"]) for row in scale_rows]
        save_detection_confusion_matrix(
            args.output_dir, scale, labels, predicted_labels
        )
        runtimes = [row["runtime_seconds"] for row in scale_rows]
        memories = [
            row["incremental_peak_gpu_memory_mb"]
            for row in scale_rows
            if not math.isnan(row["incremental_peak_gpu_memory_mb"])
        ]

        attribution_rows = [
            row for row in scale_rows if row["true_binary_label"] == 1 and row["true_source"]
        ]
        true_sources = [row["true_source"] for row in attribution_rows]
        predicted_sources = [row["predicted_source"] for row in attribution_rows]
        attribution_accuracy = (
            accuracy_score(true_sources, predicted_sources) if true_sources else math.nan
        )
        save_confusion_matrix(
            args.output_dir, scale, true_sources, predicted_sources
        )

        metric_row = {
            "L": scale,
            "num_images": len(scale_rows),
            "num_real": counts[0],
            "num_fake": counts[1],
            "detection_auroc": detection_auc,
            "attribution_num_images": len(attribution_rows),
            "attribution_top1_accuracy": attribution_accuracy,
            "mean_runtime_seconds": statistics.fmean(runtimes),
            "median_runtime_seconds": statistics.median(runtimes),
            "p95_runtime_seconds": percentile(runtimes, 0.95),
            "mean_incremental_peak_gpu_memory_mb": (
                statistics.fmean(memories) if memories else math.nan
            ),
            "max_incremental_peak_gpu_memory_mb": max(memories) if memories else math.nan,
            "device": str(device),
        }
        metric_rows.append(metric_row)

        real_rows = [row for row in scale_rows if row["true_binary_label"] == 0]
        source_folders = sorted(
            {row["source_folder"] for row in scale_rows if row["true_binary_label"] == 1}
        )
        for source_folder in source_folders:
            fake_rows = [
                row
                for row in scale_rows
                if row["true_binary_label"] == 1 and row["source_folder"] == source_folder
            ]
            pair = real_rows + fake_rows
            if not fake_rows:
                continue
            generator_auc_rows.append(
                {
                    "L": scale,
                    "source_folder": source_folder or "unlabelled_fake",
                    "num_real": len(real_rows),
                    "num_fake": len(fake_rows),
                    "detection_auroc": roc_auc_score(
                        [row["true_binary_label"] for row in pair],
                        [row["fake_score"] for row in pair],
                    ),
                }
            )

        print(
            f"L={scale}: AUROC={detection_auc:.4f}, "
            f"attribution={attribution_accuracy:.4f}, "
            f"median={statistics.median(runtimes):.4f}s"
        )

    write_csv(
        args.output_dir / "predictions.csv",
        prediction_rows,
        [
            "L",
            "image_path",
            "true_binary_label",
            "true_binary_name",
            "source_folder",
            "true_source",
            "z_score",
            "fake_score",
            "predicted_is_fake",
            "predicted_source",
            "source_confidence",
            "runtime_seconds",
            "incremental_peak_gpu_memory_mb",
        ],
    )
    write_csv(
        args.output_dir / "metrics_by_l.csv",
        metric_rows,
        [
            "L",
            "num_images",
            "num_real",
            "num_fake",
            "detection_auroc",
            "attribution_num_images",
            "attribution_top1_accuracy",
            "mean_runtime_seconds",
            "median_runtime_seconds",
            "p95_runtime_seconds",
            "mean_incremental_peak_gpu_memory_mb",
            "max_incremental_peak_gpu_memory_mb",
            "device",
        ],
    )
    write_csv(
        args.output_dir / "detection_auroc_by_generator.csv",
        generator_auc_rows,
        ["L", "source_folder", "num_real", "num_fake", "detection_auroc"],
    )
    print(f"\nFinished. Results written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
