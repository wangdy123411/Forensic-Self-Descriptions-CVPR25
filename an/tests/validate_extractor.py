"""Benchmark/equivalence audit for the trained FSD and its final detector score.

Example (run from repository root):
  an/.venv/Scripts/python.exe an/tests/validate_extractor.py --weights an/weights/v1.2.0 --image example.png --device cuda --max-size 1024 --skip-upstream --output an/results/extractor_validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fsd.detector import FSDDetector
from fsd.fsd_computation import compute_fsd
from fsd.projection import apply_projections
from an.robustness.extractor import extract_fsd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-size", type=int, default=64)
    parser.add_argument("--skip-upstream", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    detector = FSDDetector.load(args.weights, device=args.device)
    if args.image:
        with Image.open(args.image) as opened:
            image = opened.convert("RGB")
    else:
        # Controlled numerical fixture, not an evaluation image.
        rng = np.random.default_rng(20261108)
        fixture = rng.integers(0, 256, (131, 149, 3), dtype=np.uint8)
        image = Image.fromarray(fixture)
    kwargs = {k: detector.config["fsd"][k] for k in ["kernel_size", "num_scales", "resize_mode"]}
    kwargs["max_size"] = args.max_size
    methods = ["tiled", "fft"] if args.skip_upstream else ["upstream", "tiled", "fft"]
    report = {
        "torch": torch.__version__,
        "device": args.device,
        "image": str(args.image) if args.image else "random numerical fixture",
        "weights": str(args.weights),
        "config": kwargs,
        "results": {},
        "comparisons": {},
    }
    descriptors = {}
    for method in methods:
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        if method == "upstream":
            descriptor = compute_fsd(image, detector.fre, **kwargs)
            diagnostics = {}
        else:
            result = extract_fsd(image, detector.fre, **kwargs, method=method)
            descriptor, diagnostics = result.fsd, result.diagnostics
        with torch.no_grad():
            projected = apply_projections(
                descriptor[None].to(detector.fre.device), detector.projections
            )
            raw = detector.gmm.score_samples(projected).item()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        metrics = {
            "seconds_including_score": time.perf_counter() - start,
            "raw_score": raw,
            "z_score": (raw - detector.train_mean) / detector.train_std,
            "peak_allocated_mb": (
                torch.cuda.max_memory_allocated() / 2**20
                if args.device.startswith("cuda")
                else None
            ),
            "diagnostics": diagnostics,
        }
        report["results"][method] = metrics
        descriptors[method] = descriptor
        print(json.dumps({"method": method, **metrics}), flush=True)
    reference = methods[0]
    ref = descriptors[reference]
    passed = True
    for method in methods[1:]:
        current = descriptors[method]
        delta = current - ref
        rel_l2 = float(delta.norm() / ref.norm().clamp_min(1e-30))
        abs_z = abs(report["results"][method]["z_score"] - report["results"][reference]["z_score"])
        comparison = {
            "relative_l2": rel_l2,
            "max_abs_difference": float(delta.abs().max()),
            "absolute_z_score_difference": abs_z,
            "passed": rel_l2 < 1e-4 and abs_z < 1e-3,
        }
        passed = passed and comparison["passed"]
        report["comparisons"][f"{method}_vs_{reference}"] = comparison
    report["passed"] = passed
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["comparisons"], indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
