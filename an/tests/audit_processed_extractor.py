"""Full-resolution tiled-vs-FFT audit on real dataset files and hard conditions.

Numerical fixtures are explicitly separated from dataset images. These are
implementation validation measurements, never detector-accuracy evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fsd.detector import FSDDetector
from fsd.projection import apply_projections
from an.robustness.extractor import extract_fsd
from an.robustness.transforms import CONDITION_MAP, apply_condition


def image_candidates(manifest, per_source):
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        if row.get("split") != "test":
            continue
        groups.setdefault(row["source"], []).append(row)
    selected = []
    for source, group in sorted(groups.items()):
        group.sort(key=lambda row: hashlib.sha256(row["image_id"].encode()).hexdigest())
        # Select lower-texture examples from a deterministic small candidate pool.
        # This deliberately stresses conditioning, not accuracy estimation.
        candidates = []
        for row in group[: max(8, per_source)]:
            path = Path(row["path"])
            if not path.is_absolute():
                possibilities = [ROOT / path, ROOT / "an" / path, manifest.parent / path]
                path = next(
                    (candidate for candidate in possibilities if candidate.is_file()),
                    possibilities[0],
                )
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            gray = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float64)
            texture = float(
                np.mean(np.abs(np.diff(gray, axis=0))) + np.mean(np.abs(np.diff(gray, axis=1)))
            )
            candidates.append((texture, row, path, image))
        candidates.sort(key=lambda candidate: candidate[0])
        for texture, row, path, image in candidates[:per_source]:
            selected.append(
                (
                    {
                        "image_id": row["image_id"],
                        "source": source,
                        "label": int(row["label"]),
                        "kind": "dataset",
                        "path": str(path),
                        "texture_selection_metric": texture,
                    },
                    image,
                )
            )
    return selected


def fixtures():
    yield (
        {"image_id": "fixture_constant", "source": "numerical_fixture", "kind": "fixture"},
        Image.new("RGB", (256, 256), (128, 128, 128)),
    )
    yy, xx = np.indices((256, 320))
    gradient = 50 + 0.2 * xx + 0.3 * yy
    noise = np.random.default_rng(6192).normal(0, 0.5, gradient.shape)
    pixels = np.clip(np.round(gradient + noise), 0, 255).astype(np.uint8)
    yield (
        {"image_id": "fixture_low_texture", "source": "numerical_fixture", "kind": "fixture"},
        Image.fromarray(pixels).convert("RGB"),
    )


@torch.no_grad()
def score(detector, fsd):
    projected = apply_projections(fsd[None].to(detector.fre.device), detector.projections)
    raw = detector.gmm.score_samples(projected).item()
    attr = apply_projections(projected, detector.attribution_projections)
    loglik = {
        name: float(gmm.score_samples(attr).item())
        for name, gmm in (detector.source_gmms or {}).items()
    }
    return {
        "z_score": (raw - detector.train_mean) / detector.train_std,
        "attribution_prediction": max(loglik, key=loglik.get) if loglik else None,
        "attribution_loglik": loglik,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT / "an/weights/v1.2.0")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "an/results/extractor_processed_audit.json"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--per-source", type=int, default=1)
    parser.add_argument("--no-fixtures", action="store_true")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["clean", "blur_sigma20", "crop_area025", "resize_025", "jpeg_q50"],
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    selected = image_candidates(args.manifest, args.per_source) if args.manifest else []
    if not args.no_fixtures:
        selected.extend(fixtures())
    if not selected:
        raise ValueError("No audit images selected")
    detector = FSDDetector.load(args.weights, device=args.device, attribution=True)
    report = {
        "purpose": "implementation validation, not accuracy evaluation",
        "torch": torch.__version__,
        "device": args.device,
        "fsd_config": detector.config["fsd"],
        "results": [],
        "tolerances": {"relative_l2": 1e-4, "absolute_z_difference": 1e-3},
        "passed": True,
        "state": "running",
        "expected_pairs": len(selected) * len(args.conditions),
        "completed_pairs": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for metadata, image in selected:
        for name in args.conditions:
            processed = apply_condition(image, CONDITION_MAP[name])
            results = {}
            for method in ["tiled", "fft"]:
                start = time.perf_counter()
                result = extract_fsd(
                    processed, detector.fre, **detector.config["fsd"], method=method
                )
                z = score(detector, result.fsd)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                results[method] = (result, z, time.perf_counter() - start)
            tiled, fft = results["tiled"][0].fsd, results["fft"][0].fsd
            delta = fft - tiled
            relative_l2 = float(delta.norm() / tiled.norm().clamp_min(1e-30))
            abs_z = abs(results["tiled"][1]["z_score"] - results["fft"][1]["z_score"])
            same_source = (
                results["tiled"][1]["attribution_prediction"]
                == results["fft"][1]["attribution_prediction"]
            )
            attr_difference = max(
                (
                    abs(results["tiled"][1]["attribution_loglik"][source] - value)
                    for source, value in results["fft"][1]["attribution_loglik"].items()
                ),
                default=0,
            )
            record = {
                **metadata,
                "condition": name,
                "processed_size": list(processed.size),
                "relative_l2": relative_l2,
                "max_abs_difference": float(delta.abs().max()),
                "absolute_z_difference": abs_z,
                "attribution_prediction_identical": same_source,
                "max_attribution_loglik_abs_difference": attr_difference,
                "passed": relative_l2 < 1e-4 and abs_z < 1e-3 and same_source,
                "methods": {
                    method: {**value[1], "seconds": value[2], "diagnostics": value[0].diagnostics}
                    for method, value in results.items()
                },
            }
            report["results"].append(record)
            report["completed_pairs"] = len(report["results"])
            if report["completed_pairs"] == report["expected_pairs"]:
                report["state"] = "complete"
            report["passed"] = report["passed"] and record["passed"]
            temp = args.output.with_suffix(".tmp")
            temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temp.replace(args.output)
            print(
                json.dumps(
                    {
                        key: record[key]
                        for key in [
                            "image_id",
                            "condition",
                            "relative_l2",
                            "absolute_z_difference",
                            "passed",
                        ]
                    }
                ),
                flush=True,
            )
    print(
        f"Audited {len(report['results'])} image-condition pairs; passed={report['passed']}",
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
