from project_paths import CODE_ROOT, EXTENSION_ROOT

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch


SOURCES = ("ImageNet-1k", "ProGAN", "StyleGAN3", "GLIDE", "DALLE 3", "FireFly")
VERIFY_REL_TOL = 1e-6


def read_manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_npy(path, array):
    temporary = path.with_suffix(".npy.partial")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def grid_sizes(size, border, maximum):

    residual = tuple(side - 2 * border for side in size)
    scale = maximum / min(residual)
    original = tuple(min(maximum, round(side * scale)) for side in residual)
    capped = tuple(min(maximum, round(side * min(1.0, scale))) for side in residual)
    return original, capped, scale


def extract(args):
    import json
    from feature_cache import FeatureCache
    from fsd import FSDDetector
    from fsd.fsd_computation import compute_fsd
    from zeroshot_common import child_path, file_hash, write_csv, write_json

    started = time.monotonic()
    out = args.output_dir
    project = args.project_root
    inputs = args.input_dir
    reference = project / "Results/reproduction/section43_ossia_progan_20261110/4.3"
    reference_files = json.loads((reference / "checksums.json").read_text())
    for name in ("input_index.csv", "run.json"):
        if file_hash(reference / name) != reference_files[name]:
            raise ValueError(f"Reference checksum mismatch for {name}")
    published = {row["image_id"]: row for row in read_manifest(reference / "input_index.csv")}
    identity = json.loads((reference / "run.json").read_text())["feature_extraction_identity"]
    weights = project / "Weights/fsd_v1.2.0"
    config = json.loads((weights / "config.json").read_text())
    if config["fre"] != identity["fre_config"] or config["fsd"] != identity["fsd_config"]:
        raise ValueError("FRE or FSD configuration differs from reference")
    for name, digest in identity["extractor_code_sha256"].items():
        if file_hash(CODE_ROOT / "fsd" / name) != digest:
            raise ValueError(f"Reference extractor checksum mismatch for {name}")
    if file_hash(weights / config["fre"]["weights_file"]) != identity["fre_weights_sha256"]:
        raise ValueError("Reference FRE weights checksum mismatch")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != identity["gpu"]:
        raise ValueError("Expected the Blackwell GPU used for the reference run")
    if str(torch.__version__) != identity["torch_version"] or torch.version.cuda != identity["cuda_version"]:
        raise ValueError("Torch or CUDA runtime differs from the reference")
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    rows = read_manifest(inputs / "manifest.csv")
    if Counter(row["source"] for row in rows) != Counter({s: 100 for s in SOURCES}):
        raise ValueError("Expected the six reference sources with 100 images per source")
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image IDs")
    baseline = np.load(inputs / "baseline.npy", allow_pickle=False)
    if baseline.dtype != np.dtype("<f8") or baseline.shape != (600, 960) or not np.isfinite(baseline).all():
        raise ValueError("Invalid baseline matrix")
    fsd_config = config["fsd"]
    if fsd_config["resize_mode"] != "resize_and_crop":
        raise ValueError("Expected resize_and_crop as the released baseline")
    dimensions = fsd_config["max_size"]
    border = config["fre"]["kernel_size"] // 2
    min_short_side = (fsd_config["kernel_size"] // 2 + 1) * 2 ** (fsd_config["num_scales"] - 1)
    data_root = args.data_root or inputs
    paths, sizes = [], []
    for row, original in zip(rows, baseline):
        row["sha256"] = row.get("image_sha256") or row["sha256"]
        row["label"] = int(row["source"] != "ImageNet-1k")
        expected = published[row["image_id"]]
        feature_hash = hashlib.sha256(original.tobytes()).hexdigest()
        if (row["source"], row["sha256"], row["feature_sha256"], feature_hash) != (
            expected["source"], expected["image_sha256"], expected["feature_sha256"], expected["feature_sha256"]
        ):
            raise ValueError(f"Manifest or baseline disagrees with published vector: {row['image_id']}")
        path = child_path(data_root, row["relative_path"])
        if file_hash(path) != row["sha256"]:
            raise ValueError(f"Image checksum mismatch: {row['image_id']}")
        with Image.open(path) as image:
            size = image.size
        if min(size) - 2 * border < min_short_side:
            raise ValueError(f"Native residual too small for released reflect padding: {row['image_id']} {size}")
        paths.append(path)
        sizes.append(size)

    if len({r["sha256"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image bytes")
    version = dict(identity, experiment="residual no-upsample",
                   implementation_sha256=file_hash(Path(__file__)),
                   resize_rule="For residual short side < max_size use official crop mode, otherwise official resize_and_crop")
    verify_ids = set()
    for source in SOURCES:
        source_ids = sorted(r["image_id"] for r in rows if r["source"] == source)
        verify_ids.update(source_ids if args.fresh_baseline else source_ids[:2])

    detector = FSDDetector.load(weights_dir=weights, device="cuda:0", attribution=False)
    cap_features, active_baseline, grid_log, verification = [], [], [], []
    with (FeatureCache(out / "checkpoint" / "capped", version) as cap_cache,
          FeatureCache(out / "checkpoint" / "official", identity) as original_cache):
        for index, (row, original, path, size) in enumerate(zip(rows, baseline, paths, sizes)):
            orig_grid, cap_grid, original_scale = grid_sizes(size, border, dimensions)
            check_original = row["image_id"] in verify_ids
            with Image.open(path) as image, torch.inference_mode():
                reference_result = original
                if check_original:
                    fresh = original_cache.get(row)
                    if fresh is None:
                        fresh = detector.compute_fsd(image).numpy()
                        original_cache.put(row, fresh)
                    rel = float(np.linalg.norm(fresh - original) / np.linalg.norm(original))
                    if rel > VERIFY_REL_TOL:
                        raise ValueError(f"Fresh baseline relative difference {rel:.4g} for {row['image_id']}")
                    verification.append(dict(image_id=row["image_id"], source=row["source"], relative_l2=rel))
                    if args.fresh_baseline:
                        reference_result = fresh
                capped = cap_cache.get(row)
                if capped is None:


                    if original_scale > 1:
                        capped = compute_fsd(image, detector.fre, **dict(fsd_config, resize_mode="crop")).numpy()
                    elif check_original:
                        capped = fresh
                    else:
                        capped = detector.compute_fsd(image).numpy()
                    cap_cache.put(row, capped)
            cap_features.append(capped)
            active_baseline.append(reference_result)
            grid_log.append(dict(image_id=row["image_id"], source=row["source"], width=size[0], height=size[1],
                                 original_residual_scale=original_scale, capped_residual_scale=min(1.0, original_scale),
                                 original_grid_width=orig_grid[0], original_grid_height=orig_grid[1],
                                 capped_grid_width=cap_grid[0], capped_grid_height=cap_grid[1]))
            completed = index + 1
            if completed == 1 or completed % 10 == 0:
                progress = dict(phase="extracting", completed=completed, total=len(rows),
                                elapsed_seconds=round(time.monotonic() - started, 2),
                                utc=datetime.now(timezone.utc).isoformat())
                write_json(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        cap_cache.integrity_check()
        original_cache.integrity_check()
        cap_signature = cap_cache.signature

    atomic_npy(out / "baseline.npy", np.stack(active_baseline))
    atomic_npy(out / "no_upsample.npy", np.stack(cap_features))
    write_csv(out / "grids.csv", tuple(grid_log[0]), grid_log)
    write_csv(out / "baseline_verification.csv", tuple(verification[0]), verification)
    write_csv(out / "manifest.csv", tuple(rows[0]), rows)
    result = dict(phase="complete", images=len(rows), source_counts=dict(Counter(r["source"] for r in rows)),
                  capped_signature=cap_signature, variant=version, released=identity,
                  affected_by_cap=sum(item["original_residual_scale"] > 1 for item in grid_log),
                  input_manifest_sha256=file_hash(inputs / "manifest.csv"),
                  source_index_sha256=reference_files["input_index.csv"], verified_image_hashes=len(rows),
                  verified_baseline_hashes=len(rows), baseline_gpu_checks=len(verification),
                  baseline_matrix_source="fresh Blackwell" if args.fresh_baseline else "verified reference",
                  baseline_gpu_max_relative_l2=max(item["relative_l2"] for item in verification),
                  seconds=round(time.monotonic() - started, 2))
    result["files"] = {name: file_hash(out / name) for name in
                       ("baseline.npy", "no_upsample.npy", "manifest.csv", "grids.csv", "baseline_verification.csv")}
    write_json(out / "extraction.json", result)
    write_json(out / "progress.json", dict(phase="complete", completed=len(rows), total=len(rows)))


def main():
    parser = argparse.ArgumentParser(description='compute')
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--fresh-baseline", action="store_true")
    args = parser.parse_args()
    code = CODE_ROOT
    sys.path[:0] = [str(code), str(EXTENSION_ROOT / "zeroshot_validation")]
    from zeroshot_common import write_json
    import fcntl
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "extraction.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(args.output_dir / "progress.json", dict(phase="verifying"))
            extract(args)
        except Exception as exc:
            write_json(args.output_dir / "progress.json", dict(phase="failed", error=repr(exc)))
            raise


if __name__ == "__main__":
    main()
