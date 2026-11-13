from project_paths import CODE_ROOT, EXTENSION_ROOT

import argparse
from collections import Counter
import csv
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch


PILOT = "section43_residual_upsampling_pilot600_20261111"
PILOT_CHECKSUMS_SHA = "478316ca942cce1bfd525e2f3f459786147fc4b9f4ad4e971b417734d2909205"


def capped_extractor(released):

    src = inspect.getsource(released)
    old = "scale_factor = max_size / min(h, w)"
    new = "scale_factor = min(2.0, max_size / min(h, w))"
    if src.count(old) != 1:
        raise ValueError("Release source does not have the expected single resize operation")
    patched = src.replace(old, new)
    namespace = dict(released.__globals__)
    exec(compile(patched, "<verified compute_fsd with a 2x cap>", "exec"), namespace)
    return namespace[released.__name__], hashlib.sha256(patched.encode("utf-8")).hexdigest()


def checked(path, digest):
    from zeroshot_common import file_hash
    if file_hash(path) != digest:
        raise ValueError(f"SHA256 mismatch: {path}")
    return path


def extract(args):
    from feature_cache import FeatureCache
    from fsd import FSDDetector
    from fsd.fsd_computation import compute_fsd
    from zeroshot_common import child_path, file_hash, write_csv, write_json

    started = time.monotonic()
    parent = args.project_root / "Results/reproduction" / PILOT
    out = args.output_dir
    inventory = json.loads(checked(parent / "checksums.json", PILOT_CHECKSUMS_SHA).read_text())
    paths = {key: checked(parent / key, inventory[key]) for key in
             ("features/baseline.npy", "features/manifest.csv", "features/extraction.json", "selection.json")}
    selection = json.loads(paths["selection.json"].read_text())
    checked(args.input_dir / "manifest.csv", selection["manifest_sha256"])
    cached = np.load(checked(args.input_dir / "baseline.npy", selection["baseline_sha256"]), allow_pickle=False)
    fresh = np.load(paths["features/baseline.npy"], allow_pickle=False)
    if any(x.dtype != np.dtype("<f8") or x.shape != (600, 960) or not np.isfinite(x).all() for x in (fresh, cached)):
        raise ValueError("Invalid parent matrix")
    with paths["features/manifest.csv"].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if (len(rows) != len(fresh) or len(set(r["image_id"] for r in rows)) != 600 or
            Counter(r["source"] for r in rows) != selection["source_counts"]):
        raise ValueError("Cohort has changed")
    parent_run = json.loads(paths["features/extraction.json"].read_text())
    if parent_run["baseline_matrix_source"] != "fresh Blackwell":
        raise ValueError("Parent matrix is not the recomputed Blackwell baseline")
    identity = parent_run["released"]
    weights = args.project_root / "Weights/fsd_v1.2.0"
    cfg = json.loads((weights / "config.json").read_text())
    if cfg["fre"] != identity["fre_config"] or cfg["fsd"] != identity["fsd_config"]:
        raise ValueError("Release configuration differs from the pilot")
    for name, expected in identity["extractor_code_sha256"].items():
        checked(CODE_ROOT / "fsd" / name, expected)
    checked(weights / cfg["fre"]["weights_file"], identity["fre_weights_sha256"])
    if (not torch.cuda.is_available() or torch.cuda.get_device_name(0) != identity["gpu"] or
            str(torch.__version__) != identity["torch_version"] or torch.version.cuda != identity["cuda_version"]):
        raise ValueError("Expected the Blackwell environment used for the pilot")
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cap_fsd, patched_hash = capped_extractor(compute_fsd)
    signature = dict(identity, experiment="residual upsampling cap of 2",
                     patched_source_sha256=patched_hash, compute_sha256=file_hash(Path(__file__)))
    if cfg["fsd"]["resize_mode"] != "resize_and_crop":
        raise ValueError("The released resize_and_crop pipeline must be used")
    maximum = cfg["fsd"]["max_size"]
    border = cfg["fre"]["kernel_size"] // 2
    detector = FSDDetector.load(weights_dir=weights, device="cuda:0", attribution=False)
    cap_features = np.empty_like(fresh)
    grid_rows = []
    processed = 0
    adjusted = 0
    with FeatureCache(out / "checkpoint", signature) as cache:
        for i, row in enumerate(rows):
            if int(row["row"]) != i or hashlib.sha256(cached[i].tobytes()).hexdigest() != row["feature_sha256"]:
                raise ValueError(f"Archived vector or row mismatch for {row['image_id']}")
            path = checked(child_path(args.input_dir, row["relative_path"]), row["image_sha256"])
            with Image.open(path) as image, torch.inference_mode():
                width, height = image.size
                if (width, height) != (int(row["width"]), int(row["height"])):
                    raise ValueError(f"Image dimensions have changed for {row['image_id']}")
                w, h = width - 2 * border, height - 2 * border
                original = maximum / min(w, h)
                capped = min(2.0, original)
                if original <= 2.0:
                    cap_features[i] = fresh[i]
                else:
                    adjusted += 1
                    feature = cache.get(row)
                    if feature is None:
                        feature = cap_fsd(image, detector.fre, **cfg["fsd"]).numpy()
                        cache.put(row, feature)
                        processed += 1
                    cap_features[i] = feature
            grid_rows.append(dict(image_id=row["image_id"], source=row["source"], width=width, height=height,
                                  original_residual_scale=original, capped_residual_scale=capped,
                                  original_grid_width=min(maximum, round(w * original)),
                                  original_grid_height=min(maximum, round(h * original)),
                                  capped_grid_width=min(maximum, round(w * capped)),
                                  capped_grid_height=min(maximum, round(h * capped))))
            n = i + 1
            if n == 1 or n % 10 == 0:
                progress = dict(phase="extracting", completed=n, total=len(rows), affected=adjusted,
                                newly_extracted=processed, elapsed_seconds=round(time.monotonic() - started, 2))
                write_json(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        cache.integrity_check()
    with (out / "cap2.npy.partial").open("wb") as stream:
        np.save(stream, cap_features, allow_pickle=False)
    (out / "cap2.npy.partial").replace(out / "cap2.npy")
    shutil.copyfile(paths["features/manifest.csv"], out / "manifest.csv")
    write_csv(out / "grids.csv", tuple(grid_rows[0]), grid_rows)
    report = dict(images=len(rows), source_counts=dict(Counter(r["source"] for r in rows)), affected_by_cap=adjusted,
                  reused_baseline=len(rows) - adjusted, computed_or_resumed=adjusted, newly_extracted=processed,
                  parent_run=PILOT, parent_checksums_sha256=PILOT_CHECKSUMS_SHA,
                  parent_manifest_sha256=inventory["features/manifest.csv"], parent_baseline_sha256=inventory["features/baseline.npy"],
                  variant=signature, seconds=round(time.monotonic() - started, 2),
                  input_manifest_sha256=selection["manifest_sha256"],
                  files={n: file_hash(out / n) for n in ("cap2.npy", "manifest.csv", "grids.csv")})
    write_json(out / "extraction.json", report)
    write_json(out / "progress.json", dict(phase="complete", completed=len(rows), total=len(rows)))


def main():
    import fcntl
    parser = argparse.ArgumentParser(description='compute')
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    code = CODE_ROOT
    sys.path[:0] = [str(code), str(EXTENSION_ROOT / "zeroshot_validation")]
    from zeroshot_common import write_json
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
