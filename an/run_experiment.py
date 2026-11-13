"""Run a paired, resumable processing-robustness study with released FSD weights.

All outputs stay beneath an/ unless an explicit output directory is supplied.
Transforms affect both labels identically; no model fitting happens here.
"""

from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
import traceback

AN = Path(__file__).resolve().parent
REPO = AN.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("MPLCONFIGDIR", str(AN / ".cache" / "matplotlib"))

import numpy as np
from PIL import Image
import psutil
import torch

from fsd import FSDDetector
from fsd.projection import apply_projections
from an.robustness.extractor import extract_fsd
from an.robustness.transforms import CONDITIONS, CONDITION_MAP, apply_condition


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def load_records(path):
    """Read completed scores, preserving and truncating only an interrupted final line."""
    if not path.exists():
        return []
    rows = []
    contents = path.read_bytes()
    offset = 0
    lines = contents.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if i != len(lines) - 1:
                raise ValueError(f"Invalid record at line {i + 1} in {path}")
            # A process interruption can leave only the final JSON line partial.
            path.with_suffix(".interrupted_tail.txt").write_bytes(line)
            with path.open("r+b") as f:
                f.truncate(offset)
            break
        offset += len(line)
    return rows


def load_manifest(path):
    """Validate identifiers, splits, image decodability, hashes, and byte uniqueness."""
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    seen_ids, seen_hash = set(), {}
    for row in rows:
        row["label"] = int(row["label"])
        if row["label"] not in (0, 1) or row["split"] not in {"calibration", "test"}:
            raise ValueError(f"Invalid manifest label/split: {row}")
        if row["image_id"] in seen_ids:
            raise ValueError(f"Duplicate image ID {row['image_id']}")
        seen_ids.add(row["image_id"])
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", row["image_id"]):
            raise ValueError(f"Image ID must be safe as a filename: {row['image_id']}")
        p = Path(row["path"])
        if not p.is_absolute():
            candidates = [REPO / p, AN / p, path.parent / p]
            p = next((item for item in candidates if item.is_file()), candidates[0])
        if not p.is_file():
            raise FileNotFoundError(p)
        row["resolved_path"] = str(p.resolve())
        with Image.open(p) as image:
            image.verify()
        digest = sha256(p)
        if row.get("sha256") and row["sha256"] != digest:
            raise ValueError(f"Image content changed: {p}")
        if digest in seen_hash:
            raise ValueError(f"Duplicate bytes: {p} and {seen_hash[digest]}")
        seen_hash[digest] = str(p)
        row["sha256"] = digest
    return rows


def select_rows(rows, per_source, calibration_count, seed):
    """Select a stable cohort and interleave test sources for balanced partial runs."""

    # Hash ordering avoids reliance on filesystem order or grouping by filename.
    def order(row):
        return hashlib.sha256(f"{seed}:{row['image_id']}".encode()).hexdigest()

    calibration = sorted(
        [r for r in rows if r["split"] == "calibration" and r["label"] == 0], key=order
    )
    if calibration_count:
        calibration = calibration[:calibration_count]
    groups = defaultdict(list)
    for row in rows:
        if row["split"] == "test":
            groups[row["source"]].append(row)
    for key in groups:
        groups[key].sort(key=order)
        if per_source:
            groups[key] = groups[key][:per_source]
    # Finish a complete 19-condition image before starting the next one.
    # Round robin sources keeps a time-limited prefix approximately balanced.
    test = []
    for index in range(max(map(len, groups.values()), default=0)):
        for source in sorted(groups):
            if index < len(groups[source]):
                test.append(groups[source][index])
    return calibration + test


def feature_drift(value, reference):
    """Return relative L2 error and cosine distance from the paired clean descriptor."""
    denom = max(float(np.linalg.norm(reference)), 1e-30)
    relative = float(np.linalg.norm(value - reference) / denom)
    cos_denom = float(np.linalg.norm(value)) * denom
    cosine = float(1 - np.dot(value, reference) / max(cos_denom, 1e-30))
    return relative, float(np.clip(cosine, 0, 2))


@torch.no_grad()
def score_features(detector, fsd):
    """Apply released detection and attribution scoring to one descriptor.

    Larger anomaly scores indicate synthetic images. Source probabilities use
    the released temperature of 750; the resulting argmax has no unknown class.
    """
    raw = fsd.to(detector.fre.device).unsqueeze(0)
    projected = apply_projections(raw, detector.projections)
    likelihood = float(detector.gmm.score_samples(projected).item())
    z = (likelihood - detector.train_mean) / detector.train_std
    attr = apply_projections(projected, detector.attribution_projections)
    likelihoods = (
        {name: float(gmm.score_samples(attr).item()) for name, gmm in detector.source_gmms.items()}
        if detector.source_gmms
        else {}
    )
    pred, confidence = None, None
    if likelihoods:
        names = list(likelihoods)
        values = torch.tensor(list(likelihoods.values()), dtype=torch.float64)
        pred = names[int(values.argmax())]
        confidence = float(torch.softmax((values - values.max()) / 750.0, 0).max())
    return (
        dict(
            raw_score=likelihood,
            z_score=z,
            anomaly_score=-z,
            attribution_pred=pred,
            attribution_confidence=confidence,
            attribution_loglik=likelihoods,
        ),
        projected[0].cpu().numpy(),
        attr[0].cpu().numpy(),
    )


def environment(detector):
    import PIL

    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        revision = "unknown"
    result = dict(
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        torch=torch.__version__,
        numpy=np.__version__,
        pillow=PIL.__version__,
        cuda_runtime=torch.version.cuda,
        cpu_count=os.cpu_count(),
        memory_bytes=psutil.virtual_memory().total,
        repository_revision=revision,
        device=str(detector.fre.device),
        model_config=detector.config,
        fre_shape=list(detector.fre.conv.w.shape),
        dtype="float64",
        tf32=False,
        source_gmm_names=list(detector.source_gmms or {}),
    )
    if detector.fre.device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        result.update(
            gpu=props.name,
            gpu_total_memory=props.total_memory,
            gpu_capability=list(torch.cuda.get_device_capability(0)),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=AN / "data" / "manifest.csv")
    parser.add_argument("--output", type=Path, default=AN / "results" / "main")
    parser.add_argument("--weights", type=Path, default=AN / "weights" / "v1.2.0")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--method", choices=["tiled", "fft"], default="fft")
    parser.add_argument("--conditions", nargs="+", default=[c.name for c in CONDITIONS])
    parser.add_argument("--limit-per-source", type=int, default=0)
    parser.add_argument("--calibration-count", type=int, default=60)
    parser.add_argument("--calibration-all-conditions", action="store_true")
    parser.add_argument("--hours", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--preview-per-source", type=int, default=2)
    parser.add_argument("--no-attribution", action="store_true")
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Override ONLY for explicitly labeled pilot/debug run; main uses release1024",
    )
    args = parser.parse_args()
    if "clean" not in args.conditions:
        raise ValueError("All studies require a paired clean baseline")
    if args.hours <= 0:
        raise ValueError("--hours must be positive")
    conditions = [
        CONDITION_MAP[name] for name in ["clean"] + [x for x in args.conditions if x != "clean"]
    ]
    if len(set(c.name for c in conditions)) != len(conditions):
        raise ValueError("Duplicate conditions")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; refusing a silent CPU fallback")
    rows = select_rows(
        load_manifest(args.manifest), args.limit_per_source, args.calibration_count, args.seed
    )
    if not rows:
        raise ValueError("No images selected")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for folder in ("features", "previews", "examples"):
        (out / folder).mkdir(exist_ok=True)
    detector = FSDDetector.load(
        args.weights, device=args.device, attribution=not args.no_attribution
    )
    config = dict(detector.config["fsd"])
    if args.max_size:
        config["max_size"] = args.max_size
    plan = {
        r["image_id"]: (
            [c.name for c in conditions]
            if r["split"] == "test" or args.calibration_all_conditions
            else ["clean"]
        )
        for r in rows
    }
    code_paths = [
        Path(__file__),
        AN / "robustness" / "extractor.py",
        AN / "robustness" / "transforms.py",
    ]
    signature = dict(
        manifest_sha256=sha256(args.manifest),
        conditions=[c.to_dict() for c in conditions],
        fsd_config=config,
        weights={p.name: sha256(p) for p in args.weights.glob("*") if p.is_file()},
        method=args.method,
        seed=args.seed,
        selected_image_ids=[r["image_id"] for r in rows],
        calibration_all_conditions=args.calibration_all_conditions,
        attribution=not args.no_attribution,
        code_sha256={p.name: sha256(p) for p in code_paths},
    )
    meta_path = out / "run_metadata.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old["signature"] != signature:
            raise ValueError(
                "Output directory belongs to a different experiment; choose a new --output"
            )
        metadata = old
    else:
        metadata = dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            signature=signature,
            expected_conditions=[c.name for c in conditions],
            expected_image_conditions=plan,
            selected_manifest=rows,
            environment=environment(detector),
            sessions=[],
        )
    metadata["sessions"].append(
        dict(
            started_utc=datetime.now(timezone.utc).isoformat(),
            args=vars(args)
            | {
                "manifest": str(args.manifest),
                "output": str(args.output),
                "weights": str(args.weights),
            },
            code_sha256={p.name: sha256(p) for p in code_paths},
        )
    )
    atomic_json(meta_path, metadata)
    records_path = out / "records.jsonl"
    records = load_records(records_path)
    done = {(r["image_id"], r["condition"]) for r in records}
    expected_count = sum(map(len, plan.values()))
    started, deadline = time.monotonic(), time.monotonic() + args.hours * 3600
    preview_counts = defaultdict(int)
    errors = 0
    stopped = False
    print(
        f"Run {out}; {len(rows)} original images; {expected_count} image-condition pairs; {len(done)} cached",
        flush=True,
    )
    with records_path.open("a", encoding="utf-8", buffering=1) as stream:
        for row in rows:
            if time.monotonic() >= deadline or (out / "STOP").exists():
                stopped = True
                break
            wanted = [CONDITION_MAP[name] for name in plan[row["image_id"]]]
            keep_preview = (
                preview_counts[row["source"]] < args.preview_per_source and row["split"] == "test"
            )
            if row["split"] == "test":
                preview_counts[row["source"]] += 1
            if all((row["image_id"], c.name) in done for c in wanted):
                continue
            with Image.open(row["resolved_path"]) as opened:
                original = opened.convert("RGB")
            clean_features = None
            clean_rms = None
            for condition in wanted:
                key = (row["image_id"], condition.name)
                feature_path = out / "features" / f"{row['image_id']}__{condition.name}.npz"
                if key in done:
                    if condition.name == "clean":
                        with np.load(feature_path) as data:
                            clean_features = {
                                name: data[name].copy()
                                for name in ["raw", "projected", "attribution"]
                            }
                        clean_rms = next(
                            r["residual_rms"]
                            for r in records
                            if (r["image_id"], r["condition"]) == key
                        )
                    continue
                try:
                    tick = time.perf_counter()
                    image = apply_condition(original, condition)
                    if args.device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    extraction = extract_fsd(
                        image,
                        detector.fre,
                        **config,
                        method=args.method,
                        return_preview=keep_preview,
                    )
                    scoring, projected, attr = score_features(detector, extraction.fsd)
                    raw = extraction.fsd.numpy()
                    if not all(
                        np.isfinite(x).all() for x in (raw, projected, attr)
                    ) or not np.isfinite(scoring["z_score"]):
                        raise FloatingPointError("Nonfinite feature or score")
                    current = dict(raw=raw, projected=projected, attribution=attr)
                    np.savez_compressed(feature_path, **current)
                    if condition.name == "clean":
                        clean_features = current
                        clean_rms = extraction.diagnostics["residual_rms"]
                    if clean_features is None:
                        raise RuntimeError("Missing clean reference")
                    raw_l2, raw_cos = feature_drift(raw, clean_features["raw"])
                    proj_l2, proj_cos = feature_drift(projected, clean_features["projected"])
                    record = {key: value for key, value in row.items() if key != "resolved_path"}
                    record.update(
                        condition=condition.name,
                        family=condition.family,
                        severity=condition.severity,
                        **scoring,
                        fsd_l2_relative=raw_l2,
                        fsd_cosine_distance=raw_cos,
                        projected_l2_relative=proj_l2,
                        projected_cosine_distance=proj_cos,
                        residual_rms=extraction.diagnostics["residual_rms"],
                        residual_rms_ratio=extraction.diagnostics["residual_rms"]
                        / max(clean_rms, 1e-30),
                        seconds=time.perf_counter() - tick,
                        peak_cuda_mb=(
                            torch.cuda.max_memory_allocated() / 1024**2
                            if args.device == "cuda"
                            else 0
                        ),
                        feature_path=str(feature_path.relative_to(out)),
                        diagnostics=extraction.diagnostics,
                        processed_size=list(image.size),
                    )
                    if keep_preview:
                        preview_data = dict(residual=extraction.residual_preview)
                        if getattr(extraction, "residual_patch", None) is not None:
                            preview_data["residual_patch"] = extraction.residual_patch
                        np.savez_compressed(
                            out / "previews" / f"{row['image_id']}__{condition.name}.npz",
                            **preview_data,
                        )
                        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                        image.save(out / "examples" / f"{row['image_id']}__{condition.name}.png")
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    records.append(record)
                    done.add(key)
                    print(
                        f"[{len(done)}/{expected_count}] {row['source']} {row['image_id']} {condition.name} z={record['z_score']:.4f} {record['seconds']:.2f}s VRAM={record['peak_cuda_mb']:.0f}MiB",
                        flush=True,
                    )
                except Exception as exc:
                    errors += 1
                    error = dict(
                        image_id=row["image_id"],
                        condition=condition.name,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                    with (out / "errors.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(error) + "\n")
                    print(f"ERROR {row['image_id']} {condition.name}: {exc}", flush=True)
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
                    if condition.name == "clean":
                        break
            atomic_json(
                out / "progress.json",
                dict(
                    completed=len(done),
                    expected=expected_count,
                    errors=errors,
                    elapsed_seconds=time.monotonic() - started,
                    last_image_id=row["image_id"],
                    updated_utc=datetime.now(timezone.utc).isoformat(),
                    state="running",
                ),
            )
    state = (
        "complete"
        if len(done) == expected_count
        else ("time_limit" if stopped else "incomplete_errors")
    )
    atomic_json(
        out / "progress.json",
        dict(
            completed=len(done),
            expected=expected_count,
            errors=errors,
            elapsed_seconds=time.monotonic() - started,
            updated_utc=datetime.now(timezone.utc).isoformat(),
            state=state,
        ),
    )
    print(f"{state}: {len(done)}/{expected_count} image-condition pairs", flush=True)
    if state == "incomplete_errors":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
