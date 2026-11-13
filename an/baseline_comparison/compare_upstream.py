"""Paired comparison of the unchanged released FSD function and the AN FFT path.

Default: the frozen 25 originals, seven conditions, and official 1024 grid.
No upstream functions are monkey-patched. Both methods receive the same RGB
pixels and use the same loaded FRE, projections, GMMs, and classify function.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "an"
sys.path.insert(0, str(ROOT))

import numpy as np
import PIL
from PIL import Image
import psutil
import torch

from fsd.detector import FSDDetector
from fsd.attribution import classify
from fsd.projection import apply_projections
from an.robustness.extractor import extract_fsd
from an.robustness.transforms import CONDITION_MAP
from an.robustness.transforms import apply_condition
from an.run_experiment import load_manifest, load_records

DEFAULT_CONDITIONS = [
    "clean",
    "jpeg_q50",
    "resize_050",
    "crop_area025",
    "blur_sigma20",
    "sharpen_200",
    "resize050_jpeg50",
]
TOLERANCES = {"feature_relative_l2": 1e-4, "z_score_absolute": 1e-3, "probabilities_absolute": 1e-4}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _executable_ast(source: bytes, filename: str) -> str:
    """Normalize formatting and docstrings without removing executable statements."""
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
    return ast.dump(tree, include_attributes=False)


def verify_main_source(
    current_path: Path,
    recorded_sha256: str,
    historical_archive: Path | None = None,
    *,
    source_root: Path = AN,
) -> dict:
    """Check exact run bytes, or explicitly prove a documentation-only refactor.

    The optional ZIP contains paths relative to ``an``. Its member must match
    the run's original SHA-256 before comparing executable syntax trees. No
    archived code is imported or executed. Numeric constants, calls, control
    flow, annotations, and imports remain part of the comparison.
    """
    current_bytes = current_path.read_bytes()
    current_sha256 = hashlib.sha256(current_bytes).hexdigest()
    evidence = {"recorded_sha256": recorded_sha256, "current_sha256": current_sha256}
    if historical_archive is None:
        if current_sha256 != recorded_sha256:
            raise ValueError(
                f"Main implementation hash changed: {current_path.name}. "
                "Use a new main run or explicitly supply --historical-source-archive "
                "to verify an archived original against a formatting/docstring cleanup."
            )
        return {**evidence, "verification": "exact_bytes"}

    member_name = current_path.resolve().relative_to(source_root.resolve()).as_posix()
    archive_bytes = historical_archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        members = [entry for entry in archive.infolist() if entry.filename == member_name]
        if len(members) != 1 or members[0].is_dir():
            raise ValueError(f"Historical source archive needs exactly one file: {member_name}")
        historical_bytes = archive.read(members[0])
    historical_sha256 = hashlib.sha256(historical_bytes).hexdigest()
    if historical_sha256 != recorded_sha256:
        raise ValueError(f"Archived source does not match the recorded run hash: {member_name}")
    if _executable_ast(current_bytes, str(current_path)) != _executable_ast(
        historical_bytes, member_name
    ):
        raise ValueError(f"Executable source differs from the recorded run: {member_name}")
    return {
        **evidence,
        "verification": "exact_bytes" if current_sha256 == recorded_sha256 else "executable_ast",
        "archive_member": member_name,
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "archived_sha256": historical_sha256,
        "executable_ast_equal": True,
        "ast_comparison_ignores": ["docstrings", "source_locations"],
    }


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def keep_awake(enable):
    """Process-thread-scoped Windows sleep request; never force display on."""
    if os.name != "nt":
        return False
    function = ctypes.windll.kernel32.SetThreadExecutionState
    function.argtypes = [ctypes.c_uint]
    function.restype = ctypes.c_uint
    flags = 0x80000000 | (0x00000001 if enable else 0)
    if function(flags) == 0:
        raise ctypes.WinError()
    return enable


def memory_snapshot():
    virtual = psutil.virtual_memory()
    result = {
        "physical_total_bytes": virtual.total,
        "physical_available_bytes": virtual.available,
        "process": dict(psutil.Process().memory_info()._asdict()),
    }
    if os.name == "nt":

        class PerformanceInformation(ctypes.Structure):
            _fields_ = (
                [("cb", ctypes.c_ulong)]
                + [
                    (name, ctypes.c_size_t)
                    for name in [
                        "CommitTotal",
                        "CommitLimit",
                        "CommitPeak",
                        "PhysicalTotal",
                        "PhysicalAvailable",
                        "SystemCache",
                        "KernelTotal",
                        "KernelPaged",
                        "KernelNonpaged",
                        "PageSize",
                    ]
                ]
                + [
                    ("HandleCount", ctypes.c_ulong),
                    ("ProcessCount", ctypes.c_ulong),
                    ("ThreadCount", ctypes.c_ulong),
                ]
            )

        info = PerformanceInformation()
        info.cb = ctypes.sizeof(info)
        try:
            function = ctypes.windll.psapi.GetPerformanceInfo
            function.argtypes = [ctypes.POINTER(PerformanceInformation), ctypes.c_ulong]
            function.restype = ctypes.c_int
            if not function(ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError()
            result.update(
                commit_total_bytes=info.CommitTotal * info.PageSize,
                commit_limit_bytes=info.CommitLimit * info.PageSize,
                commit_available_bytes=(info.CommitLimit - info.CommitTotal) * info.PageSize,
            )
        except Exception as error:
            result["commit_query_error"] = str(error)
    return result


def probabilities(loglik):
    values = torch.tensor(list(loglik.values()), dtype=torch.float64)
    output = torch.softmax((values - values.max()) / 750.0, dim=0).tolist()
    return dict(zip(loglik, output))


@torch.no_grad()
def score_vector(detector, vector, calibrated_threshold):
    raw = vector[None].to(detector.fre.device)
    projected = apply_projections(raw, detector.projections)
    likelihood = float(detector.gmm.score_samples(projected).item())
    z_score = (likelihood - detector.train_mean) / detector.train_std
    attribution = apply_projections(projected, detector.attribution_projections)
    # Call the original public classification component, including its softmax.
    source, confidence, probs = classify(attribution, detector.source_gmms)
    # Retain raw likelihoods as well. This extra diagnostic pass is identical
    # for both timed paths; it is not present in the public attribute endpoint.
    loglik = {
        name: float(gmm.score_samples(attribution).item())
        for name, gmm in detector.source_gmms.items()
    }
    values = dict(
        raw_score=likelihood,
        z_score=z_score,
        anomaly_score=-z_score,
        is_fake=bool(z_score < detector.threshold),
        calibrated_is_fake=bool(-z_score > calibrated_threshold),
        attribution_pred=source,
        attribution_confidence=confidence,
        attribution_loglik=loglik,
        attribution_probabilities=probs,
    )
    arrays = {
        "raw": vector.cpu().numpy().copy(),
        "projected": projected[0].cpu().numpy().copy(),
        "attribution": attribution[0].cpu().numpy().copy(),
    }
    return values, arrays


def vector_difference(left, right):
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return {
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(right), 1e-30)),
        "max_abs": float(np.max(np.abs(delta))),
    }


def compare_values(left, right, left_features, right_features):
    keys = set(left["attribution_loglik"])
    if keys != set(right["attribution_loglik"]):
        raise ValueError("Attribution label sets differ")
    output = {
        name: vector_difference(left_features[name], right_features[name])
        for name in ["raw", "projected", "attribution"]
    }
    output.update(
        raw_score_abs=abs(left["raw_score"] - right["raw_score"]),
        z_score_abs=abs(left["z_score"] - right["z_score"]),
        confidence_abs=abs(left["attribution_confidence"] - right["attribution_confidence"]),
        probabilities_max_abs=max(
            abs(left["attribution_probabilities"][k] - right["attribution_probabilities"][k])
            for k in keys
        ),
        loglik_max_abs=max(
            abs(left["attribution_loglik"][k] - right["attribution_loglik"][k]) for k in keys
        ),
        default_decision_match=left["is_fake"] == right["is_fake"],
        calibrated_decision_match=left["calibrated_is_fake"] == right["calibrated_is_fake"],
        source_match=left["attribution_pred"] == right["attribution_pred"],
    )
    output["numeric_pass"] = bool(
        all(
            output[name]["relative_l2"] <= TOLERANCES["feature_relative_l2"]
            for name in ["raw", "projected", "attribution"]
        )
        and output["z_score_abs"] <= TOLERANCES["z_score_absolute"]
        and output["probabilities_max_abs"] <= TOLERANCES["probabilities_absolute"]
    )
    return output


def public_attribute_comparison(probe, current):
    reference = probe["result"]
    return {
        "z_score_abs": abs(reference["z_score"] - current["z_score"]),
        "source_match": reference["source"] == current["attribution_pred"],
        "default_decision_match": reference["is_fake"] == current["is_fake"],
        "confidence_abs": abs(reference["confidence"] - current["attribution_confidence"]),
        "probabilities_max_abs": max(
            abs(reference["scores"][name] - probability)
            for name, probability in current["attribution_probabilities"].items()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=AN / "baseline_comparison/manifest.csv")
    parser.add_argument("--main-run", type=Path, default=AN / "results/main")
    parser.add_argument("--output", type=Path, default=AN / "baseline_comparison/run1024")
    parser.add_argument("--weights", type=Path, default=AN / "weights/v1.2.0")
    parser.add_argument(
        "--historical-source-archive",
        type=Path,
        help=(
            "Explicitly verify cleaned AN sources against original source bytes in this ZIP. "
            "Members use AN-relative paths; original hashes and executable AST must match. "
            "Use a new comparison output directory after a source cleanup."
        ),
    )
    parser.add_argument(
        "--conditions", nargs="+", default=DEFAULT_CONDITIONS, choices=CONDITION_MAP
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument(
        "--skip-score-api-checks",
        action="store_true",
        help="Otherwise score() is checked once for each label on a clean input.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify inputs/signature without loading CUDA or starting inference",
    )
    args = parser.parse_args()
    if (
        args.hours <= 0
        or len(set(args.conditions)) != len(args.conditions)
        or "clean" not in args.conditions
    ):
        raise ValueError("Require positive hours, unique conditions, and clean baseline")
    conditions = ["clean"] + [name for name in args.conditions if name != "clean"]
    torch.set_num_threads(args.threads)
    manifest = load_manifest(args.manifest)
    if not manifest or any(row["split"] != "test" for row in manifest):
        raise ValueError("Comparison cohort must contain test originals only")
    main_dir, out = args.main_run.resolve(), args.output.resolve()
    main_metadata = json.loads((main_dir / "run_metadata.json").read_text(encoding="utf-8-sig"))
    main_summary = json.loads((main_dir / "analysis_summary.json").read_text(encoding="utf-8-sig"))
    main_environment = main_metadata.get("environment", {})
    current_versions = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "device": "cuda",
        "dtype": "float64",
    }
    version_comparison, version_warnings = {}, []
    for name, current in current_versions.items():
        previous = main_environment.get(name)
        equivalent = previous == current
        if name == "device" and previous in {"cuda", "cuda:0"}:
            equivalent = True
        version_comparison[name] = {"main": previous, "comparison": current, "matches": equivalent}
        if previous is None:
            version_warnings.append(
                f"Main environment does not record {name}; compatibility cannot be confirmed for this field."
            )
        elif not equivalent:
            if name in {"torch", "device", "dtype"}:
                raise ValueError(f"Main/comparison {name} mismatch: {previous} versus {current}")
            version_warnings.append(
                f"{name} version differs: main={previous}, comparison={current}. Pixel reconstruction and cached-feature equivalence must be interpreted with this version difference."
            )
    calibrated_threshold = main_summary["calibrated_anomaly_threshold"]
    if calibrated_threshold is None:
        raise ValueError("Main run lacks a frozen calibrated threshold")
    main_records = load_records(main_dir / "records.jsonl")
    reference_records = {(row["image_id"], row["condition"]): row for row in main_records}
    if len(reference_records) != len(main_records):
        raise ValueError("Duplicate main-run image/condition records")
    config = json.loads((args.weights / "config.json").read_text())
    if config["fsd"] != {
        "kernel_size": 11,
        "num_scales": 3,
        "max_size": 1024,
        "resize_mode": "resize_and_crop",
    }:
        raise ValueError("This comparison requires the unchanged official 1024 configuration")
    code_paths = [
        Path(__file__),
        ROOT / "fsd/detector.py",
        ROOT / "fsd/fsd_computation.py",
        ROOT / "fsd/fre.py",
        ROOT / "fsd/projection.py",
        ROOT / "fsd/gmm.py",
        ROOT / "fsd/attribution.py",
        AN / "robustness/extractor.py",
        AN / "robustness/transforms.py",
        AN / "run_experiment.py",
    ]
    code_hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in code_paths}
    original_hashes, feature_hashes = {}, {}
    for original in manifest:
        identifier = original["image_id"]
        original_hashes[identifier] = original["sha256"]
        for condition in conditions:
            reference = reference_records.get((identifier, condition))
            if reference is None:
                raise ValueError(f"Missing main reference: {identifier}/{condition}")
            if any(
                reference[key] != original[key] for key in ["source", "label", "split", "sha256"]
            ):
                raise ValueError(f"Main/original identity mismatch for {identifier}")
            feature_hashes[f"{identifier}|{condition}"] = sha256(
                main_dir / reference["feature_path"]
            )
    signature = dict(
        manifest_sha256=sha256(args.manifest),
        main_records_sha256=sha256(main_dir / "records.jsonl"),
        main_metadata_sha256=sha256(main_dir / "run_metadata.json"),
        main_analysis_summary_sha256=sha256(main_dir / "analysis_summary.json"),
        conditions=[CONDITION_MAP[name].to_dict() for name in conditions],
        fsd_config=config["fsd"],
        code_sha256=code_hashes,
        weights={path.name: sha256(path) for path in args.weights.glob("*") if path.is_file()},
        original_sha256=original_hashes,
        reference_feature_sha256=feature_hashes,
        selected_image_ids=[row["image_id"] for row in manifest],
        device="cuda",
        dtype="float64",
        skip_score_api_checks=args.skip_score_api_checks,
        threads=args.threads,
        runtime_versions=current_versions,
        method_order_policy="fixed_planned_case_index_even_upstream_first_odd_an_first",
    )
    # Frozen phase signatures remain authoritative. An explicitly supplied
    # source archive can establish equivalence after formatting/docstring edits.
    main_signatures = [main_metadata.get("signature", {})]
    for phase in main_metadata.get("phases", []):
        if isinstance(phase, dict):
            main_signatures.append(phase.get("signature", {}))
    checked_main_protocol = []
    source_compatibility = {}
    for previous in main_signatures:
        if previous.get("fsd_config") and previous["fsd_config"] != config["fsd"]:
            raise ValueError("Main and comparison FSD configurations differ")
        if previous.get("weights") and previous["weights"] != signature["weights"]:
            raise ValueError("Main and comparison weight hashes differ")
        for name, digest in previous.get("code_sha256", {}).items():
            matches = [path for path in code_paths if path.name == Path(name).name]
            if matches:
                current_path = matches[0]
                archive = (
                    args.historical_source_archive if current_path.is_relative_to(AN) else None
                )
                evidence = verify_main_source(current_path, digest, archive)
                source_compatibility[current_path.relative_to(ROOT).as_posix()] = evidence
        if previous:
            checked_main_protocol.append(previous)
    if not checked_main_protocol:
        raise ValueError("Could not locate main protocol signature for compatibility checking")
    if args.historical_source_archive:
        signature["historical_source_compatibility"] = source_compatibility
    plan = {row["image_id"]: conditions for row in manifest}
    metadata_path = out / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["signature"] != signature:
            raise ValueError("Existing output signature differs; use a new output directory")
    else:
        metadata = dict(
            schema_version=1,
            created_utc=utc(),
            signature=signature,
            selected_manifest=manifest,
            expected_image_conditions=plan,
            expected_conditions=conditions,
            main_run_dir=str(main_dir),
            manifest_path=str(args.manifest.resolve()),
            tolerances=TOLERANCES,
            calibrated_anomaly_threshold=calibrated_threshold,
            version_comparison=version_comparison,
            warnings=version_warnings,
            source_compatibility=source_compatibility,
            released_z_threshold=config["scoring"]["default_threshold"],
            timer_notes={
                "per_method": "Synchronized FSD extraction, original detection and attribution, and an additional raw-likelihood diagnostic pass over all 14 sources. Includes CPU output copies; excludes image processing, feature writes, differences, and public API checks.",
                "an_extra": "AN extraction includes its production residual/numerical diagnostics; upstream compute_fsd does not.",
                "allocator": "Timed method order alternates by fixed planned case index: even upstream then AN; odd AN then upstream. Resume retains that order. CUDA cache is retained, so reserved memory can include preceding allocations. Allocated and reserved values measure logical PyTorch memory, not resident VRAM; WDDM system-memory fallback can permit values above 6 GiB.",
                "reference": "Saved main seconds used a different timer and are not the paired method timing.",
                "processed_pixels": "Literal upstream and fresh AN receive identical RGB PIL pixels and share the recorded pixel hash. Saved main has no processed pixel hash. Reconstruction is linked to original-image hashes and source compatibility checks, which require exact bytes or explicitly verified executable AST equivalence.",
            },
            sessions=[],
        )
    print(
        json.dumps(
            {
                "status": "inputs_verified",
                "originals": len(manifest),
                "cases": len(manifest) * len(conditions),
                "output": str(out),
                "fsd_config": config["fsd"],
                "version_comparison": version_comparison,
                "warnings": version_warnings,
                "source_compatibility": source_compatibility,
            }
        ),
        flush=True,
    )
    if args.check_only:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    out.mkdir(parents=True, exist_ok=True)
    (out / "features").mkdir(exist_ok=True)
    records_path = out / "records.jsonl"
    records = load_records(records_path)
    done = {(row["image_id"], row["condition"]) for row in records}
    expected = {
        (identifier, condition) for identifier, names in plan.items() for condition in names
    }
    planned_case_indices = {
        key: index
        for index, key in enumerate(
            (identifier, condition) for identifier, names in plan.items() for condition in names
        )
    }
    if len(done) != len(records) or not done <= expected:
        raise ValueError("Duplicate or unexpected comparison records")
    for record in records:
        for method in ["upstream", "an"]:
            feature_path = out / record[method]["feature_path"]
            if (
                not feature_path.is_file()
                or sha256(feature_path) != record[method]["feature_sha256"]
            ):
                raise ValueError(f"Completed feature cache missing/changed: {feature_path}")
    checked_labels = {int(row["label"]) for row in records if "public_score_api_check" in row}
    probes = []
    for path in (AN / "baseline_comparison").glob("upstream_default1024_*.json"):
        probe = json.loads(path.read_text())
        if probe.get("status") == "success" and probe.get("requested_max_size") == 1024:
            probes.append((path, probe))
    errors = 0
    stopped = False
    started = time.monotonic()
    deadline = started + args.hours * 3600
    awake = False
    try:
        awake = keep_awake(True)
        detector = FSDDetector.load(args.weights, device="cuda", attribution=True)
        metadata["environment"] = dict(
            python=sys.version,
            executable=sys.executable,
            platform=platform.platform(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            numpy=np.__version__,
            pillow=PIL.__version__,
            gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
            physical_memory_bytes=psutil.virtual_memory().total,
            upstream_function_module="fsd.fsd_computation.compute_fsd",
            upstream_function_file=str(ROOT / "fsd/fsd_computation.py"),
            dtype="float64",
            tf32=False,
        )
        metadata["public_attribute_probes"] = [
            {"path": str(path), "sha256": sha256(path)} for path, _ in probes
        ]
        metadata["sessions"].append(
            dict(
                started_utc=utc(),
                arguments={
                    name: str(value) if isinstance(value, Path) else value
                    for name, value in vars(args).items()
                },
                keep_awake=awake,
                memory=memory_snapshot(),
            )
        )
        atomic_json(metadata_path, metadata)
        with records_path.open("a", encoding="utf-8", buffering=1) as stream:
            for original in manifest:
                with Image.open(original["resolved_path"]) as opened:
                    pristine = opened.convert("RGB")
                for name in conditions:
                    key = (original["image_id"], name)
                    if key in done:
                        continue
                    if time.monotonic() >= deadline or (out / "STOP").exists():
                        stopped = True
                        break
                    record = {k: original[k] for k in ["image_id", "source", "label", "split"]}
                    record.update(
                        condition=name,
                        family=CONDITION_MAP[name].family,
                        severity=CONDITION_MAP[name].severity,
                        image_sha256=original["sha256"],
                        started_utc=utc(),
                        memory_before=memory_snapshot(),
                        planned_case_index=planned_case_indices[key],
                        method_order=(
                            ["upstream", "an"]
                            if planned_case_indices[key] % 2 == 0
                            else ["an", "upstream"]
                        ),
                    )
                    try:
                        image = apply_condition(pristine, CONDITION_MAP[name])
                        record["processed_size"] = list(image.size)
                        record["processed_pixel_sha256"] = hashlib.sha256(
                            f"RGB:{image.width}:{image.height}:".encode() + image.tobytes()
                        ).hexdigest()
                        reference = reference_records[key]
                        with np.load(main_dir / reference["feature_path"]) as cached:
                            reference_features = {
                                k: cached[k].copy() for k in ["raw", "projected", "attribution"]
                            }
                        reference_scores = {
                            k: reference[k]
                            for k in [
                                "raw_score",
                                "z_score",
                                "anomaly_score",
                                "attribution_pred",
                                "attribution_confidence",
                                "attribution_loglik",
                            ]
                        }
                        reference_scores.update(
                            is_fake=reference["z_score"] < detector.threshold,
                            calibrated_is_fake=reference["anomaly_score"] > calibrated_threshold,
                            attribution_probabilities=probabilities(
                                reference["attribution_loglik"]
                            ),
                            feature_path=str(main_dir / reference["feature_path"]),
                            feature_sha256=feature_hashes[f"{key[0]}|{key[1]}"],
                            seconds=reference["seconds"],
                        )
                        record["reference"] = reference_scores
                        features = {}
                        for method in record["method_order"]:
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            tick = time.perf_counter()
                            if method == "upstream":
                                vector = detector.compute_fsd(image)
                                diagnostics = None
                            else:
                                extraction = extract_fsd(
                                    image,
                                    detector.fre,
                                    **detector.config["fsd"],
                                    method="fft",
                                    return_preview=False,
                                )
                                vector, diagnostics = extraction.fsd, extraction.diagnostics
                            values, arrays = score_vector(detector, vector, calibrated_threshold)
                            torch.cuda.synchronize()
                            values.update(
                                seconds=time.perf_counter() - tick,
                                peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                                peak_cuda_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                            )
                            if (
                                not all(np.isfinite(array).all() for array in arrays.values())
                                or not np.isfinite(
                                    list(values["attribution_loglik"].values())
                                    + [values["z_score"], values["raw_score"]]
                                ).all()
                            ):
                                raise FloatingPointError(f"Nonfinite {method} result")
                            feature_path = (
                                out / "features" / f"{original['image_id']}__{name}__{method}.npz"
                            )
                            with feature_path.with_suffix(".tmp").open("wb") as file:
                                np.savez_compressed(file, **arrays)
                            feature_path.with_suffix(".tmp").replace(feature_path)
                            values.update(
                                feature_path=feature_path.relative_to(out).as_posix(),
                                feature_sha256=sha256(feature_path),
                            )
                            if diagnostics is not None:
                                values["diagnostics"] = diagnostics
                            record[method], features[method] = values, arrays
                        record["differences"] = {
                            "upstream_vs_an": compare_values(
                                record["upstream"],
                                record["an"],
                                features["upstream"],
                                features["an"],
                            ),
                            "an_vs_main": compare_values(
                                record["an"], reference_scores, features["an"], reference_features
                            ),
                            "upstream_vs_main": compare_values(
                                record["upstream"],
                                reference_scores,
                                features["upstream"],
                                reference_features,
                            ),
                        }
                        if (
                            name == "clean"
                            and int(original["label"]) not in checked_labels
                            and not args.skip_score_api_checks
                        ):
                            torch.cuda.synchronize()
                            tick = time.perf_counter()
                            public = detector.score(image)
                            torch.cuda.synchronize()
                            record["public_score_api_check"] = dict(
                                seconds=time.perf_counter() - tick,
                                raw_score=public.raw_score,
                                z_score=public.z_score,
                                is_fake=public.is_fake,
                                threshold=public.threshold,
                                raw_score_abs=abs(
                                    public.raw_score - record["upstream"]["raw_score"]
                                ),
                                z_score_abs=abs(public.z_score - record["upstream"]["z_score"]),
                                default_decision_match=public.is_fake
                                == record["upstream"]["is_fake"],
                            )
                            checked_labels.add(int(original["label"]))
                        record["public_attribute_api_checks"] = []
                        for path, probe in probes:
                            if (
                                probe["image_sha256"] == original["sha256"]
                                and probe["condition"] == name
                            ):
                                record["public_attribute_api_checks"].append(
                                    dict(
                                        probe_path=str(path),
                                        **public_attribute_comparison(probe, record["upstream"]),
                                    )
                                )
                        record.update(memory_after=memory_snapshot(), finished_utc=utc())
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        stream.flush()
                        records.append(record)
                        done.add(key)
                        delta = record["differences"]["upstream_vs_an"]
                        print(
                            f"[{len(done)}/{len(expected)}] {key[0]} {name} upstream={record['upstream']['seconds']:.2f}s AN={record['an']['seconds']:.2f}s delta_z={delta['z_score_abs']:.3g} raw_rel={delta['raw']['relative_l2']:.3g} pass={delta['numeric_pass']}",
                            flush=True,
                        )
                    except Exception as error:
                        errors += 1
                        with (out / "errors.jsonl").open("a", encoding="utf-8") as log:
                            log.write(
                                json.dumps(
                                    dict(
                                        image_id=key[0],
                                        condition=name,
                                        utc=utc(),
                                        error=str(error),
                                        traceback=traceback.format_exc(),
                                    )
                                )
                                + "\n"
                            )
                        print(f"ERROR {key}: {error}", flush=True)
                        if isinstance(error, torch.OutOfMemoryError):
                            torch.cuda.empty_cache()
                    numeric_failures = sum(
                        not r["differences"]["upstream_vs_an"]["numeric_pass"]
                        or not r["differences"]["an_vs_main"]["numeric_pass"]
                        for r in records
                    )
                    atomic_json(
                        out / "progress.json",
                        dict(
                            state="running",
                            completed=len(done),
                            expected=len(expected),
                            errors=errors,
                            numeric_failures=numeric_failures,
                            elapsed_seconds=time.monotonic() - started,
                            updated_utc=utc(),
                            last_case=list(key),
                        ),
                    )
                if stopped:
                    break
        state = (
            "complete"
            if len(done) == len(expected)
            else ("time_limit_or_stop" if stopped else "incomplete_errors")
        )
        atomic_json(
            out / "progress.json",
            dict(
                state=state,
                completed=len(done),
                expected=len(expected),
                errors=errors,
                numeric_failures=sum(
                    not r["differences"]["upstream_vs_an"]["numeric_pass"]
                    or not r["differences"]["an_vs_main"]["numeric_pass"]
                    for r in records
                ),
                elapsed_seconds=time.monotonic() - started,
                updated_utc=utc(),
            ),
        )
        print(f"{state}: {len(done)}/{len(expected)} paired cases", flush=True)
        return 2 if state == "incomplete_errors" else 0
    finally:
        if awake:
            keep_awake(False)


if __name__ == "__main__":
    raise SystemExit(main())
