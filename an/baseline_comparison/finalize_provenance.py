"""Freeze final comparison provenance using only local CPU/file operations.

Run after scoring, analysis, and report edits finish. The original AN source
snapshot is preserved. Large feature NPZ files are not enumerated or rehashed;
their verification is already documented by the comparison analyzer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
MAIN_RECORDS_SHA256 = "ad2b760a9319b5cdb3d01845150d402c320b3346257057ee6e77b2a6c8777735"
OFFICIAL_MAIN = "50f2eae06efdac2e5a33f407ca9a27a2295133ac"
EXCLUDED_PARTS = {"features", "__pycache__", ".venv", ".cache"}
ARTIFACT_SUFFIXES = {".py", ".md", ".csv", ".json", ".jsonl", ".png", ".patch", ".txt", ".log"}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_record(path):
    path = Path(path).resolve()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def require_complete(pipeline, summary, progress):
    if pipeline.get("stage") != "complete":
        raise ValueError(f"Pipeline is not complete: {pipeline.get('stage')}")
    if summary.get("status") != "complete":
        raise ValueError(f"Comparison analysis is not complete: {summary.get('status')}")
    if progress.get("state") != "complete":
        raise ValueError(f"Scoring is not complete: {progress.get('state')}")
    if (
        summary.get("expected_cases") != 175
        or summary.get("completed_cases") != 175
        or summary.get("missing_cases")
    ):
        raise ValueError("Expected a complete 25-original, seven-condition comparison (175 cases)")
    if progress.get("expected") != 175 or progress.get("completed") != 175:
        raise ValueError("Scoring progress does not match complete analysis coverage")


def current_official_source(evidence):
    head = git("rev-parse", "HEAD").decode().strip()
    if head != OFFICIAL_MAIN:
        raise ValueError(f"Current checkout no longer matches observed official main: {head}")
    if evidence["official_ref_observation"]["reported_refs"]["refs/heads/main"] != OFFICIAL_MAIN:
        raise ValueError("Source-verification evidence names another official main commit")
    changes = git("status", "--porcelain", "--", "fsd").decode().strip()
    if changes:
        raise ValueError(f"Official fsd working files have changed: {changes}")
    files = []
    for item in evidence["files"]:
        path = item["path"]
        blob = git("rev-parse", f"HEAD:{path}").decode().strip()
        filtered = git("hash-object", f"--path={path}", path).decode().strip()
        canonical = git("cat-file", "blob", blob)
        canonical_sha256 = hashlib.sha256(canonical).hexdigest()
        if not blob == filtered == item["official_main_git_blob_sha1"]:
            raise ValueError(f"Current Git identity differs from verified official main: {path}")
        if canonical_sha256 != item["canonical_main_sha256"]:
            raise ValueError(f"Canonical Git bytes differ from source evidence: {path}")
        for field, expected_field in [
            ("canonical_main_snapshot", "canonical_main_sha256"),
            ("canonical_tag_snapshot", "canonical_tag_sha256"),
        ]:
            if sha256(BASE / item[field]) != item[expected_field]:
                raise ValueError(f"Preserved canonical source snapshot changed: {item[field]}")
        files.append(
            {
                **file_record(ROOT / path),
                "current_head_git_blob_sha1": blob,
                "working_tree_filtered_git_blob_sha1": filtered,
                "canonical_git_blob_sha256": canonical_sha256,
                "matches_observed_official_main": True,
            }
        )
    if len(files) != 11:
        raise ValueError(f"Expected 11 official fsd source files, found {len(files)}")
    return {
        "head": head,
        "origin": git("remote", "get-url", "origin").decode().strip(),
        "official_upstream": evidence["upstream_repository"],
        "official_ref_observation": evidence["official_ref_observation"],
        "network_requeried": False,
        "main_is_exact_v120_tag": evidence["main_is_exact_tag_commit"],
        "tag_commit": evidence["tag_commit"],
        "fsd_working_tree_changes": changes,
        "newline_note": "Git-filtered working blobs match official main. Raw Windows working bytes can differ from canonical Git bytes due to newline conversion.",
        "files": files,
    }


def artifact_inventory(output):
    # Filter at discovery time so the large feature directory is not traversed.
    args = [
        "rg",
        "--files",
        "an/baseline_comparison",
        "-g",
        "!**/features/**",
        "-g",
        "!**/*.npz",
        "-g",
        "!**/__pycache__/**",
        "-g",
        "!**/*.pyc",
        "-g",
        "!**/*.tmp",
    ]
    discovered = subprocess.check_output(args, cwd=ROOT, text=True, encoding="utf-8").splitlines()
    paths = []
    for relative in discovered:
        path = (ROOT / relative).resolve()
        if path == output or path.suffix not in ARTIFACT_SUFFIXES:
            continue
        if set(path.relative_to(BASE).parts[:-1]) & EXCLUDED_PARTS:
            continue
        paths.append(path)
    return [file_record(path) for path in sorted(set(paths))]


def build_snapshot(output):
    pipeline = read_json(BASE / "pipeline_status.json")
    run = BASE / "run1024"
    summary = read_json(run / "comparison_summary.json")
    progress = read_json(run / "progress.json")
    require_complete(pipeline, summary, progress)
    preserved = ROOT / "an/source_provenance.json"
    preserved_before = file_record(preserved)
    main_records = ROOT / "an/results/main/records.jsonl"
    actual_main_digest = sha256(main_records)
    if actual_main_digest != MAIN_RECORDS_SHA256:
        raise ValueError("Original main prediction records changed")
    main_metadata = read_json(ROOT / "an/results/main/run_metadata.json")
    if main_metadata.get("merged_records_sha256") != MAIN_RECORDS_SHA256:
        raise ValueError("Main merge metadata no longer identifies the original record digest")
    run_metadata = read_json(run / "run_metadata.json")
    if run_metadata["signature"]["main_records_sha256"] != MAIN_RECORDS_SHA256:
        raise ValueError("Comparison used a different main-record reference")
    selection = read_json(BASE / "selection.json")
    manifest_digest = sha256(BASE / "manifest.csv")
    if (
        not manifest_digest
        == selection["manifest"]["sha256"]
        == run_metadata["signature"]["manifest_sha256"]
    ):
        raise ValueError("Frozen selection, manifest, and executed comparison disagree")
    source = current_official_source(read_json(BASE / "source_verification.json"))
    artifacts = artifact_inventory(output)
    if file_record(preserved) != preserved_before:
        raise ValueError("Original AN source snapshot changed during provenance collection")
    packages = {
        dist.metadata["Name"]: dist.version
        for dist in package_metadata.distributions()
        if dist.metadata.get("Name")
    }
    external_references = [
        preserved,
        ROOT / "an/weights_provenance.json",
        ROOT / "an/requirements-lock.txt",
        ROOT / "an/data/manifest_real.csv",
        ROOT / "an/data/manifest_fake.csv",
        ROOT / "an/results/main/run_metadata.json",
        ROOT / "an/results/main/verification.json",
        ROOT / "an/README.md",
        ROOT / "an/RESULTS.md",
        ROOT / "an/docs/implementation_notes.md",
        ROOT / "an/tests/test_baseline_comparison.py",
    ]
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Final local comparison artifact/source inventory after complete scoring, analysis and report edits",
        "pipeline": pipeline,
        "completion": {
            "status": summary["status"],
            "completed_cases": summary["completed_cases"],
            "expected_cases": summary["expected_cases"],
            "missing_cases": summary["missing_cases"],
            "numeric_verdicts": summary["independently_recomputed_numeric_pass"],
            "agreement": summary["agreement"],
            "errors_recorded": summary["errors_recorded"],
        },
        "original_main_predictions": {
            "path": main_records.relative_to(ROOT).as_posix(),
            "expected_sha256": MAIN_RECORDS_SHA256,
            "observed_sha256": actual_main_digest,
            "unchanged": True,
        },
        "preserved_original_an_source_snapshot": {**preserved_before, "modified": False},
        "current_official_fsd_source": source,
        "comparison_manifest_sha256": manifest_digest,
        "artifacts": artifacts,
        "external_provenance_references": [file_record(path) for path in external_references],
        "feature_cache_verification": {
            "rehashed_by_this_script": False,
            "reason": "The CPU analyzer already verifies original, reference, and paired feature cache hashes and independently recomputes numerical differences. This final inventory avoids repeating NPZ hashing.",
            "analyzer_summary": "an/baseline_comparison/run1024/comparison_summary.json",
            "analyzer_hash_checks_recorded": len(summary["hash_checks"]),
            "analyzer_processed_pixel_hash_checks": summary[
                "processed_pixel_hashes_reconstructed_and_verified"
            ],
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
            "gpu_libraries_imported_by_finalizer": False,
            "network_used": False,
        },
        "inventory_policy": "rg --files respecting ignores; baseline scripts/docs/results/plots/manifests/source snapshots/API probes/power evidence/logs included; NPZ, feature directories, bytecode, temporary files, and this output excluded",
        "snapshot_limit": "Hashes describe files at finalization time. Later edits or appended logs require a new snapshot; the earlier AN source_provenance.json is intentionally retained unchanged.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "final_provenance.json")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate completion and collect hashes without writing a snapshot",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly refresh an existing final comparison snapshot",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(BASE) or output == ROOT / "an/source_provenance.json":
        parser.error("Output must stay under an/baseline_comparison")
    if output.exists() and not args.overwrite and not args.check_only:
        parser.error("Final snapshot already exists; use another output or explicitly --overwrite")
    try:
        snapshot = build_snapshot(output)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"finalize_provenance: {error}\n")
    if not args.check_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(snapshot, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
    print(
        json.dumps(
            {
                "status": "checked" if args.check_only else "written",
                "output": output.relative_to(ROOT).as_posix(),
                "artifact_count": len(snapshot["artifacts"]),
                "official_fsd_files": len(snapshot["current_official_fsd_source"]["files"]),
                "main_records_unchanged": True,
                "original_an_source_snapshot_preserved": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
