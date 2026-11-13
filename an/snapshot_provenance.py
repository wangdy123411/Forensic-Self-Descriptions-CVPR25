"""Record current source hashes without replacing the completed study's provenance.

The default destination is an/provenance/source_snapshot.json. Historical
snapshots describe the bytes used for the completed run and remain unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "an"
EXCLUDED = {
    ".venv",
    ".cache",
    ".maintenance",
    "data",
    "results",
    "weights",
    "__pycache__",
    "generated",
    "fixtures",
    "provenance",
}


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True, encoding="utf-8").strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def manifest_record(filename):
    path = AN / "data" / filename
    record = file_record(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    record["original_count"] = len(rows)
    record["source_split_format_counts"] = dict(
        sorted(
            Counter(
                f"{row['source']} | {row['split']} | {row.get('format', 'unrecorded')}"
                for row in rows
            ).items()
        )
    )
    companion = path.with_suffix(".provenance.json")
    if companion.exists():
        record["provenance"] = file_record(companion)
        recorded = json.loads(companion.read_text(encoding="utf-8"))
        record["matches_frozen_provenance_hash"] = recorded["manifest_sha256"] == record["sha256"]
        if not record["matches_frozen_provenance_hash"]:
            raise ValueError(f"Frozen manifest/provenance disagree: {path}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        default="current-working-tree",
        help="Human-readable label for the current source snapshot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=AN / "provenance" / "source_snapshot.json",
        help="Destination for this snapshot; historical study provenance is preserved by default",
    )
    args = parser.parse_args()
    # rg respects repository ignores and does not walk the ignored virtualenv.
    paths = [ROOT / item for item in command("rg", "--files", "fsd", "an").splitlines()]
    python_paths = sorted(
        (
            path
            for path in paths
            if path.suffix == ".py" and not (set(path.relative_to(ROOT).parts[1:-1]) & EXCLUDED)
        ),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    official = [
        file_record(path) for path in python_paths if path.relative_to(ROOT).parts[0] == "fsd"
    ]
    added = [file_record(path) for path in python_paths if path.relative_to(ROOT).parts[0] == "an"]
    tracked_changes = command("git", "diff", "--name-status", "HEAD").splitlines()
    official_changes = command("git", "diff", "--name-status", "HEAD", "--", "fsd").splitlines()
    untracked = command(
        "git", "ls-files", "--others", "--exclude-standard", "--", "an"
    ).splitlines()
    source_hashes = {item["path"]: item["sha256"] for item in official + added}
    runtime_sources = {
        "run_experiment.py": "an/run_experiment.py",
        "extractor.py": "an/robustness/extractor.py",
        "transforms.py": "an/robustness/transforms.py",
    }
    phase_metadata = []
    for phase in ["real_phase", "fake_phase"]:
        path = AN / "results" / phase / "run_metadata.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        comparison = {}
        for name, expected in data["signature"]["code_sha256"].items():
            current = source_hashes[runtime_sources[name]]
            comparison[name] = {
                "recorded_sha256": expected,
                "current_sha256": current,
                "matches": current == expected,
            }
        phase_metadata.append(
            {
                **file_record(path),
                "scoring_source_hash_checks": comparison,
                "expected_image_conditions": data["expected_image_conditions"],
            }
        )
    references = [
        "an/weights_provenance.json",
        "an/data/coco_provenance.json",
        "an/data/provenance_synthbuster.json",
        "an/data/inventory_synthbuster.json",
        "an/data/shared_stem_audit.json",
        "an/requirements.txt",
        "an/requirements-lock.txt",
        "an/results/test_summary.json",
        "an/results/documentation_verification.json",
        "an/results/main/run_metadata.json",
        "an/results/main/report.md",
        "an/results/main/analysis_summary.json",
        "an/results/main/verification.json",
        "an/results/main/independent_data_audit.json",
        "an/results/main/metrics.csv",
        "an/results/main/attribution_metrics.csv",
        "an/results/main/content_sensitivity_metrics.csv",
    ]
    docs = sorted(
        [
            path
            for path in paths
            if path.suffix == ".md"
            and (path in (AN / "README.md", AN / "RESULTS.md") or path.is_relative_to(AN / "docs"))
        ]
    )
    data = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "snapshot_notice": "Working-tree hashes at this timestamp, not a Git commit. Source edits after scoring may differ from the frozen phase metadata. Historical study provenance is retained separately in an/source_provenance.json.",
        "repository": {
            "head": command("git", "rev-parse", "HEAD"),
            "origin": command("git", "remote", "get-url", "origin"),
            "official_upstream_declared_by_repository": "https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25",
            "tracked_changes_vs_head": tracked_changes,
            "fsd_changes_vs_head": official_changes,
            "fsd_git_unchanged_vs_head": not official_changes,
            "untracked_an_python_sources": sorted(
                path for path in untracked if path.endswith(".py")
            ),
            "an_work_is_uncommitted": any(path.startswith("an/") for path in untracked),
        },
        "source_selection": {
            "inventory_command": "rg --files fsd an",
            "filter": "Python files only; respect rg ignores; exclude environment/cache/data/results/weights/generated-fixture directories",
            "excluded_directory_names": sorted(EXCLUDED),
            "hash_definition": "SHA256 of exact working-tree file bytes, not Git-normalized blobs",
            "official_fsd_file_count": len(official),
            "an_python_file_count": len(added),
        },
        "official_fsd_python_sources": official,
        "an_python_sources": added,
        "frozen_study_manifests": [
            manifest_record("manifest_real.csv"),
            manifest_record("manifest_fake.csv"),
        ],
        "provenance_references": [
            file_record(ROOT / item) for item in references if (ROOT / item).exists()
        ],
        "weights_notice": "Per-weight release URLs, sizes and recorded SHA256 values are in weights_provenance.json; this source snapshot hashes that provenance file and does not rehash the large weight payloads.",
        "phase_metadata_snapshots": phase_metadata,
        "documentation_snapshot": [file_record(path) for path in docs],
        "final_plot_snapshot": [
            file_record(path) for path in sorted((AN / "results" / "main" / "plots").glob("*.png"))
        ],
        "data_interpretation": {
            "originals": 400,
            "test_originals": 300,
            "calibration_originals": 100,
            "local_calibration_test_partitions": "Disjoint original IDs and file hashes; all derivatives stay with their original",
            "checkpoint_training_overlap": "Unknown: authors' image-level checkpoint training manifests were not published in the inspected release",
            "raise_included": False,
            "native_encodings": "COCO and Firefly originals are JPEG payloads; DALL-E3, SDXL and GLIDE originals are PNG",
            "cross_generator_content_dependence": "Some Synthbuster archive stems recur across generators; distinct bytes do not guarantee independent semantic content. See shared_stem_audit.json and content sensitivity analysis.",
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": (
                    output.relative_to(ROOT).as_posix()
                    if output.is_relative_to(ROOT)
                    else str(output)
                ),
                "head": data["repository"]["head"],
                "official_fsd_files": len(official),
                "an_python_files": len(added),
                "tracked_changes": tracked_changes,
                "all_scoring_source_hashes_match": all(
                    check["matches"]
                    for phase in phase_metadata
                    for check in phase["scoring_source_hash_checks"].values()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
