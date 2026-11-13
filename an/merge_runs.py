"""Merge compatible, independently scored AN phases without rerunning inference.

The output is a new analysis-only snapshot. Its signature intentionally cannot
match run_experiment.py, preventing scoring/resume into a merged directory.
Run phases must be stopped before merging; incomplete expected pairs are retained.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import uuid

AN = Path(__file__).resolve().parent
PROTOCOL_KEYS = (
    "conditions",
    "fsd_config",
    "weights",
    "method",
    "seed",
    "code_sha256",
    "attribution",
    "calibration_all_conditions",
)
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
DEFAULT_SOURCE_MAPPING = {
    "COCO2017": "Real",
    "RAISE-1k": "Real",
    "Real": "Real",
    "DALL-E 3": "DALL-E 3",
    "Adobe Firefly": "Adobe Firefly",
    "Stable Diffusion XL": "Stable Diffusion XL",
}


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_json_snapshot(path: Path):
    payload = path.read_bytes()
    return json.loads(payload.decode("utf-8-sig")), payload


def _validate_directory(path: Path, description: str) -> Path:
    path = path.resolve()
    if not path.is_relative_to(AN):
        raise ValueError(f"{description} must stay under an/: {path}")
    return path


def _load_phase(path: Path) -> dict:
    path = _validate_directory(path, "Input phase")
    metadata, meta_payload = read_json_snapshot(path / "run_metadata.json")
    if metadata.get("merged_run"):
        raise ValueError(f"Input is already a merged run; provide original scoring phases: {path}")
    signature = metadata.get("signature", {})
    missing = [key for key in PROTOCOL_KEYS if key not in signature]
    if missing:
        raise ValueError(f"Missing protocol signature keys in {path}: {missing}")
    protocol = {key: signature[key] for key in PROTOCOL_KEYS}
    canonical(protocol)  # Reject NaN and nonserializable signatures.
    conditions = metadata.get("expected_conditions")
    signed_conditions = [row["name"] for row in protocol["conditions"]]
    if (
        conditions != signed_conditions
        or len(set(conditions)) != len(conditions)
        or "clean" not in conditions
    ):
        raise ValueError(f"Expected conditions disagree with signed protocol in {path}")
    if any(not isinstance(name, str) or not SAFE_NAME.fullmatch(name) for name in conditions):
        raise ValueError(f"Unsafe condition filename in {path}")
    manifest = metadata.get("selected_manifest")
    expected = metadata.get("expected_image_conditions")
    if not isinstance(manifest, list) or not isinstance(expected, dict):
        raise ValueError(f"Missing selected manifest or expected pairs in {path}")
    by_id, hashes = {}, {}
    for row in manifest:
        identifier = str(row.get("image_id", ""))
        digest = str(row.get("sha256", "")).lower()
        if not SAFE_NAME.fullmatch(identifier) or identifier in by_id:
            raise ValueError(f"Invalid/duplicate original image ID in {path}: {identifier!r}")
        if not SHA256.fullmatch(digest) or digest in hashes:
            raise ValueError(f"Missing/duplicate original SHA-256 in {path}: {identifier}")
        if int(row.get("label", -1)) not in (0, 1) or row.get("split") not in {
            "test",
            "calibration",
        }:
            raise ValueError(f"Invalid selected image label/split: {identifier}")
        by_id[identifier], hashes[digest] = row, identifier
    if list(by_id) != signature.get("selected_image_ids") or set(by_id) != set(expected):
        raise ValueError(f"Selected IDs, signature IDs, and expected-pair IDs disagree in {path}")
    for identifier, names in expected.items():
        row = by_id[identifier]
        required = (
            conditions
            if row["split"] == "test" or protocol["calibration_all_conditions"]
            else ["clean"]
        )
        if names != required:
            raise ValueError(f"Unexpected condition plan for {identifier}: {names}")

    progress, progress_payload = {}, None
    if (path / "progress.json").is_file():
        progress, progress_payload = read_json_snapshot(path / "progress.json")
        if progress.get("state") == "running":
            raise ValueError(
                f"Phase still reports running; finish or stop it before merging: {path}"
            )
    records_payload = (path / "records.jsonl").read_bytes()
    records, done, artifacts = [], set(), []
    for line_number, line in enumerate(records_payload.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Malformed/incomplete records line {line_number} in {path}"
            ) from error
        identifier, condition = str(record.get("image_id", "")), record.get("condition")
        if identifier not in by_id or condition not in expected[identifier]:
            raise ValueError(f"Unplanned scored pair in {path}: {identifier}/{condition}")
        key = identifier, condition
        if key in done:
            raise ValueError(f"Duplicate scored pair in {path}: {key}")
        original = by_id[identifier]
        for field in ("source", "split", "sha256"):
            if str(record.get(field, "")) != str(original[field]):
                raise ValueError(f"Record {field} differs from selected manifest: {key}")
        if int(record.get("label", -1)) != int(original["label"]):
            raise ValueError(f"Record label differs from selected manifest: {key}")
        for field in ("z_score", "anomaly_score", "raw_score"):
            if not isinstance(record.get(field), (int, float)) or not math.isfinite(record[field]):
                raise ValueError(f"Nonfinite/missing {field} in {path}: {key}")
        if not math.isclose(
            record["anomaly_score"], -record["z_score"], rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"Anomaly/z-score sign mismatch: {key}")
        stem = f"{identifier}__{condition}"
        feature_relative = Path("features") / f"{stem}.npz"
        stored_relative = Path(record.get("feature_path", str(feature_relative)))
        if (
            stored_relative.is_absolute()
            or stored_relative.as_posix() != feature_relative.as_posix()
        ):
            raise ValueError(f"Unexpected feature path in {path}: {stored_relative}")
        for folder, suffix, required in (
            ("features", ".npz", True),
            ("previews", ".npz", False),
            ("examples", ".png", False),
        ):
            relative = Path(folder) / f"{stem}{suffix}"
            source = (path / relative).resolve()
            if not source.is_relative_to(path):
                raise ValueError(f"Artifact escapes input phase: {source}")
            if not source.is_file():
                if required:
                    raise FileNotFoundError(f"Scored pair has no cached feature file: {source}")
                continue
            artifacts.append((relative, source))
        record = dict(record)
        record["feature_path"] = feature_relative.as_posix()
        record["merged_from_phase"] = path.name
        records.append(record)
        done.add(key)
    expected_count = sum(len(names) for names in expected.values())
    return dict(
        path=path,
        metadata=metadata,
        protocol=protocol,
        manifest=manifest,
        by_id=by_id,
        hashes=hashes,
        expected=expected,
        records=records,
        records_payload=records_payload,
        meta_payload=meta_payload,
        progress=progress,
        progress_payload=progress_payload,
        artifacts=artifacts,
        expected_count=expected_count,
        missing_count=expected_count - len(done),
    )


def merge_runs(
    input_dirs: list[str | Path],
    output: str | Path,
    source_mapping: dict[str, str] | str | Path | None = None,
) -> dict:
    """Create a validated analysis-only run; output must not already exist."""
    if len(input_dirs) < 2:
        raise ValueError("At least two independently scored phases are required")
    output = _validate_directory(Path(output), "Output")
    if output.exists():
        raise FileExistsError(f"Output already exists; use a new directory: {output}")
    phases = [_load_phase(Path(path)) for path in input_dirs]
    roots = [phase["path"] for phase in phases]
    if len(set(roots)) != len(roots):
        raise ValueError("The same input phase was specified more than once")
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in roots):
        raise ValueError("Output and input phases must not contain one another")
    reference = phases[0]["protocol"]
    for phase in phases[1:]:
        different = [
            key
            for key in PROTOCOL_KEYS
            if canonical(reference[key]) != canonical(phase["protocol"][key])
        ]
        if different:
            raise ValueError(f"Incompatible scoring protocol in {phase['path']}: {different}")
        first_environment = phases[0]["metadata"].get("environment", {})
        next_environment = phase["metadata"].get("environment", {})
        numerical_keys = (
            "torch",
            "numpy",
            "pillow",
            "dtype",
            "tf32",
            "device",
            "cuda_runtime",
            "fre_shape",
            "source_gmm_names",
        )
        different_runtime = [
            key
            for key in numerical_keys
            if canonical(first_environment.get(key)) != canonical(next_environment.get(key))
        ]
        if different_runtime:
            raise ValueError(
                f"Incompatible numerical runtime in {phase['path']}: {different_runtime}"
            )
    manifest, expected, records, seen_ids, seen_hashes = [], {}, [], {}, {}
    for phase in phases:
        for identifier, row in phase["by_id"].items():
            digest = row["sha256"].lower()
            if identifier in seen_ids:
                raise ValueError(
                    f"Image ID occurs in multiple phases: {identifier} ({seen_ids[identifier]} and {phase['path']})"
                )
            if digest in seen_hashes:
                raise ValueError(
                    f"Byte-identical original occurs in multiple phases: {identifier} and {seen_hashes[digest]}"
                )
            seen_ids[identifier], seen_hashes[digest] = str(phase["path"]), identifier
        manifest.extend(phase["manifest"])
        expected.update(phase["expected"])
        records.extend(phase["records"])
    all_sources = {row["source"] for row in manifest}
    if source_mapping is None:
        mapping = {
            source: value
            for source, value in DEFAULT_SOURCE_MAPPING.items()
            if source in all_sources
        }
    elif isinstance(source_mapping, dict):
        mapping = dict(source_mapping)
    else:
        mapping = json.loads(Path(source_mapping).read_text(encoding="utf-8-sig"))
    if not isinstance(mapping, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in mapping.items()
    ):
        raise ValueError(
            "source_mapping must be an object mapping source names to checkpoint names"
        )
    checkpoint_sources = set(
        phases[0]["metadata"].get("environment", {}).get("source_gmm_names", [])
    )
    if not checkpoint_sources:
        checkpoint_sources = {name for row in records for name in row.get("attribution_loglik", {})}
    if (
        reference["attribution"]
        and checkpoint_sources
        and set(mapping.values()) - checkpoint_sources
    ):
        raise ValueError(
            f"Mappings target nonexistent checkpoint labels: {set(mapping.values()) - checkpoint_sources}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = _validate_directory(
        output.parent / (output.name + ".merge-staging-" + uuid.uuid4().hex[:10]), "Staging output"
    )
    staging.mkdir()
    artifact_inventory, provenance = [], []
    for index, phase in enumerate(phases):
        prefix = f"phase_{index:02d}"
        snapshot_dir = staging / "phases" / prefix
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "run_metadata.json").write_bytes(phase["meta_payload"])
        (snapshot_dir / "records.jsonl").write_bytes(phase["records_payload"])
        if phase["progress_payload"] is not None:
            (snapshot_dir / "progress.json").write_bytes(phase["progress_payload"])
        hashes = {
            "run_metadata.json": digest_bytes(phase["meta_payload"]),
            "records.jsonl": digest_bytes(phase["records_payload"]),
        }
        if phase["progress_payload"] is not None:
            hashes["progress.json"] = digest_bytes(phase["progress_payload"])
        if (phase["path"] / "errors.jsonl").is_file():
            payload = (phase["path"] / "errors.jsonl").read_bytes()
            (snapshot_dir / "errors.jsonl").write_bytes(payload)
            hashes["errors.jsonl"] = digest_bytes(payload)
        for relative, source in phase["artifacts"]:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_hash = digest_file(source)
            shutil.copyfile(source, destination)
            if digest_file(destination) != source_hash:
                raise IOError(
                    f"Copied artifact hash mismatch: {source}; staging retained at {staging}"
                )
            artifact_inventory.append(
                dict(
                    path=relative.as_posix(),
                    source_phase=prefix,
                    bytes=destination.stat().st_size,
                    sha256=source_hash,
                )
            )
        # Detect appending/resuming during the merge. Never publish a moving input.
        if (
            digest_file(phase["path"] / "records.jsonl") != hashes["records.jsonl"]
            or digest_file(phase["path"] / "run_metadata.json") != hashes["run_metadata.json"]
        ):
            raise RuntimeError(
                f"Input phase changed during merge: {phase['path']}; staging retained at {staging}"
            )
        provenance.append(
            dict(
                phase=prefix,
                original_directory=str(phase["path"]),
                snapshot_directory=(Path("phases") / prefix).as_posix(),
                hashes=hashes,
                expected_pairs=phase["expected_count"],
                completed_pairs=len(phase["records"]),
                missing_pairs=phase["missing_count"],
                original_progress=phase["progress"],
            )
        )
    records_bytes = (
        "\n".join(json.dumps(record, allow_nan=False) for record in records) + "\n"
    ).encode("utf-8")
    (staging / "records.jsonl").write_bytes(records_bytes)
    now = datetime.now(timezone.utc).isoformat()
    expected_count = sum(len(value) for value in expected.values())
    metadata = dict(
        created_utc=now,
        merged_run=True,
        scoring_resume_supported=False,
        signature={
            **reference,
            "merged_analysis_only": True,
            "phase_metadata_sha256": [row["hashes"]["run_metadata.json"] for row in provenance],
        },
        expected_conditions=phases[0]["metadata"]["expected_conditions"],
        expected_image_conditions=expected,
        selected_manifest=manifest,
        environment=phases[0]["metadata"].get("environment", {}),
        source_mapping=mapping,
        phase_provenance=provenance,
        merged_records_sha256=digest_bytes(records_bytes),
        merge_script_sha256=digest_file(Path(__file__)),
        notes=[
            "Analysis-only merge; resume original scoring phases separately.",
            "Manifest hashes/selected IDs differ by phase; all declared scoring protocol fields match exactly.",
            "Missing expected image/condition pairs remain in the merged plan and must be reported.",
            "Clean means native supplied bytes with no added processing; it does not guarantee lossless or never-compressed content.",
        ],
    )
    progress = dict(
        state="complete" if len(records) == expected_count else "incomplete_merged_phases",
        completed=len(records),
        expected=expected_count,
        missing=expected_count - len(records),
        updated_utc=now,
        merged_run=True,
    )
    for filename, value in (
        ("run_metadata.json", metadata),
        ("progress.json", progress),
        ("source_mapping.json", mapping),
        ("artifact_hashes.json", artifact_inventory),
    ):
        (staging / filename).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )
    (staging / "MERGED_ANALYSIS_ONLY.txt").write_text(
        "This is a validated analysis-only merge. Do not run inference or resume scoring here.\n"
        "Resume original phases, then merge to a new output directory if further results are needed.\n",
        encoding="utf-8",
    )
    # Both resolved paths were checked to stay under an/ before this directory move.
    staging.rename(output)
    return dict(
        output=str(output),
        **progress,
        originals=len(manifest),
        artifacts=len(artifact_inventory),
        source_counts=dict(Counter(row["source"] for row in manifest)),
        source_mapping=mapping,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Stopped scoring-phase directories")
    parser.add_argument(
        "--output", required=True, type=Path, help="New analysis-only directory beneath an/"
    )
    parser.add_argument(
        "--source-mapping", type=Path, help="Optional explicit JSON source-to-checkpoint mapping"
    )
    args = parser.parse_args()
    result = merge_runs(args.runs, args.output, args.source_mapping)
    print(json.dumps(result, indent=2))
    print(f"Next: an/.venv/Scripts/python.exe an/analyze_results.py {args.output}")


if __name__ == "__main__":
    main()
