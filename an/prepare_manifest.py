"""Freeze an audited, deterministic subset from acquisition manifests.

Input order is the acquisition seed's stable rank. Incomplete acquisition
manifests are accepted only when every requested source/split is already ready.
No image processing or new split assignment occurs here.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

AN = Path(__file__).resolve().parent
ROOT = AN.parent
SYNTHETIC_SOURCES = ["DALL-E 3", "Adobe Firefly", "Stable Diffusion XL", "GLIDE"]
CORE_FIELDS = [
    "image_id",
    "path",
    "label",
    "source",
    "split",
    "sha256",
    "width",
    "height",
    "source_url",
    "dataset",
    "format",
    "archive_member",
    "archive_crc32",
    "selection_seed",
    "original_split",
    "acquisition_manifest",
]


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("Counts must be nonnegative")
    return result


def read_manifest(path):
    if not path.is_file():
        raise ValueError(f"Acquisition is not ready: missing {path}")
    payload = path.read_bytes()  # One immutable snapshot while downloader runs.
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    return rows, hashlib.sha256(payload).hexdigest()


def select(rows, count, predicate, description):
    candidates = [dict(row) for row in rows if predicate(row)]
    if len(candidates) < count:
        raise ValueError(
            f"Acquisition is not ready for {description}: requested {count}, available {len(candidates)}"
        )
    return candidates[:count]


def verify_rows(rows):
    seen_ids, seen_paths, seen_hashes = set(), set(), {}
    for row in rows:
        image_id = row["image_id"]
        if not image_id or image_id in seen_ids:
            raise ValueError(f"Empty or duplicate image ID: {image_id!r}")
        seen_ids.add(image_id)
        if str(row["label"]) not in {"0", "1"} or row["split"] not in {"calibration", "test"}:
            raise ValueError(f"Invalid label/split for {image_id}")
        if row["split"] == "calibration" and str(row["label"]) != "0":
            raise ValueError("Calibration must contain real originals only")
        path = Path(row["path"])
        path = (path if path.is_absolute() else ROOT / path).resolve()
        if not path.is_relative_to(AN / "data" / "images"):
            raise ValueError(f"Input image is outside an/data/images: {path}")
        if path in seen_paths:
            raise ValueError(f"Duplicate original path: {path}")
        seen_paths.add(path)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if row.get("sha256") != digest:
            raise ValueError(f"SHA-256 mismatch for {image_id}")
        if digest in seen_hashes:
            raise ValueError(f"Duplicate image bytes: {image_id} and {seen_hashes[digest]}")
        seen_hashes[digest] = image_id
        with Image.open(io.BytesIO(payload)) as image:
            native_format = image.format
            width, height = image.size
            image.verify()
        if int(row["width"]) != width or int(row["height"]) != height:
            raise ValueError(f"Native dimensions disagree for {image_id}")
        row["path"] = path.relative_to(ROOT).as_posix()
        row["format"] = native_format
        row["selection_seed"] = row.get("selection_seed") or row.get("sampling_seed", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-source", type=nonnegative, default=50)
    parser.add_argument("--coco-test", type=nonnegative, default=100)
    parser.add_argument("--raise-test", type=nonnegative, default=0)
    parser.add_argument("--coco-calibration", type=nonnegative, default=None)
    parser.add_argument("--raise-calibration", type=nonnegative, default=None)
    parser.add_argument(
        "--calibration-count-per-domain",
        type=nonnegative,
        default=None,
        help="Set both domains unless a domain-specific calibration count is supplied",
    )
    parser.add_argument("--output", type=Path, default=AN / "data" / "manifest_main.csv")
    parser.add_argument(
        "--force", action="store_true", help="Explicitly replace a previously frozen manifest"
    )
    args = parser.parse_args()
    common = args.calibration_count_per_domain
    coco_cal = (
        args.coco_calibration
        if args.coco_calibration is not None
        else (common if common is not None else 100)
    )
    raise_cal = (
        args.raise_calibration
        if args.raise_calibration is not None
        else (common if common is not None else 0)
    )
    output = args.output.resolve()
    if not output.is_relative_to(AN):
        parser.error("Output must stay under an/")
    selected, inputs = [], {}
    counts = {
        "synthetic_test_per_source": args.per_source,
        "coco_test": args.coco_test,
        "coco_calibration": coco_cal,
        "raise_test": args.raise_test,
        "raise_calibration": raise_cal,
    }
    try:
        if args.per_source:
            source_path = AN / "data" / "manifest_synthbuster.csv"
            rows, digest = read_manifest(source_path)
            inputs[source_path.relative_to(ROOT).as_posix()] = digest
            groups = [
                select(
                    rows,
                    args.per_source,
                    lambda row, source=source: row["source"] == source
                    and row["split"] == "test"
                    and str(row["label"]) == "1",
                    f"Synthbuster/{source}/test",
                )
                for source in SYNTHETIC_SOURCES
            ]
            for batch in zip(*groups):
                for row in batch:
                    row["acquisition_manifest"] = source_path.relative_to(ROOT).as_posix()
                    selected.append(row)
        for dataset, source, filename, test_count, calibration_count in [
            ("COCO2017", "COCO2017", "manifest_coco.csv", args.coco_test, coco_cal),
            ("RAISE-1k", "RAISE-1k", "manifest_raise.csv", args.raise_test, raise_cal),
        ]:
            if not test_count and not calibration_count:
                continue
            source_path = AN / "data" / filename
            rows, digest = read_manifest(source_path)
            inputs[source_path.relative_to(ROOT).as_posix()] = digest
            for split, count in [("calibration", calibration_count), ("test", test_count)]:
                chosen = select(
                    rows,
                    count,
                    lambda row, split=split: row["split"] == split and str(row["label"]) == "0",
                    f"{dataset}/{split}",
                )
                for row in chosen:
                    row["source"] = source
                    row["dataset"] = dataset
                    row["acquisition_manifest"] = source_path.relative_to(ROOT).as_posix()
                    selected.append(row)
        if not selected:
            raise ValueError("Requested subset is empty")
        verify_rows(selected)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer, fieldnames=CORE_FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(selected)
        manifest_bytes = buffer.getvalue().encode("utf-8")
        if output.exists() and output.read_bytes() != manifest_bytes and not args.force:
            raise ValueError(
                f"Frozen manifest already exists with different contents: {output}. Choose another output or explicitly --force."
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists() or output.read_bytes() != manifest_bytes:
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_bytes(manifest_bytes)
            temporary.replace(output)
        summary = Counter(f"{r['source']} | {r['split']} | {r['format']}" for r in selected)
        provenance = {
            "manifest": output.relative_to(ROOT).as_posix(),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "original_count": len(selected),
            "requested_counts": counts,
            "actual_counts": dict(sorted(summary.items())),
            "input_manifest_snapshots_sha256": inputs,
            "selection": "First requested count per source/split in acquisition's pre-existing deterministic seeded rank; no repartitioning",
            "validation": "Unique image IDs, resolved original paths and SHA256; recorded hashes/dimensions verified against native image bytes; real-only calibration",
            "limitations": "Not the FSD paper's unpublished ID split; overlap with released pretrained source models is unknown",
        }
        provenance_path = output.with_suffix(".provenance.json")
        # Preserve the first snapshot when an identical frozen manifest is reused.
        if not provenance_path.exists() or args.force:
            temporary = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
            temporary.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            temporary.replace(provenance_path)
        print(json.dumps(provenance, indent=2), flush=True)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"prepare_manifest: {error}\n")


if __name__ == "__main__":
    main()
