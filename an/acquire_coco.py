"""Acquire a seeded subset of official COCO2017 validation photographs.

This uses original JPEG bytes from the official archive. Their existing JPEG
history is retained and must be distinguished from an uncompressed baseline.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import zipfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
URL = "http://images.cocodataset.org/zips/val2017.zip"


def main():
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-count", type=int, default=200)
    parser.add_argument("--calibration-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()
    archive = ROOT / "data" / "archives" / "coco_val2017.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        for attempt in range(5):
            try:
                req = Request(URL, headers={"User-Agent": "AN-FSD-robustness-research"})
                with urlopen(req, timeout=60) as response, partial.open("wb") as out:
                    total = int(response.headers.get("Content-Length", 0))
                    written, last = 0, time.monotonic()
                    while block := response.read(1024 * 1024):
                        out.write(block)
                        written += len(block)
                        if time.monotonic() - last > 15:
                            print(
                                f"COCO download {written / 2**20:.0f}/{total / 2**20:.0f} MiB",
                                flush=True,
                            )
                            last = time.monotonic()
                if total and partial.stat().st_size != total:
                    raise IOError("Incomplete archive")
                partial.replace(archive)
                break
            except Exception as exc:
                print(f"Download attempt {attempt+1}: {exc}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
    rows = []
    image_dir = ROOT / "data" / "images" / "coco2017"
    image_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = [
            i for i in zf.infolist() if not i.is_dir() and i.filename.lower().endswith(".jpg")
        ]
        members.sort(
            key=lambda item: hashlib.sha256(f"{args.seed}:{item.filename}".encode()).hexdigest()
        )
        selection = members[: args.calibration_count + args.test_count]
        if len(selection) != args.calibration_count + args.test_count:
            raise ValueError("Not enough originals for requested selection")
        for index, member in enumerate(selection):
            payload = zf.read(member)  # zipfile verifies CRC32.
            name = Path(member.filename).name
            destination = image_dir / name
            destination.write_bytes(payload)
            with Image.open(destination) as image:
                width, height, fmt = image.width, image.height, image.format
                image.verify()
            rows.append(
                dict(
                    image_id=f"coco2017_{Path(name).stem}",
                    path=destination.relative_to(ROOT.parent).as_posix(),
                    label=0,
                    source="COCO2017",
                    split="calibration" if index < args.calibration_count else "test",
                    sha256=hashlib.sha256(payload).hexdigest(),
                    width=width,
                    height=height,
                    source_url=URL,
                    dataset="COCO2017",
                    original_split="val2017",
                    archive_member=member.filename,
                    format=fmt,
                    sampling_seed=args.seed,
                )
            )
    with (ROOT / "data" / "manifest_coco.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = dict(
        url=URL,
        dataset_page="https://cocodataset.org/#download",
        original_split="val2017",
        archive_bytes=archive.stat().st_size,
        calibration=args.calibration_count,
        test=args.test_count,
        seed=args.seed,
        selection="SHA256(seed:archive_member) ascending, calibration first then test",
        caveat="Original JPEG files; checkpoint training image IDs unavailable, overlap cannot be excluded.",
    )
    (ROOT / "data" / "coco_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
