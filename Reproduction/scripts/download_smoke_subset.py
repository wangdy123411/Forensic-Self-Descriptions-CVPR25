from project_paths import PROJECT_ROOT

import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

import pyarrow.parquet as pq
from PIL import Image

DATASET = "TheKernel01/Tiny-GenImage"
REVISION = "89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69"
SHARD = "data/validation-00000-of-00004.parquet"
SIZE = 433181356
URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{SHARD}"
GENERATORS = {0: "Real", 1: "ADM", 2: "BigGAN", 3: "GLIDE", 4: "Midjourney", 5: "SD14", 6: "SD15", 7: "VQDM", 8: "Wukong"}
PROJECT = PROJECT_ROOT
DEST = PROJECT / "Data" / "tiny_genimage_smoke_200"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class RangeFile(io.RawIOBase):
    def __init__(self):
        super().__init__()
        self.pos = 0
        self.transferred = 0
        self.cache = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=os.SEEK_SET):
        self.pos = offset if whence == os.SEEK_SET else (self.pos if whence == os.SEEK_CUR else SIZE) + offset
        if self.pos < 0:
            raise ValueError("Negative seek")
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = SIZE - self.pos
        n = min(n, SIZE - self.pos)
        if n <= 0:
            return b""
        key = (self.pos, n)
        if key in self.cache:
            self.pos += n
            return self.cache[key]
        if self.transferred + n > 520_000_000:
            raise RuntimeError("Download safety limit exceeded")
        start = self.pos
        for attempt in range(4):
            try:
                request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{start+n-1}", "User-Agent": "CE7454-dataset-download/1.0"})
                with OPENER.open(request, timeout=45) as response:
                    expected = f"bytes {start}-{start+n-1}/{SIZE}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError("Server did not honor the requested byte range")
                    data = response.read(n + 1)
                if len(data) != n:
                    raise RuntimeError("Incomplete byte range")
                self.transferred += n
                self.pos += n
                if n < 100000:
                    self.cache[key] = data
                print(f"Downloaded range: {n / 1048576:.2f} MiB; total {self.transferred / 1048576:.2f} MiB", flush=True)
                return data
            except Exception as exc:
                if attempt == 3:
                    raise
                print(f"Retry {attempt+1}: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(2 ** attempt)


def write_bytes(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing to overwrite different existing file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


def run(inspect_only=False):
    remote = RangeFile()
    parquet = pq.ParquetFile(remote, pre_buffer=False)
    print("Metadata:", json.dumps({"rows": parquet.metadata.num_rows, "row_groups": parquet.metadata.num_row_groups, "schema": str(parquet.schema_arrow)}), flush=True)
    if inspect_only:
        print([(parquet.metadata.row_group(i).num_rows, parquet.metadata.row_group(i).total_byte_size) for i in range(parquet.metadata.num_row_groups)], flush=True)
        return
    if not DEST.resolve().is_relative_to((PROJECT / "Data").resolve()):
        raise RuntimeError("Unexpected destination")
    counts = {0: 0, 1: 0}
    seen = set()
    records = []
    skipped = collections.Counter()
    offset = 0
    for group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(group, use_threads=False)
        for local_index, row in enumerate(table.to_pylist()):
            label = int(row["label"])
            if label not in counts:
                raise RuntimeError("Unexpected label")
            if counts[label] >= 100:
                continue
            raw = row["image"]["bytes"]
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen:
                skipped["duplicate"] += 1
                continue
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
                fmt = image.format
                image.verify()
            if min(width, height) < 256:
                skipped["short_side_below_256"] += 1
                continue
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
            extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tiff"}.get(fmt)
            if extension is None:
                raise RuntimeError(f"Unsupported format: {fmt}")
            generator_id = int(row["generator"])
            generator = GENERATORS[generator_id]
            if (label == 0) != (generator_id == 0):
                raise RuntimeError("Inconsistent label and generator")
            kind = "real" if label == 0 else "fake"
            relative = f"{kind}/{kind}_{counts[label]+1:04d}_{generator.lower()}{extension}"
            write_bytes(DEST / relative, raw)
            counts[label] += 1
            seen.add(digest)
            records.append({"relative_path": relative, "label": label, "label_name": kind, "generator": generator, "width": width, "height": height, "format": fmt, "bytes": len(raw), "sha256": digest, "source_dataset": DATASET, "source_revision": REVISION, "source_split": "validation", "source_shard": SHARD, "source_row": offset+local_index, "source_image_path": row["image"].get("path") or ""})
        offset += table.num_rows
        print(f"Selected after group {group}: real={counts[0]}, fake={counts[1]}, skipped={dict(skipped)}", flush=True)
        if counts == {0: 100, 1: 100}:
            break
    if counts != {0: 100, 1: 100}:
        raise RuntimeError(f"Not enough eligible images: {counts}")
    records.sort(key=lambda r: (r["label"], r["relative_path"]))
    manifest = io.StringIO(newline="")
    writer = csv.DictWriter(manifest, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    write_bytes(DEST / "manifest.csv", manifest.getvalue().encode("utf-8-sig"))
    info = {
        "dataset": DATASET, "dataset_url": f"https://huggingface.co/datasets/{DATASET}",
        "revision": REVISION, "split": "validation", "source_url": URL,
        "upstream_source": "https://www.kaggle.com/datasets/yangsangtai/tiny-genimage",
        "dataset_card_license": "CC BY-NC-SA 4.0",
        "purpose": "Inference pipeline smoke test; not the complete FSD paper benchmark",
        "selection": "First 100 unique, decodable images per label in validation shard 0, in row order, with both dimensions >=256. Deterministic convenience subset, not a representative random sample.",
        "image_handling": "Exact image.bytes from the pinned Parquet source; no resizing, cropping, recompression, or re-encoding. Upstream transformations before upload are not independently verified.",
        "label_mapping": {"0": "real", "1": "fake"},
        "counts": {"real": counts[0], "fake": counts[1]},
        "generator_counts": dict(collections.Counter(r["generator"] for r in records)),
        "total_image_bytes": sum(r["bytes"] for r in records),
        "min_width": min(r["width"] for r in records), "min_height": min(r["height"] for r in records),
        "skipped": dict(skipped), "source_rows_scanned": offset,
        "validation": "All 200 images passed PIL verify(), full decode, label consistency, SHA-256 uniqueness, and byte-preserving extraction."
    }
    write_bytes(DEST / "dataset_info.json", (json.dumps(info, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    notes = """Tiny-GenImage inference smoke-test subset
========================================

Contents: real/ contains 100 real images; fake/ contains 100 AI-generated images.
Labels: 0 = real, 1 = fake. manifest.csv records labels, generators, dimensions,
source revision and row, and SHA-256 checksums. dataset_info.json records provenance.

Source: https://huggingface.co/datasets/TheKernel01/Tiny-GenImage
Upstream: https://www.kaggle.com/datasets/yangsangtai/tiny-genimage
The source dataset card declares CC BY-NC-SA 4.0. Retain attribution and consult
the source terms when reusing or redistributing images.

Selection: deterministic first 100 eligible unique images per class from the
first validation shard, with short side >=256 pixels. This is a community subset
for checking the inference workflow, not the complete FSD paper evaluation set.
It is not a statistically representative sample. Images retain the exact bytes
stored in the source Parquet file; no local resizing or JPEG recompression was
performed. Pre-upload transformations by upstream curators are unverified.

After installing the FSD project dependencies and downloading weights, run from
the Project2/Code directory (the CLI scans one directory at a time):

  fsd-score --dir ../Data/tiny_genimage_smoke_200/real --csv
  fsd-score --dir ../Data/tiny_genimage_smoke_200/fake --csv

Inference has not been run during dataset preparation.
"""
    write_bytes(DEST / "README.txt", notes.encode("utf-8"))
    print("COMPLETE", json.dumps(info, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    run(parser.parse_args().inspect)
