"""Download reproducible native image subsets without whole multi-GB ZIPs.

Only public, official dataset archives are used. Range reads require HTTP 206
and exact Content-Range validation; each selected member is CRC-32 verified.
No resize, format conversion, or recompression is applied to downloaded bytes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

SYNTHBUSTER_URL = "https://zenodo.org/records/10066460/files/synthbuster.zip?download=1"
RAISE_URL = "https://www.grip.unina.it/download/prog/DMimageDetection/real_RAISE_1k.zip"
FIELDS = [
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
    "archive_member",
    "archive_crc32",
    "selection_seed",
]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class HTTPRangeFile(io.RawIOBase):
    """Seekable remote file with strict bounded HTTP reads and short retries."""

    def __init__(self, url: str, timeout: float = 90, retries: int = 5):
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.pos = 0
        self.total = None
        self.bytes_received = 0
        self._cache_start = 0
        self._cache = b""
        self.range(0, 0)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == 1:
            offset += self.pos
        elif whence == 2:
            offset += self.total
        if offset < 0:
            raise ValueError("Negative file seek")
        self.pos = offset
        return self.pos

    def range(self, start: int, end: int) -> bytes:
        last_error = None
        for attempt in range(self.retries):
            try:
                # A unique query prevents misconfigured intermediary caches from
                # reusing the first partial response for subsequent byte ranges.
                separator = "&" if "?" in self.url else "?"
                request_url = self.url + separator + f"an_range={start}-{end}"
                req = urllib.request.Request(
                    request_url,
                    headers={
                        "User-Agent": "AN-FSD-robustness-research/1.0",
                        "Range": f"bytes={start}-{end}",
                        "Accept-Encoding": "identity",
                    },
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if response.status != 206 or not match:
                        raise RuntimeError(
                            f"Server did not honor byte range: {response.status} {content_range!r}"
                        )
                    actual_start, actual_end, total = map(int, match.groups())
                    if actual_start != start or actual_end != min(end, total - 1):
                        raise RuntimeError(f"Unexpected Content-Range {content_range!r}")
                    if self.total is not None and self.total != total:
                        raise RuntimeError("Remote archive size changed during download")
                    self.total = total
                    size = actual_end - actual_start + 1
                    data = response.read(size + 1)
                    if len(data) != size:
                        raise IOError(
                            f"Truncated or oversized range: expected {size}, got {len(data)}"
                        )
                    self.bytes_received += len(data)
                    return data
            except (urllib.error.URLError, OSError, RuntimeError) as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(min(2**attempt, 10))
        raise RuntimeError(
            f"Range download failed for {self.url} bytes {start}-{end}: {last_error}"
        )

    def read(self, size=-1):
        if size < 0:
            size = self.total - self.pos
        size = min(size, self.total - self.pos)
        if size <= 0:
            return b""
        end = self.pos + size
        if self._cache_start <= self.pos and end <= self._cache_start + len(self._cache):
            result = self._cache[self.pos - self._cache_start : end - self._cache_start]
        else:
            # Read-ahead is deliberately bounded; never cache a whole archive.
            self._cache_start = self.pos
            self._cache = self.range(self.pos, min(self.total - 1, self.pos + max(size, 65536) - 1))
            result = self._cache[:size]
        self.pos += len(result)
        return result


class RemoteZip:
    def __init__(self, url: str):
        self.remote = HTTPRangeFile(url)
        self.archive = zipfile.ZipFile(self.remote)

    def members(self):
        return [
            item
            for item in self.archive.infolist()
            if not item.is_dir()
            and Path(item.filename).suffix.lower() in IMAGE_SUFFIXES
            and not item.filename.startswith("__MACOSX/")
        ]

    def extract_bytes(self, member: zipfile.ZipInfo):
        # One HTTP request normally retrieves local header and compressed bytes.
        # Central/local extra field lengths can differ: request an extra 64 KiB,
        # then use the exact lengths from the actual local header.
        stop = (
            member.header_offset
            + 30
            + len(member.filename.encode("utf-8"))
            + 65535
            + member.compress_size
            - 1
        )
        raw = self.remote.range(member.header_offset, min(stop, self.remote.total - 1))
        if raw[:4] != b"PK\x03\x04":
            raise ValueError(f"Invalid local ZIP header for {member.filename}")
        header = struct.unpack("<4s5H3I2H", raw[:30])
        flags, compression = header[2], header[3]
        name_length, extra_length = header[-2:]
        if flags & 1:
            raise ValueError("Encrypted archive entries are unsupported")
        offset = 30 + name_length + extra_length
        compressed = raw[offset : offset + member.compress_size]
        if len(compressed) != member.compress_size:
            raise ValueError("Truncated compressed member")
        if compression == zipfile.ZIP_STORED:
            data = compressed
        elif compression == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(compressed, -15)
        elif compression == zipfile.ZIP_BZIP2:
            import bz2

            data = bz2.decompress(compressed)
        else:
            raise ValueError(f"Unsupported ZIP compression {compression}")
        if len(data) != member.file_size or (zlib.crc32(data) & 0xFFFFFFFF) != member.CRC:
            raise ValueError(f"CRC/size check failed for {member.filename}")
        return data


def stable_rank(name: str, seed: int):
    return hashlib.sha256(f"{seed}|{name}".encode()).hexdigest()


def source_for_member(name: str):
    """Map only explicit folder names, never guess from image content."""
    normalized = re.sub(r"[^a-z0-9]", "", str(Path(name).parent).lower())
    if "dalle3" in normalized:
        return "DALL-E 3"
    if "firefly" in normalized:
        return "Adobe Firefly"
    if "stablediffusionxl" in normalized or "sdxl" in normalized:
        return "Stable Diffusion XL"
    if "glide" in normalized:
        return "GLIDE"
    return None


def write_manifest(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def acquire_subset(
    root: Path,
    dataset: str,
    per_source: int = 100,
    real_test: int = 100,
    real_calibration: int = 60,
    seed: int = 20261108,
    list_only: bool = False,
    source_url: str | None = None,
):
    """Return manifest rows; checkpoint a valid manifest after every image."""
    from PIL import Image

    root = root.resolve()
    dataset_name = "Synthbuster" if dataset == "synthbuster" else "RAISE-1k"
    url = source_url or (SYNTHBUSTER_URL if dataset == "synthbuster" else RAISE_URL)
    remote_zip = RemoteZip(url)
    members = remote_zip.members()
    print(
        f"Opened {dataset_name}: {len(members)} images, archive bytes={remote_zip.remote.total}",
        flush=True,
    )
    inventory = {}
    for member in members:
        folder = str(Path(member.filename).parent)
        inventory[folder] = inventory.get(folder, 0) + 1
    inventory_path = root / "an" / "data" / f"inventory_{dataset}.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(
            {"url": url, "archive_bytes": remote_zip.remote.total, "folders": inventory}, indent=2
        ),
        encoding="utf-8",
    )
    print(json.dumps(inventory, indent=2), flush=True)
    if list_only:
        return []
    selected = []
    if dataset == "synthbuster":
        selected_by_source = []
        for source in ["DALL-E 3", "Adobe Firefly", "Stable Diffusion XL", "GLIDE"]:
            candidates = [m for m in members if source_for_member(m.filename) == source]
            candidates.sort(key=lambda m: stable_rank(m.filename, seed))
            if len(candidates) < per_source:
                raise ValueError(
                    f"Expected >= {per_source} native {source} images, found {len(candidates)}"
                )
            selected_by_source.append([(m, source, "test", 1) for m in candidates[:per_source]])
        # Interleave sources so early partial manifests already contain each one.
        selected = [item for batch in zip(*selected_by_source) for item in batch]
    else:
        members.sort(key=lambda m: stable_rank(m.filename, seed))
        if len(members) < real_test + real_calibration:
            raise ValueError("Insufficient RAISE images")
        selected = [
            (m, "Real", "calibration" if i < real_calibration else "test", 0)
            for i, m in enumerate(members[: real_calibration + real_test])
        ]
    manifest_path = root / "an" / "data" / f"manifest_{dataset}.csv"
    rows, seen_hashes = [], set()
    for index, (member, source, split, label) in enumerate(selected):
        source_slug = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
        image_id = f"{dataset}_{hashlib.sha256(member.filename.encode()).hexdigest()[:16]}"
        target = (
            root
            / "an"
            / "data"
            / "images"
            / dataset
            / source_slug
            / (image_id + Path(member.filename).suffix.lower())
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            image_bytes = target.read_bytes()
            if (
                len(image_bytes) != member.file_size
                or zlib.crc32(image_bytes) & 0xFFFFFFFF != member.CRC
            ):
                raise ValueError(f"Existing file has wrong CRC: {target}")
        else:
            image_bytes = remote_zip.extract_bytes(member)
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(image_bytes)
            temporary.replace(target)
        sha256 = hashlib.sha256(image_bytes).hexdigest()
        if sha256 in seen_hashes:
            raise ValueError(
                f"Duplicate byte-identical image in selected subset: {member.filename}"
            )
        seen_hashes.add(sha256)
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            width, height = image.size
        row = dict(
            image_id=image_id,
            path=target.relative_to(root).as_posix(),
            label=label,
            source=source,
            split=split,
            sha256=sha256,
            width=width,
            height=height,
            source_url=url,
            dataset=dataset_name,
            archive_member=member.filename,
            archive_crc32=f"{member.CRC:08x}",
            selection_seed=seed,
        )
        rows.append(row)
        write_manifest(manifest_path, rows)
        print(
            f"{index + 1}/{len(selected)} {dataset_name} {source}: {width}x{height} {image_id}",
            flush=True,
        )
    provenance = {
        "dataset": dataset_name,
        "url": url,
        "seed": seed,
        "count": len(rows),
        "downloaded_bytes": remote_zip.remote.bytes_received,
        "archive_bytes": remote_zip.remote.total,
        "selection": "SHA256(seed|archive_member) ranking within source; native bytes",
        "split_policy": "Real calibration/test disjoint original images; synthetic test only",
        "not_claimed": "Exact FSD paper sample IDs or disjointness from unpublished checkpoint training IDs",
    }
    (root / "an" / "data" / f"provenance_{dataset}.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return rows
