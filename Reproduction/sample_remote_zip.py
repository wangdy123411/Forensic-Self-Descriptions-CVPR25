from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import http.client
import io
import json
from pathlib import Path, PurePosixPath
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib

from zeroshot_common import IMAGE_SUFFIXES, MANIFEST_FIELDS, child_path, image_id, write_csv, write_json


class TransferBudget:
    def __init__(self, limit: int):
        self.limit, self.used = limit, 0


class RangeFile(io.RawIOBase):

    def __init__(self, url, size, budget, *, chunk_size=1024 * 1024, retries=3, timeout=30):
        if urllib.parse.urlparse(url).scheme not in ("http", "https"):
            raise ValueError("ZIP URL must be public HTTP(S)")
        if type(size) is not int or size <= 0:
            raise ValueError("Archive size must be a positive integer")
        self.url, self.size, self.budget, self.pos = url, size, budget, 0
        self.chunk_size, self.retries, self.timeout = chunk_size, retries, timeout

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2): raise ValueError("Invalid seek mode")
        pos = offset + (0 if whence == 0 else self.pos if whence == 1 else self.size)
        if pos < 0: raise ValueError("Negative seek")
        self.pos = pos
        return pos

    def _range(self, start, count):
        for attempt in range(self.retries + 1):
            if self.budget.used + count + 1 > self.budget.limit:
                raise ValueError("Transfer budget exceeded; increase --max-transfer-mib explicitly")
            req = urllib.request.Request(self.url, headers={
                "Range": f"bytes={start}-{start + count - 1}",
                "Accept-Encoding": "identity", "User-Agent": "FSD-ZIP-sampler/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    expected = f"bytes {start}-{start + count - 1}/{self.size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor exact Range; refusing full ZIP or HTML response")
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise ValueError("Unexpected HTTP content encoding")
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None and int(content_length) != count:
                        raise ValueError("Range Content-Length mismatch")
                    try:
                        data = response.read(count + 1)
                    except http.client.IncompleteRead as error:
                        self.budget.used += len(error.partial)
                        raise OSError("Truncated byte range") from error
                    self.budget.used += len(data)
                    if len(data) != count: raise OSError("Truncated byte range")
                    return data
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == self.retries: raise
            except (OSError, TimeoutError):
                if attempt == self.retries: raise
            time.sleep(min(2 ** attempt, 4))

    def read(self, count=-1):
        count = max(0, min(self.size - self.pos, count if count >= 0 else self.size - self.pos))
        if count > 64 * 1024 * 1024:
            raise ValueError("ZIP metadata/read exceeds 64 MiB safety bound")
        parts = []
        while count:
            data = self._range(self.pos, min(count, self.chunk_size))
            parts.append(data)
            count -= len(data)
            self.pos += len(data)
        return b"".join(parts)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def select_members(config, per_source, seed, budget, *, max_image_bytes):
    archives, selected, archive_ids, sources = {}, [], set(), set()
    specs = config.get("archives")
    if not isinstance(specs, list) or not specs: raise ValueError("Require nonempty archives list")
    try:
        for spec in specs:
            aid = spec["id"]
            if not isinstance(aid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", aid) or aid in archive_ids:
                raise ValueError("Archive id must be unique and use letters, digits, underscore or hyphen")
            archive_ids.add(aid)
            z = zipfile.ZipFile(RangeFile(spec["url"], spec["size"], budget))
            archives[aid] = z
            infos = {i.filename: i for i in z.infolist()}
            if len(infos) != len(z.infolist()): raise ValueError("Duplicate ZIP member names")
            groups = spec.get("groups")
            if not isinstance(groups, list) or not groups: raise ValueError("Require nonempty groups")
            used = set()
            for group in groups:
                source, label = group["source"], group["label"]
                if not isinstance(source, str) or not source.strip() or source in sources:
                    raise ValueError("Source names must be nonempty and unique across the sampling config")
                if type(label) is not int or label not in (0, 1): raise ValueError("label must be 0 or 1")
                sources.add(source)
                prefixes = group["prefixes"]
                if not isinstance(prefixes, list) or not prefixes or any(not isinstance(p, str) or not p.endswith("/") for p in prefixes):
                    raise ValueError("prefixes must be a nonempty list of directory prefixes ending in /")
                count = group.get("count", per_source)
                if type(count) is not int or count <= 0: raise ValueError("count must be a positive integer")
                candidates = sorted(name for name, info in infos.items() if not info.is_dir()
                    and PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES and any(name.startswith(p) for p in prefixes))
                if len(candidates) < count: raise ValueError(f"{source}: only {len(candidates)} images, requested {count}")
                random.Random(f"{seed}:{aid}:{source}").shuffle(candidates)
                for name in sorted(candidates[:count]):
                    info = infos[name]
                    if name in used: raise ValueError("Same member selected for multiple source groups")
                    if info.file_size <= 0 or info.file_size > max_image_bytes or info.compress_size > max_image_bytes:
                        raise ValueError(f"Image exceeds size bounds: {name}")
                    if info.flag_bits & 1: raise ValueError("Encrypted ZIP members are unsupported")
                    used.add(name)
                    destination = group.get("destination", spec.get("destination"))
                    if destination is not None:
                        if not isinstance(destination, str) or not destination:
                            raise ValueError("destination must be a nonempty relative directory")
                        if "\\" in name or ":" in name or name.startswith("/") or any(p in ("", ".", "..") for p in name.split("/")):
                            raise ValueError("Unsafe ZIP member path")
                        relative = destination.rstrip("/") + "/" + name
                        child_path(Path.cwd(), relative)
                    else:
                        relative = f"images/{aid}/{digest(name.encode())}{PurePosixPath(name).suffix.lower()}"
                    selected.append({"archive_id": aid, "archive_url": spec["url"], "archive_size": spec["size"],
                        "member": name, "member_size": info.file_size, "member_crc32": f"{info.CRC:08x}",
                        "source": source, "label": str(label), "split": "test",
                        "relative_path": relative})
        if len({r["relative_path"].casefold() for r in selected}) != len(selected):
            raise ValueError("Sampling destinations collide")
        return archives, selected
    except Exception:
        for z in archives.values(): z.close()
        raise


def run(config, root, *, per_source=1000, seed=2026, max_transfer=256 * 1024 * 1024, max_image_bytes=64 * 1024 * 1024, resume=False, plan=False):
    if per_source <= 0 or max_transfer <= 0 or max_image_bytes <= 0: raise ValueError("Limits must be positive")
    root = Path(root).resolve()
    state_path, manifest_path = root / "sample_state.json", root / "manifest.csv"
    fingerprint = digest(json.dumps({"config": config, "per_source": per_source, "seed": seed}, sort_keys=True).encode())
    existing = None
    if state_path.exists() and not plan:
        if not resume: raise ValueError("Existing sampling state; use --resume")
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if existing["config_sha256"] != fingerprint: raise ValueError("Resume config or seed differs")
    elif not plan and (manifest_path.exists() or (root / "images").exists()):
        raise ValueError("Untracked sampling output exists; choose a new output root")
    budget = TransferBudget(max_transfer)
    archives, selected = select_members(config, per_source, seed, budget, max_image_bytes=max_image_bytes)
    try:
        if plan:
            return {"status": "plan", "images": selected, "transferred_bytes_this_run": budget.used,
                "selected_uncompressed_bytes": sum(r["member_size"] for r in selected),
                "note": "ZIP directory metadata only; no images downloaded or output files written"}
        old = {row["relative_path"]: row for row in existing["images"]} if existing else {}
        if existing and existing["selection"] != selected: raise ValueError("Remote archive selection changed")
        if set(old) - {r["relative_path"] for r in selected}: raise ValueError("Unrecognized resume entries")

        for row in selected:
            previous = old.get(row["relative_path"])
            if previous:
                target = child_path(root, row["relative_path"])
                if any(previous.get(k) != v for k, v in row.items()): raise ValueError("Resume provenance differs")
                if not target.is_file() or target.stat().st_size != row["member_size"]:
                    raise ValueError(f"Resume image missing or size differs: {target}")
                if digest(target.read_bytes()) != previous["sha256"]:
                    raise ValueError(f"Resume SHA256 mismatch: {target}")
        completed = dict(old)
        state = {"config_sha256": fingerprint, "seed": seed, "status": "partial",
            "exact_paper_sample_list": False, "byte_preserving": True, "selection": selected, "images": []}
        root.mkdir(parents=True, exist_ok=True)

        def checkpoint():
            state["transferred_bytes_this_run"] = budget.used
            state["images"] = [completed[r["relative_path"]] for r in selected if r["relative_path"] in completed]
            write_json(state_path, state)
            write_csv(manifest_path, MANIFEST_FIELDS, [{k: row[k] for k in MANIFEST_FIELDS} for row in state["images"]])

        for row in selected:
            target = child_path(root, row["relative_path"])
            previous = old.get(row["relative_path"])
            if previous:
                continue
            if target.exists(): raise ValueError(f"Refusing to overwrite untracked image: {target}")
            data = archives[row["archive_id"]].read(row["member"])
            if len(data) != row["member_size"] or f"{zlib.crc32(data):08x}" != row["member_crc32"]:
                raise ValueError(f"ZIP member CRC/size mismatch: {row['member']}")
            sha = digest(data)
            record = {**row, "sha256": sha, "image_id": image_id(row["source"], row["relative_path"], sha)}
            if not previous:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(data)
                temporary.replace(target)
            completed[row["relative_path"]] = record
            checkpoint()
        state["status"] = "complete"
        checkpoint()
        return state
    finally:
        for archive in archives.values(): archive.close()


def main():
    parser = argparse.ArgumentParser(description='sample remote zip')
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-transfer-mib", type=int, default=256)
    parser.add_argument("--max-image-mib", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Read remote ZIP directories only; write nothing and download no image bytes")
    args = parser.parse_args()
    try:
        state = run(json.loads(args.config.read_text(encoding="utf-8-sig")), args.output_root,
            per_source=args.per_source, seed=args.seed, max_transfer=args.max_transfer_mib * 1024 * 1024,
            max_image_bytes=args.max_image_mib * 1024 * 1024, resume=args.resume, plan=args.plan)
        print(json.dumps({"status": state["status"], "images": len(state["images"]),
            "manifest": None if args.plan else str((args.output_root / "manifest.csv").resolve()),
            "counts_by_source": dict(Counter(row["source"] for row in state["images"])),
            "selected_uncompressed_bytes": state.get("selected_uncompressed_bytes"),
            "transferred_bytes": state["transferred_bytes_this_run"]}))
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile, RuntimeError) as error:
        parser.exit(2, f"ZIP sampling failed: {error}\n")


if __name__ == "__main__":
    main()
