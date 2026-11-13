from __future__ import annotations

from project_paths import ARCHIVE_ROOT


import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".avif"}
CHUNK = 1024 * 1024
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError(f"Refusing linked status path: {path}")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def safe_relative(value):

    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"Invalid relative path: {value!r}")
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or ":" in normalized:
        raise ValueError(f"Absolute or drive-qualified path: {value!r}")
    parts = normalized.rstrip("/").split("/")
    if any(p in ("", ".", "..") or p.endswith((" ", ".")) or
           any(c in p for c in '<>"|?*') or any(ord(c) < 32 for c in p) or
           p.split(".")[0].upper() in RESERVED for p in parts):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return PurePosixPath(*parts)


def normalized_resolve(path):

    resolved = Path(path).resolve()
    spelling = str(resolved)
    if os.name == "nt":
        if spelling[:8].upper() == "\\\\?\\UNC\\":
            resolved = Path("\\\\" + spelling[8:])
        elif spelling.startswith("\\\\?\\") and re.match(r"[A-Za-z]:\\", spelling[4:]):
            resolved = Path(spelling[4:])
    return resolved


def contained_path(root, value):
    root = normalized_resolve(root)
    relative = safe_relative(value)
    target = root.joinpath(*relative.parts)
    if not normalized_resolve(target).is_relative_to(root):
        raise ValueError(f"Path escapes root: {value!r}")

    current = target
    while current != root:
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ValueError(f"Refusing linked destination: {current}")
        current = current.parent
    return target


def digests(path):
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    size = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            sha.update(block)
            md5.update(block)
            size += len(block)
    return {"bytes": size, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def verify_file(path, source):
    result = digests(path)
    expected = source.get("expected_bytes")
    if expected is not None and result["bytes"] != expected:
        raise ValueError(f"Size mismatch for {path}: {result['bytes']} != {expected}")
    for algorithm in ("sha256", "md5"):
        if source.get(algorithm) and result[algorithm] != source[algorithm].lower():
            raise ValueError(f"{algorithm.upper()} mismatch for {path}")
    result["verification"] = "source_digest" if source.get("sha256") or source.get("md5") else "local_digest_and_http_length"
    return result


def validate_config(config):
    datasets = config.get("datasets", config.get("sources"))
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Configuration must contain a nonempty 'datasets' list")
    names = set()
    for dataset in datasets:
        name = dataset.get("name")
        if len(safe_relative(name).parts) != 1 or name.casefold() in names:
            raise ValueError(f"Dataset name must be unique and a single directory: {name}")
        names.add(name.casefold())
        safe_relative(dataset.get("extract_subdir", "images"))
        sources = dataset.get("sources", [])
        if not isinstance(sources, list):
            raise ValueError(f"sources must be a list for {name}")
        filenames = set()
        for source in sources:
            filename = safe_relative(source.get("filename"))
            if str(filename).casefold() in filenames:
                raise ValueError(f"Duplicate source filename: {name}/{filename}")
            filenames.add(str(filename).casefold())
            url = source.get("url")
            if url is not None:
                parsed = urllib.parse.urlsplit(url)
                if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
                    raise ValueError(f"Source URL must be public HTTP(S), without embedded credentials: {name}/{filename}")
            expected = source.get("expected_bytes")
            if expected is not None and (not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0):
                raise ValueError("expected_bytes must be a positive integer")
            for algorithm, length in (("sha256", 64), ("md5", 32)):
                value = source.get(algorithm)
                if value and (not isinstance(value, str) or not re.fullmatch(f"[0-9a-fA-F]{{{length}}}", value)):
                    raise ValueError(f"Invalid {algorithm} digest for {name}/{filename}")
    return datasets


def reject_error_response(response, first_block):
    mime = response.headers.get("Content-Type", "").lower().split(";")[0].strip()
    prefix = first_block[:1024].lstrip().lower()
    if mime in ("text/html", "application/xhtml+xml", "application/json", "application/xml", "text/xml") or prefix.startswith((b"<!doctype html", b"<html", b"<?xml", b"<error>")):
        raise ValueError(f"Server returned an error/login/document body ({mime}) rather than archive bytes")
    if response.headers.get("Content-Encoding", "identity").lower() not in ("", "identity"):
        raise ValueError("Encoded HTTP response cannot be used for byte-preserving downloads")


def download_source(source, destination, previous=None, retries=4, timeout=60, opener=None, progress=None):

    destination = Path(destination)
    if source.get("blocked_reason"):
        return {"status": "blocked", "filename": source["filename"], "reason": source["blocked_reason"]}
    if not source.get("url"):
        return {"status": "needs_access", "filename": source["filename"], "reason": source.get("access_note", "No public download URL configured")}
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verified = verify_file(destination, source)
        known = source.get("sha256") or source.get("md5") or (previous or {}).get("sha256")
        if not known:
            raise ValueError(f"Existing file has no trusted digest/history; move it aside or add its authoritative digest to the config: {destination}")
        if not (source.get("sha256") or source.get("md5")) and verified["sha256"] != previous["sha256"]:
            raise ValueError(f"Existing file differs from recorded SHA256: {destination}")
        return {"status": "complete", "action": "verified_existing", "filename": source["filename"], **verified}
    partial = destination.with_name(destination.name + ".part")
    meta_path = destination.with_name(destination.name + ".part.json")
    if any(p.is_symlink() for p in (destination, partial, meta_path)):
        raise ValueError(f"Refusing linked download path: {destination}")
    opener = opener or urllib.request.build_opener()
    last_error = None
    for attempt in range(retries):
        metadata = read_json(meta_path, {})
        stable_identity = metadata.get("etag") or metadata.get("last_modified") or source.get("sha256") or source.get("md5")
        offset = partial.stat().st_size if partial.exists() else 0
        if metadata.get("url") != source["url"] or not stable_identity:
            offset = 0
        headers = {"User-Agent": "FSD-ZeroShot-Dataset/1.0", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            if metadata.get("etag") and not metadata["etag"].startswith("W/"):
                headers["If-Range"] = metadata["etag"]
            elif metadata.get("last_modified"):
                headers["If-Range"] = metadata["last_modified"]
        try:
            request = urllib.request.Request(source["url"], headers=headers)
            with opener.open(request, timeout=timeout) as response:
                status = response.status
                if status not in (200, 206):
                    raise ValueError(f"Unexpected HTTP status {status}")
                total = None
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match:
                        raise ValueError("Missing or malformed Content-Range")
                    start, end, total = map(int, match.groups())
                    if start != offset or end < start or total <= end:
                        raise ValueError(f"Incorrect Content-Range: {response.headers.get('Content-Range')}")
                    response_bytes = end - start + 1
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) != response_bytes:
                        raise ValueError("Content-Length does not match Content-Range")
                else:

                    offset = 0
                    length = response.headers.get("Content-Length")
                    response_bytes = int(length) if length is not None else None
                    total = response_bytes
                if source.get("expected_bytes") is not None and total is not None and total != source["expected_bytes"]:
                    raise ValueError(f"HTTP size {total} != configured size {source['expected_bytes']}")
                first = response.read(CHUNK)
                reject_error_response(response, first)
                if not first:
                    raise ValueError("Empty archive response")
                if offset and metadata.get("etag") and response.headers.get("ETag") and metadata["etag"] != response.headers.get("ETag"):
                    raise ValueError("ETag changed while resuming")
                metadata = {"url": source["url"], "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified"), "total_bytes": total}
                atomic_json(meta_path, metadata)
                received = 0
                last_report = time.monotonic()
                with partial.open("ab" if offset else "wb") as target:
                    block = first
                    while block:
                        target.write(block)
                        received += len(block)
                        if response_bytes is not None and received > response_bytes:
                            raise ValueError("HTTP body exceeds declared size")
                        if progress and time.monotonic() - last_report >= 10:
                            progress({"status": "downloading", "bytes": offset + received, "total_bytes": total, "attempt": attempt + 1})
                            last_report = time.monotonic()
                        block = response.read(CHUNK)
                if response_bytes is not None and received != response_bytes:
                    raise ValueError(f"Truncated response: {received}/{response_bytes}")
                if total is not None and partial.stat().st_size != total:

                    raise ValueError(f"Incomplete object: {partial.stat().st_size}/{total}")
            verified = verify_file(partial, source)
            os.replace(partial, destination)
            meta_path.unlink(missing_ok=True)
            return {"status": "complete", "action": "downloaded", "filename": source["filename"], **verified}
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 416:

                if partial.exists() and (source.get("sha256") or source.get("md5")):
                    try:
                        verified = verify_file(partial, source)
                        os.replace(partial, destination)
                        meta_path.unlink(missing_ok=True)
                        return {"status": "complete", "action": "verified_complete_partial", "filename": source["filename"], **verified}
                    except ValueError:
                        pass
                meta_path.unlink(missing_ok=True)
            elif "mismatch" in str(exc).lower() or "etag changed" in str(exc).lower():
                meta_path.unlink(missing_ok=True)
            if progress:
                progress({"status": "retrying", "attempt": attempt + 1, "error": last_error, "bytes": partial.stat().st_size if partial.exists() else 0})
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Download failed after {retries} attempts: {last_error}")


def archive_members(archive):

    entries = []
    seen = set()
    if isinstance(archive, zipfile.ZipFile):
        members = archive.infolist()
        for member in members:
            path = safe_relative(member.filename)
            mode = (member.external_attr >> 16) & 0o170000
            if mode == 0o120000 or mode not in (0, 0o100000, 0o040000):
                raise ValueError(f"Archive link or special file: {member.filename}")
            if member.is_dir():
                continue
            key = str(path).casefold()
            if key in seen:
                raise ValueError(f"Duplicate archive path: {member.filename}")
            seen.add(key)
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                entries.append((str(path), member, member.file_size))
    else:
        for member in archive.getmembers():

            name = member.name
            while name.startswith("./"):
                name = name[2:]
            if not name and member.isdir():
                continue
            path = safe_relative(name)
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError(f"Archive link or special file: {member.name}")
            if member.isdir():
                continue
            key = str(path).casefold()
            if key in seen:
                raise ValueError(f"Duplicate archive path: {member.name}")
            seen.add(key)
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                entries.append((str(path), member, member.size))
    return sorted(entries, key=lambda entry: entry[0])


def extract_images(archive_path, destination, max_images=None, max_member_bytes=1024 ** 3):

    archive_path, destination = Path(archive_path), Path(destination)
    opener = zipfile.ZipFile if zipfile.is_zipfile(archive_path) else tarfile.open
    records = []
    with opener(archive_path) as archive:
        entries = archive_members(archive)
        selected = entries if max_images is None else entries[:max_images]
        for name, member, size in selected:
            if size <= 0 or size > max_member_bytes:
                raise ValueError(f"Image member size outside safety limit: {name} ({size})")
            target = contained_path(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.open(member) if isinstance(archive, zipfile.ZipFile) else archive.extractfile(member)
            temporary = target.with_name(target.name + ".extracting")
            if temporary.is_symlink():
                raise ValueError(f"Linked temporary extraction path: {temporary}")
            digest = hashlib.sha256()
            written = 0
            try:
                with source, temporary.open("wb") as stream:
                    for block in iter(lambda: source.read(CHUNK), b""):
                        written += len(block)
                        if written > size:
                            raise ValueError(f"Member exceeds declared size: {name}")
                        digest.update(block)
                        stream.write(block)
                if written != size:
                    raise ValueError(f"Truncated archive member: {name}")
                sha256 = digest.hexdigest()
                if target.exists():
                    if digests(target)["sha256"] != sha256:
                        raise ValueError(f"Refusing to replace different existing image: {target}")
                    temporary.unlink()
                else:
                    os.replace(temporary, target)
                records.append({"member": name, "relative_path": target.relative_to(destination).as_posix(), "bytes": written, "sha256": sha256})
            finally:
                if temporary.exists() and not temporary.is_symlink():
                    temporary.unlink()
    return {"available_images": len(entries), "selected_images": len(records), "selection": "Lexicographic original archive member order; convenience subset, not authors' exact split", "images": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description='download zeroshot datasets', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--only", nargs="+", help="Exact dataset names to process")
    parser.add_argument("--plan", action="store_true", help="Validate and display plan without network or filesystem writes")
    parser.add_argument("--extract", action="store_true", help="Extract image members after verifying archives")
    parser.add_argument("--max-images", type=int, help="Extract at most N images per dataset, in source filename/member order")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent archive downloads per dataset (extraction stays sequential)")
    parser.add_argument("--no-proxy", action="store_true", help="Use direct connections instead of environment proxy settings")
    args = parser.parse_args(argv)
    if args.retries < 1 or args.timeout < 1 or args.workers < 1 or (args.max_images is not None and args.max_images < 1):
        parser.error("retries, timeout, workers, and max-images must be positive")
    datasets = validate_config(json.loads(args.config.read_text(encoding="utf-8-sig")))
    if args.only:
        missing = set(args.only) - {d["name"] for d in datasets}
        if missing:
            parser.error(f"Unknown dataset names: {sorted(missing)}")
        datasets = [d for d in datasets if d["name"] in args.only]
    root = args.root.expanduser().resolve()
    plan = {"root": str(root), "datasets": datasets, "extract": args.extract, "max_images": args.max_images,
            "known_download_bytes": sum(s.get("expected_bytes") or 0 for d in datasets for s in d.get("sources", [])),
            "selection_note": "Original image bytes. A limited extraction is not the authors' exact evaluation split."}
    if args.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    status_path = contained_path(root, "download_status.json")
    state = read_json(status_path, {"datasets": {}})
    state.update({"root": str(root), "config": str(args.config.resolve()), "updated_at_utc": utc_now()})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if args.no_proxy else urllib.request.build_opener()
    failures = 0
    needs_access = 0
    blocked = 0
    state_lock = threading.Lock()
    for dataset in datasets:
        name = dataset["name"]
        dataset_root = contained_path(root, name)
        previous = state["datasets"].get(name, {}).get("sources", {})
        record = {"status": "running", "sources": {}, "extractions": {}, "max_images": args.max_images}
        state["datasets"][name] = record
        extracted = 0
        source_list = sorted(dataset.get("sources", []), key=lambda s: s["filename"])
        if dataset.get("blocked_reason"):
            record.update(status="blocked", reason=dataset["blocked_reason"])
            blocked += 1
            atomic_json(status_path, state)
            print(f"{name}: BLOCKED {dataset['blocked_reason']}", flush=True)
            continue
        if not source_list:
            record.update(status="needs_access", reason=dataset.get("access_note", "No sources configured"))
            needs_access += 1
        def transfer(source):
            filename = source["filename"]
            target = contained_path(dataset_root, filename)
            def progress(update):
                with state_lock:
                    record["sources"][filename] = update
                    state["updated_at_utc"] = utc_now()
                    atomic_json(status_path, state)
                print(f"{name}/{filename}: {json.dumps(update, ensure_ascii=False)}", flush=True)
            try:
                result = download_source(source, target, previous.get(filename), args.retries, args.timeout, opener, progress)
                print(f"{name}/{filename}: {result['status']}", flush=True)
            except Exception as exc:
                result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                print(f"{name}/{filename}: ERROR {exc}", file=sys.stderr, flush=True)
            with state_lock:
                record["sources"][filename] = result
                state["updated_at_utc"] = utc_now()
                atomic_json(status_path, state)
            return result
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(transfer, source) for source in source_list]
            for future in as_completed(futures):
                result = future.result()
                failures += int(result["status"] == "error")
                needs_access += int(result["status"] == "needs_access")
                blocked += int(result["status"] == "blocked")
        for source in source_list:
            filename = source["filename"]
            if record["sources"][filename]["status"] != "complete" or not args.extract or (args.max_images is not None and extracted >= args.max_images):
                continue
            try:
                remaining = None if args.max_images is None else args.max_images - extracted
                destination = contained_path(dataset_root, dataset.get("extract_subdir", "images"))
                extraction = extract_images(contained_path(dataset_root, filename), destination, remaining)
                record["extractions"][filename] = extraction
                extracted += extraction["selected_images"]
            except Exception as exc:
                record["extractions"][filename] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                failures += 1
                print(f"{name}/{filename}: EXTRACTION ERROR {exc}", file=sys.stderr, flush=True)
            atomic_json(status_path, state)
        statuses = [v["status"] for v in record["sources"].values()]
        extraction_failed = any(v.get("status") == "error" for v in record["extractions"].values())
        record["status"] = "error" if "error" in statuses or extraction_failed else "blocked" if "blocked" in statuses else "needs_access" if not statuses or "needs_access" in statuses else "complete"
        record["extracted_images"] = extracted
        if args.extract and args.max_images and extracted < args.max_images and record["status"] == "complete":
            record["extraction_note"] = f"Only {extracted} images available; requested limit {args.max_images}"
        atomic_json(status_path, state)
    print(json.dumps({"status_file": str(status_path), "failed_sources": failures, "needs_access": needs_access, "blocked": blocked}, ensure_ascii=False), flush=True)
    return 1 if failures else 2 if needs_access or blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
