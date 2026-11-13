from __future__ import annotations

from project_paths import ARCHIVE_ROOT


import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import http.client
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request

from download_zeroshot_datasets import contained_path, normalized_resolve
from zeroshot_common import MANIFEST_FIELDS, file_hash, image_id, write_csv, write_json

REPOSITORY = "ductai199x/open-set-synth-img-attribution"
TARGETS = {"stylegan": "StyleGAN", "projected_gan": "Proj.GAN", "tam_trans": "Tam.Xformer", "stable_diffusion": "SD1.4"}
SOURCE_MAPPING_NOTE = "OSSIA stable_diffusion mapped to SD1.4 based on FSD paper Table 9 source attribution (OSSIA/SB), not inferred from an unspecified architecture name."


class TransferLimit(OSError):
    pass


class Budget:
    def __init__(self, limit):
        self.limit, self.used, self.reserved = limit, 0, 0
        self.lock = threading.Lock()

    def reserve(self, count):


        with self.lock:
            if self.reserved + count > self.limit:
                raise TransferLimit(f"Network budget reached: {self.reserved}/{self.limit} reserved bytes; retained samples remain usable")
            self.reserved += count

    def account(self, count):
        with self.lock:
            self.used += count


class BoundedHTTPReader(io.RawIOBase):

    def __init__(self, url, size, budget, *, chunk_bytes=8 * 1024 * 1024, workers=4, opener=None, timeout=60, retries=4, notify=None):
        self.url, self.size, self.budget = url, size, budget
        self.chunk_bytes, self.opener = chunk_bytes, opener or urllib.request.build_opener()
        self.timeout, self.retries, self.notify = timeout, retries, notify
        self.position = 0
        self.buffer, self.buffer_pos = b"", 0
        self.workers = workers
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.pending = {}
        self.next_offset = 0

    def readable(self):
        return True

    def tell(self):
        return self.position

    def _fetch(self, start):
        count = min(self.chunk_bytes, self.size - start)
        for attempt in range(self.retries):
            self.budget.reserve(count + 1)
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + count - 1}", "Accept-Encoding": "identity", "User-Agent": "FSD-OSSIA-prefix-sampler/1.0"})
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    expected = f"bytes {start}-{start + count - 1}/{self.size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor exact bounded Range; refusing full response")
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise ValueError("Unexpected HTTP encoding")
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) != count:
                        raise ValueError("Invalid HTTP range length")
                    data = bytearray()
                    try:
                        while len(data) <= count:
                            block = response.read(min(1024 * 1024, count + 1 - len(data)))
                            if not block:
                                break
                            self.budget.account(len(block))
                            data.extend(block)
                    except http.client.IncompleteRead as error:
                        self.budget.account(len(error.partial))
                        raise OSError("Interrupted HTTP range") from error
                    if len(data) != count:
                        raise OSError("Incomplete HTTP range")
                return bytes(data)
            except urllib.error.HTTPError as error:
                if error.code not in (408, 429, 500, 502, 503, 504) or attempt + 1 == self.retries:
                    raise
            except (OSError, http.client.HTTPException):
                if attempt + 1 == self.retries:
                    raise
            time.sleep(min(2 ** attempt, 8))
        raise OSError("Range retries exhausted")

    def _schedule(self):
        while len(self.pending) < self.workers and self.next_offset < self.size:
            start = self.next_offset
            self.pending[start] = self.pool.submit(self._fetch, start)
            self.next_offset += min(self.chunk_bytes, self.size - start)

    def _fill(self):
        if self.position >= self.size:
            return False
        self._schedule()
        data = self.pending.pop(self.position).result()
        self.buffer, self.buffer_pos = data, 0
        self._schedule()
        if self.notify:
            self.notify(self.position + len(data), self.budget.used)
        return True

    def close(self):
        if not self.closed:
            self.pool.shutdown(wait=True, cancel_futures=True)
        super().close()

    def read(self, count=-1):
        if count < 0:
            raise ValueError("Unbounded HTTP reads are forbidden")
        output = bytearray()
        while len(output) < count:
            if self.buffer_pos == len(self.buffer) and not self._fill():
                break
            take = min(count - len(output), len(self.buffer) - self.buffer_pos)
            output.extend(self.buffer[self.buffer_pos:self.buffer_pos + take])
            self.buffer_pos += take
            self.position += take
        return bytes(output)


def read_test_index(path):
    selected = {}
    counts = Counter()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"arch", "dataset", "compression", "width", "height", "path"}
        if set(reader.fieldnames or []) != required:
            raise ValueError("Unexpected author test.csv schema")
        for row in reader:
            arch = row["arch"]
            if arch not in TARGETS:
                continue
            parts = row["path"].split("/")


            if len(parts) != 4 or parts[0] != arch or not parts[1] or parts[2] != "uncompressed" or row["compression"] != "":
                raise ValueError(f"Unexpected target test-index provenance: {row['path']}")
            if row["path"] in selected:
                raise ValueError(f"Duplicate author test-index path: {row['path']}")
            selected[row["path"]] = row
            counts[arch] += 1
    if set(counts) != set(TARGETS):
        raise ValueError("Author test index does not cover all requested architectures")
    return selected, counts


def make_archives(tree, revision):
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("Pin an exact 40-character repository revision")
    archives = []
    for entry in tree:
        match = re.fullmatch(r"data/test\.(\d+)\.tar\.gz", entry.get("path", ""))
        if match:
            archives.append({"shard": int(match[1]), "path": entry["path"], "size": entry["size"], "sha256": entry.get("lfs", {}).get("oid"), "url": f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{entry['path']}"})
    if not archives:
        raise ValueError("No official test shards in supplied tree metadata")
    return sorted(archives, key=lambda x: x["shard"])


def retry_checkpoint(writer, *args):

    for attempt in range(8):
        try:
            return writer(*args)
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 1.0))


def run(index_path, archives, revision, output, data_root, *, per_source=1000, max_transfer=8 * 1024 ** 3, resume=False, extend_targets=False, allow_code_update=False, start_shard=0, chunk_bytes=8 * 1024 * 1024, workers=4, opener=None):
    if per_source < 1 or max_transfer < 1 or chunk_bytes < 1 or workers < 1:
        raise ValueError("Counts and byte limits must be positive")
    data_root = normalized_resolve(data_root)
    output = normalized_resolve(output)
    if not output.is_relative_to(data_root):
        raise ValueError("Output root must be inside data root")
    author_rows, available = read_test_index(index_path)
    if any(n < per_source for n in available.values()):
        raise ValueError("Requested count exceeds author test-index population")
    identity = {"repository": REPOSITORY, "revision": revision, "test_index_sha256": file_hash(Path(index_path)), "targets": TARGETS, "per_source": per_source, "archives": archives, "selection": "First unique indexed uncompressed test images encountered in explicitly recorded author-shuffled test-shard prefixes; interrupted runs may continue on a later shard", "script_sha256": file_hash(Path(__file__)), "data_root": str(data_root)}
    state_path, manifest_path = output / "sample_state.json", output / "manifest.csv"
    state = None
    if state_path.exists():
        if not resume:
            raise ValueError("Existing sampling state; use --resume")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["identity"] != identity:
            old = state["identity"]
            invariant_keys = set(identity) - {"targets", "script_sha256"}
            compatible_extension = (extend_targets and set(old["targets"]) < set(TARGETS)
                                    and all(TARGETS[k] == v for k, v in old["targets"].items())
                                    and all(old.get(k) == identity[k] for k in invariant_keys))
            compatible_code_update = allow_code_update and all(old.get(k) == identity[k] for k in set(identity) - {"script_sha256", "selection"})
            if not (compatible_extension or compatible_code_update):
                raise ValueError("Resume identity changed; use a new output directory")
            state.setdefault("identity_history", []).append(old)
            state["identity"] = identity
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output directory contains untracked files")
    output.mkdir(parents=True, exist_ok=True)
    if state is None:
        state = {"identity": identity, "status": "running", "images": [], "transferred_bytes_previous_runs": 0,
                 "exact_paper_split": False, "uniform_random_sample": False,
                 "integrity_note": "Original tar member bytes, complete-member length, author test index and local SHA256 verified. Whole compressed archive SHA256/gzip trailer cannot be verified when stopping at a prefix."}
    else:
        state["transferred_bytes_previous_runs"] = state.get("transferred_bytes_previous_runs", 0) + state.get("transferred_bytes_this_run", 0)
    records = {r["member"]: r for r in state["images"]}
    hashes = {source: set() for source in TARGETS.values()}
    counts = Counter()
    for member, record in records.items():
        expected = author_rows.get(member)
        if not expected or TARGETS[expected["arch"]] != record["source"]:
            raise ValueError("Resume provenance differs from author index")
        target = contained_path(data_root, record["relative_path"])
        if target.stat().st_size != record["bytes"] or file_hash(target) != record["sha256"]:
            raise ValueError(f"Resume original image bytes changed: {target}")
        if record["sha256"] in hashes[record["source"]]:
            raise ValueError("Resume duplicate image content")
        counts[record["source"]] += 1
        hashes[record["source"]].add(record["sha256"])
    budget = Budget(max_transfer)
    state["source_mapping_note"] = SOURCE_MAPPING_NOTE
    state.setdefault("stream_runs", []).append({"start_shard": start_shard, "started_at_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": identity["script_sha256"], "max_transfer_bytes": max_transfer})
    started = time.monotonic()
    last_report = 0.0
    scanned = 0

    def save(status, error=None):
        state.update(status=status, images=list(records.values()), counts={s: counts[s] for s in TARGETS.values()}, scanned_members_this_run=scanned,
                     transferred_bytes_this_run=budget.used, reserved_bytes_this_run=budget.reserved, elapsed_seconds_this_run=time.monotonic()-started,
                     updated_at_utc=datetime.now(timezone.utc).isoformat())
        if error:
            state["error"] = error
        else:
            state.pop("error", None)
        retry_checkpoint(write_json, state_path, state)
        rows = [{key: row[key] for key in MANIFEST_FIELDS} for row in records.values()]
        if rows:
            retry_checkpoint(write_csv, manifest_path, MANIFEST_FIELDS, rows)

    def notify(offset, transferred):
        nonlocal last_report
        if time.monotonic() - last_report >= 15:
            save("running")
            print(json.dumps({"counts": dict(counts), "scanned": scanned, "archive_prefix_bytes": offset, "network_bytes": transferred, "elapsed_seconds": round(time.monotonic()-started, 1)}, ensure_ascii=False), flush=True)
            last_report = time.monotonic()

    try:
        save("running")
        for spec in archives:
            if spec["shard"] < start_shard:
                continue
            if all(counts[s] >= per_source for s in TARGETS.values()):
                break
            state["active_archive"] = spec["path"]
            remote = BoundedHTTPReader(spec["url"], spec["size"], budget, chunk_bytes=chunk_bytes, workers=workers, opener=opener, notify=notify)
            with remote, tarfile.open(fileobj=remote, mode="r|gz") as archive:
                for member in archive:
                    scanned += 1
                    name = member.name
                    arch = name.split("/", 1)[0]
                    if arch not in TARGETS or counts[TARGETS[arch]] >= per_source:
                        continue
                    indexed = author_rows.get(name)
                    if indexed is None or indexed["arch"] != arch:
                        raise ValueError(f"Archive target member absent/mismatched in author test.csv: {name}")
                    if not member.isfile() or member.issym() or member.islnk() or member.size <= 0 or member.size > 64 * 1024 * 1024:
                        raise ValueError(f"Unsafe or excessive target member: {name}")
                    if name in records:
                        continue
                    source = TARGETS[arch]
                    relative_local = source + "/" + name.split("/", 1)[1]
                    destination = contained_path(output, relative_local)
                    relative_data = destination.relative_to(data_root).as_posix()
                    with archive.extractfile(member) as image_stream:
                        data = image_stream.read(member.size + 1)
                    if len(data) != member.size:
                        raise ValueError(f"Truncated original member: {name}")
                    if not (data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff")):
                        raise ValueError(f"Unexpected target image signature: {name}")
                    sha = hashlib.sha256(data).hexdigest()
                    if sha in hashes[source]:
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        if file_hash(destination) != sha:
                            raise ValueError(f"Untracked image conflicts with upstream bytes: {destination}")
                    else:
                        temporary = destination.with_name(destination.name + ".part")
                        if temporary.is_symlink():
                            raise ValueError("Linked temporary image")
                        temporary.write_bytes(data)
                        temporary.replace(destination)
                    record = {"image_id": image_id(source, relative_data, sha), "relative_path": relative_data,
                              "label": "1", "source": source, "split": "test", "sha256": sha, "bytes": len(data),
                              "member": name, "archive": spec["path"], "archive_url": spec["url"], "archive_prefix_bytes_consumed": remote.position,
                              "author_arch": arch, "author_dataset": indexed["dataset"], "author_compression": indexed["compression"],
                              "author_width": int(indexed["width"]), "author_height": int(indexed["height"])}
                    records[name] = record
                    hashes[source].add(sha)
                    counts[source] += 1

                    save("running")
                    if all(counts[s] >= per_source for s in TARGETS.values()):
                        break
        complete = all(counts[s] == per_source for s in TARGETS.values())
        save("complete" if complete else "partial", None if complete else "Available test shards exhausted before target counts")
        print(json.dumps({"status": state["status"], "counts": dict(counts), "network_bytes": budget.used, "state": str(state_path)}, ensure_ascii=False), flush=True)
        return 0 if complete else 2
    except BaseException as error:
        save("partial", f"{type(error).__name__}: {error}")
        print(json.dumps({"status": "partial", "counts": dict(counts), "network_bytes": budget.used, "error": str(error)}, ensure_ascii=False), flush=True)
        if isinstance(error, KeyboardInterrupt):
            return 130
        raise


def main():
    parser = argparse.ArgumentParser(description='sample ossia tar')
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--output-root", type=Path, default=ARCHIVE_ROOT / 'OSSIA/samples')
    parser.add_argument("--per-source", type=int, default=1000)
    parser.add_argument("--max-transfer-gib", type=float, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extend-targets", action="store_true", help="Explicitly allow adding target architectures while retaining the same source revision/index/counts and previous provenance")
    parser.add_argument("--allow-code-update", action="store_true", help="Explicitly accept a reviewed sampler-code update only; all data/selection identity fields must remain identical and history is retained")
    parser.add_argument("--workers", type=int, default=4, help="Bounded HTTP Range prefetch workers; gzip consumption stays sequential")
    parser.add_argument("--start-shard", type=int, default=0, help="Explicit test shard to start this prefix run; retained samples and prior provenance remain unchanged")
    parser.add_argument("--no-proxy", action="store_true")
    args = parser.parse_args()
    archives = make_archives(json.loads(args.tree.read_text(encoding="utf-8")), args.revision)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if args.no_proxy else None
    try:
        return run(args.test_index, archives, args.revision, args.output_root, args.data_root, per_source=args.per_source,
                   max_transfer=int(args.max_transfer_gib * 1024 ** 3), resume=args.resume, extend_targets=args.extend_targets,
                   allow_code_update=args.allow_code_update, workers=args.workers, start_shard=args.start_shard, opener=opener)
    except (ValueError, OSError, tarfile.TarError) as error:
        parser.exit(2, f"OSSIA sampling stopped: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
