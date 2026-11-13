from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import http.client
import json
from pathlib import Path
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib

from sample_remote_zip import TransferBudget, select_members
from zeroshot_common import MANIFEST_FIELDS, child_path, file_hash, image_id, write_csv, write_json


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_spans(selected, archives, max_span=32 * 1024**2, max_gap=256 * 1024):
    groups = []
    for aid, archive in archives.items():
        entries = []
        for row in selected:
            if row['archive_id'] != aid:
                continue
            info = archive.getinfo(row['member'])
            if info.compress_type not in (0, 8) or info.flag_bits & 1:
                raise ValueError('Only unencrypted stored/deflated members are supported')
            entry = dict(row, header_offset=info.header_offset, compressed_size=info.compress_size,
                         compression=info.compress_type, flag_bits=info.flag_bits)

            end = min(row['archive_size'], info.header_offset + 30 +
                      len(info.filename.encode('utf-8' if info.flag_bits & 2048 else 'cp437')) +
                      65535 + info.compress_size)
            entries.append((info.header_offset, end, entry))
        active = None
        for start, end, row in sorted(entries, key=lambda x: x[0]):
            if (active and start <= active['end'] + max_gap and
                    max(end, active['end']) - active['start'] <= max_span and
                    active['decoded_bytes'] + row['member_size'] <= 128 * 1024**2):
                active['end'] = max(end, active['end'])
                active['rows'].append(row)
                active['decoded_bytes'] += row['member_size']
            else:
                active = dict(url=row['archive_url'], size=row['archive_size'], start=start, end=end,
                              rows=[row], decoded_bytes=row['member_size'])
                if end - start > max_span + 64 * 1024**2:
                    raise ValueError('Member span exceeds bounded request limit')
                groups.append(active)
    return groups


def decode_member(block, span_start, row):
    offset = row['header_offset'] - span_start
    header = block[offset:offset + 30]
    if len(header) != 30:
        raise ValueError('Truncated local ZIP header')
    signature, version, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack('<4s5H3I2H', header)
    if signature != b'PK\x03\x04' or flags & 1 or method != row['compression']:
        raise ValueError('Invalid or inconsistent ZIP local header')
    name_start = offset + 30
    name = block[name_start:name_start + name_len].decode('utf-8' if flags & 2048 else 'cp437')
    if name != row['member']:
        raise ValueError('Local ZIP filename disagrees with directory')
    start = name_start + name_len + extra_len
    compressed = block[start:start + row['compressed_size']]
    if len(compressed) != row['compressed_size']:
        raise ValueError('Truncated compressed member')
    if method == 0:
        data = compressed
    else:
        decoder = zlib.decompressobj(-15)
        data = decoder.decompress(compressed, row['member_size'] + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('Invalid or oversized deflate stream')
    if len(data) != row['member_size'] or f'{zlib.crc32(data):08x}' != row['member_crc32']:
        raise ValueError('ZIP member size/CRC mismatch')
    return data


class Fetcher:
    def __init__(self, limit, timeout=90):
        self.limit, self.timeout, self.used = limit, timeout, 0
        self.lock = threading.Lock()

    def fetch(self, span):
        count = span['end'] - span['start']
        for attempt in range(5):
            with self.lock:
                if self.used + count > self.limit:
                    raise ValueError('Transfer budget exceeded')
                self.used += count
            request = urllib.request.Request(span['url'], headers={
                'Range': f"bytes={span['start']}-{span['end'] - 1}",
                'Accept-Encoding': 'identity', 'User-Agent': 'FSD-original-sample/1.0'})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    expected = f"bytes {span['start']}-{span['end'] - 1}/{span['size']}"
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError('Server did not honor exact bounded HTTP Range')
                    if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                        raise ValueError('Unexpected response encoding')
                    block = response.read(count + 1)
                    if len(block) != count:
                        raise OSError('Truncated byte range')
                return [(row, decode_member(block, span['start'], row)) for row in span['rows']]
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 4:
                    raise
            except (OSError, TimeoutError, http.client.IncompleteRead):
                if attempt == 4:
                    raise
            time.sleep(min(2 ** attempt, 8))


def run(config, root, *, count=1000, seed=2026, workers=6, max_transfer=32 * 1024**3,
        span_mib=32, plan=False, resume=False):
    if count < 1 or workers < 1 or span_mib < 1 or max_transfer < 1:
        raise ValueError('Counts and limits must be positive')
    root = Path(root).resolve()
    state_path, manifest_path = root / 'sample_state.json', root / 'manifest.csv'
    fingerprint = sha(json.dumps({'config': config, 'per_source': count, 'seed': seed}, sort_keys=True).encode())
    previous = {}
    if state_path.exists() and not plan:
        if not resume:
            raise ValueError('Sampling state exists; use --resume')
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if state['config_sha256'] != fingerprint:
            raise ValueError('Sampling config/seed changed')
        previous = {r['relative_path']: r for r in state['images']}
    elif manifest_path.exists() and not plan:
        raise ValueError('Untracked manifest exists')
    budget = TransferBudget(max_transfer)
    archives, selected = select_members(config, count, seed, budget, max_image_bytes=64 * 1024**2)
    try:
        if state_path.exists() and resume and not plan and state['selection'] != selected:
            raise ValueError('Remote archive selection changed')
        selected_map = {r['relative_path']: r for r in selected}
        if set(previous) - set(selected_map):
            raise ValueError('Unexpected prior image records')
        todo, completed = [], dict(previous)
        for row in selected:
            path = child_path(root, row['relative_path'])
            old = previous.get(row['relative_path'])
            if plan:
                todo.append(row)
                continue
            if old and any(old.get(k) != v for k, v in row.items()):
                raise ValueError('Prior image provenance changed')
            if old and (not path.is_file() or file_hash(path) != old['sha256']):
                raise ValueError(f'Previously downloaded image changed: {path}')
            if old:
                continue
            if path.exists():
                if path.stat().st_size != row['member_size']:
                    raise ValueError(f'Untracked image has different size: {path}')
                data = path.read_bytes()
                if len(data) != row['member_size'] or f'{zlib.crc32(data):08x}' != row['member_crc32']:
                    raise ValueError(f'Untracked image has different bytes: {path}')
                digest = sha(data)
                completed[row['relative_path']] = dict(row, sha256=digest,
                    image_id=image_id(row['source'], row['relative_path'], digest))
            else:
                todo.append(row)
        spans = make_spans(todo, archives, span_mib * 1024**2)
        planned = sum(s['end'] - s['start'] for s in spans)
        overview = dict(selected_images=len(selected), pending_images=len(todo), spans=len(spans),
                        requested_bytes=planned, counts=dict(Counter(r['source'] for r in selected)))
        print(json.dumps(overview), flush=True)
        if plan:
            return overview
        if planned + budget.used > max_transfer:
            raise ValueError('Planned reads exceed transfer budget')
        root.mkdir(parents=True, exist_ok=True)
        state = dict(config_sha256=fingerprint, seed=seed, status='partial', exact_paper_sample_list=False,
                     byte_preserving=True, selection=selected, images=[], overview=overview)
        fetcher = Fetcher(max_transfer - budget.used)
        started = time.monotonic()

        def checkpoint():
            state['images'] = [completed[r['relative_path']] for r in selected if r['relative_path'] in completed]
            state['reserved_transfer_bytes_this_run'] = fetcher.used + budget.used
            write_json(state_path, state)
            write_csv(manifest_path, MANIFEST_FIELDS, [{k: r[k] for k in MANIFEST_FIELDS} for r in state['images']])

        checkpoint()
        pool = ThreadPoolExecutor(max_workers=workers)
        remaining = iter(spans)
        futures = {pool.submit(fetcher.fetch, span) for _, span in zip(range(workers), remaining)}
        try:
            while futures:
                ready, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in ready:
                    futures.remove(future)
                    for raw_row, data in future.result():
                        row = selected_map[raw_row['relative_path']]
                        path = child_path(root, row['relative_path'])
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = path.with_suffix(path.suffix + '.part')
                        if temporary.is_symlink():
                            raise ValueError('Refusing symlink temporary file')
                        temporary.write_bytes(data)
                        temporary.replace(path)
                        digest = sha(data)
                        completed[row['relative_path']] = dict(row, sha256=digest,
                            image_id=image_id(row['source'], row['relative_path'], digest))
                    checkpoint()
                    print(json.dumps(dict(done=len(completed), total=len(selected),
                        elapsed_s=round(time.monotonic() - started),
                        reserved_bytes=fetcher.used)), flush=True)
                    span = next(remaining, None)
                    if span is not None:
                        futures.add(pool.submit(fetcher.fetch, span))
        finally:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            checkpoint()
        state['status'] = 'complete'
        checkpoint()
        return state
    finally:
        for archive in archives.values():
            archive.close()


def main():
    parser = argparse.ArgumentParser(description='sample remote zip fast')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--per-source', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--span-mib', type=int, default=32)
    parser.add_argument('--max-transfer-mib', type=int, default=32768)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-proxy', action='store_true')
    args = parser.parse_args()
    if args.no_proxy:
        urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    run(json.loads(args.config.read_text(encoding='utf-8-sig')), args.output_root,
        count=args.per_source, seed=args.seed, workers=args.workers,
        max_transfer=args.max_transfer_mib * 1024**2, span_mib=args.span_mib,
        plan=args.plan, resume=args.resume)


if __name__ == '__main__':
    main()
