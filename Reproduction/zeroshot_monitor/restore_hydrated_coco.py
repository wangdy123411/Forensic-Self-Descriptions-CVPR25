from __future__ import annotations

from project_paths import PROJECT_ROOT, DATA_ROOT


import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import time
import urllib.request

from collector import process_alive

PROJECT = PROJECT_ROOT
DATA = DATA_ROOT


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def save_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def completed_ids(path):
    raw = Path(path).read_bytes()
    if not raw.endswith(b'\n'):
        raw = raw[:raw.rfind(b'\n') + 1]
    return {r['image_id'] for r in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))) if r['status'] == 'ok'}


def checked_rows(data_root):
    manifest = read_csv(data_root / 'manifests/eval_unique.csv')
    provenance = {r['image_id']: r for r in read_csv(data_root / 'manifests/provenance.csv')}
    rows = []
    for row in manifest:
        if row['source'] != 'COCO2017':
            continue
        if row['label'] != '0' or row['split'] != 'test':
            raise ValueError('Unexpected COCO manifest label/split')
        prov = provenance[row['image_id']]
        if any(row[k] != prov[k] for k in ('relative_path', 'source', 'sha256')):
            raise ValueError('Manifest and provenance disagree')
        if not re.fullmatch(r'real/COCO2017/[a-f0-9]+\.jpg', row['relative_path']):
            raise ValueError('Unexpected recovery destination')
        original = Path(prov['original_relative_path']).name
        if not re.fullmatch(r'\d{12}\.jpg', original):
            raise ValueError('Unexpected original COCO name')
        rows.append({**row, 'original': original, 'bytes': int(prov['file_bytes']),
                     'url': 'http://images.cocodataset.org/val2017/' + original})
    return rows


def stage_one(row, stage_dir):
    target = stage_dir / Path(row['relative_path']).name
    if target.is_file() and target.stat().st_size == row['bytes'] and digest(target) == row['sha256']:
        return {**row, 'status': 'verified', 'reused': True}
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(row['url'], timeout=25) as response:
                blob = response.read(row['bytes'] + 1)
            if len(blob) != row['bytes'] or hashlib.sha256(blob).hexdigest() != row['sha256']:
                raise ValueError('Original download does not match existing manifest bytes/SHA256')
            temporary = target.with_suffix('.download.tmp')
            temporary.write_bytes(blob)
            os.replace(temporary, target)
            return {**row, 'status': 'verified', 'reused': False}
        except Exception as error:
            last_error = f'{type(error).__name__}: {error}'
            if attempt < 2:
                time.sleep(1 + attempt)
    return {**row, 'status': 'error', 'error': last_error}


def stage(args):
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    all_rows = checked_rows(args.data_root)
    successful = completed_ids(args.run_dir / 'predictions.csv')
    selected = [r for r in all_rows if r['image_id'] not in successful]
    plan = {'created_utc': utc_now(), 'data_root': str(args.data_root),
            'manifest_sha256': digest(args.data_root / 'manifests/eval_unique.csv'),
            'selected_count': len(selected), 'source': 'COCO2017', 'rows': selected}
    save_json(args.stage_dir / 'stage_plan.json', plan)
    print(f'Staging {len(selected)} existing selected COCO images; original dataset remains untouched.', flush=True)
    results = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(stage_one, row, args.stage_dir): row for row in selected}
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 25 == 0 or len(results) == len(selected):
                ok = sum(r['status'] == 'verified' for r in results)
                progress = {'updated_utc': utc_now(), 'total': len(selected), 'processed': len(results),
                            'verified': ok, 'errors': len(results) - ok,
                            'elapsed_seconds': time.monotonic() - started, 'original_dataset_changed': False}
                save_json(args.stage_dir / 'stage_progress.json', progress)
                print(json.dumps(progress), flush=True)
    results.sort(key=lambda r: r['relative_path'])
    save_json(args.stage_dir / 'stage_results.json', {'completed_utc': utc_now(), 'rows': results})
    return 0 if all(r['status'] == 'verified' for r in results) else 2


def assert_stopped(run_dir):
    paths = [run_dir / 'progress.json', run_dir.parent / (run_dir.name + '_launcher.json')]
    pids = set()
    for path in paths:
        if path.is_file():
            obj = json.loads(path.read_text(encoding='utf-8-sig'))
            if isinstance(obj.get('pid'), int):
                pids.add(obj['pid'])
    if not pids:
        raise RuntimeError('Cannot establish evaluator process identity')
    for pid in pids:
        if process_alive(pid) is not False:
            raise RuntimeError(f'Evaluator PID {pid} is alive or cannot be verified stopped; no files restored')


def restore(args):
    if not args.confirm_inference_stopped:
        raise RuntimeError('Restoration requires explicit --confirm-inference-stopped after stopping evaluation')
    assert_stopped(args.run_dir)
    plan = json.loads((args.stage_dir / 'stage_plan.json').read_text(encoding='utf-8'))
    if plan['manifest_sha256'] != digest(args.data_root / 'manifests/eval_unique.csv'):
        raise ValueError('Evaluation manifest changed after staging')
    expected = {r['image_id']: r for r in checked_rows(args.data_root)}
    rows = plan['rows']

    for row in rows:
        verified = expected.get(row['image_id'])
        if verified is None or row != verified:
            raise ValueError('Stage plan does not match a selected manifest image')
        staged = args.stage_dir / Path(row['relative_path']).name
        if staged.stat().st_size != row['bytes'] or digest(staged) != row['sha256']:
            raise ValueError(f'Staged bytes invalid: {row["original"]}')
    restored = []
    for index, row in enumerate(rows, start=1):
        assert_stopped(args.run_dir)
        target = (args.data_root / row['relative_path']).resolve()
        if not target.is_relative_to(args.data_root.resolve() / 'real' / 'COCO2017'):
            raise ValueError('Destination escapes the permitted COCO directory')
        if not target.exists():
            raise ValueError('Refusing to create a missing original dataset entry')
        staged = args.stage_dir / target.name
        temporary = target.with_name(target.name + '.verified-recovery.tmp')
        shutil.copyfile(staged, temporary)
        if digest(temporary) != row['sha256']:
            raise ValueError('Temporary replacement bytes failed verification')
        os.replace(temporary, target)
        if digest(target) != row['sha256']:
            raise ValueError('Restored bytes failed verification')
        restored.append({'image_id': row['image_id'], 'relative_path': row['relative_path'], 'sha256': row['sha256']})
        if index % 25 == 0 or index == len(rows):
            save_json(args.stage_dir / 'restore_progress.json', {'updated_utc': utc_now(), 'restored': index, 'total': len(rows)})
            print(f'Restored and verified {index}/{len(rows)}', flush=True)
    save_json(args.stage_dir / 'restore_results.json', {'completed_utc': utc_now(), 'rows': restored, 'byte_identical_to_manifest': True})
    return 0


def main():
    parser = argparse.ArgumentParser(description='restore hydrated coco')
    parser.add_argument('operation', choices=['stage', 'restore'])
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--run-dir', type=Path, default=PROJECT / 'Results/zeroshot_20261106')
    parser.add_argument('--stage-dir', type=Path, default=PROJECT / 'Results/zeroshot_monitor/coco_recovery_20261106')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--confirm-inference-stopped', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be between 1 and 8')
    args.data_root = args.data_root.resolve()
    args.run_dir = args.run_dir.resolve()
    args.stage_dir = args.stage_dir.resolve()
    return stage(args) if args.operation == 'stage' else restore(args)


if __name__ == '__main__':
    raise SystemExit(main())
