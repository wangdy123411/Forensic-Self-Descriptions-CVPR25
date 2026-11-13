from __future__ import annotations

from project_paths import PROJECT_ROOT, CODE_ROOT, EXTENSION_ROOT, DATA_ROOT, REFERENCE_PATH, WEIGHTS_ROOT


import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
CODE = CODE_ROOT
PROJECT = PROJECT_ROOT
sys.path.insert(0, str(CODE))
from evaluate_zeroshot import read_previous_predictions, select_rows, weights_identity
from zeroshot_common import (
    PREDICTION_FIELDS, check_resume_metadata, child_path, file_hash,
    load_manifest, summarize, write_csv, write_json,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def snapshot(output, predictions, manifest, selected, identity, started, status, cache=None):
    report = summarize(predictions, manifest, limited=identity['limit_per_source'] is not None)
    successful = [p for p in predictions if p['status'] == 'ok']
    seconds = sum(float(p['elapsed_seconds']) for p in successful)
    average = seconds / len(successful) if successful else None
    progress = {
        'status': status, 'updated_utc': utc_now(), 'pid': os.getpid(),
        'selected_count': len(selected), 'completed_count': len(predictions),
        'ok_count': len(successful), 'error_count': len(predictions) - len(successful),
        'remaining_count': len(selected) - len(predictions),
        'mean_seconds_per_image': average,
        'estimated_remaining_seconds': average * (len(selected) - len(predictions)) if average else None,
        'session_elapsed_seconds': time.perf_counter() - started,
        'completed_by_source': dict(Counter(p['source'] for p in predictions)),
        'expected_by_source': dict(Counter(p['source'] for p in selected)),
        'last_image': predictions[-1]['relative_path'] if predictions else None,
        'last_error': next((p['error'] for p in reversed(predictions) if p['status'] == 'error'), None),
        'device': identity['device'], 'gpu': identity['gpu'],
    }
    if cache is not None:
        coverage = cache.coverage(selected, verify_payloads=status in ('completed', 'completed_with_errors'))
        if status in ('completed', 'completed_with_errors'):
            coverage['database_integrity_verified'] = cache.integrity_check()
        coverage.pop('missing_image_ids')
        coverage.update(updated_utc=utc_now(), path=str(cache.path), phase='online')
        progress['feature_cache'] = coverage
        write_json(output / 'feature_cache' / 'status.json', coverage)
    report.update(run_status=status, is_final=status in ('completed', 'completed_with_errors'),
                  pending_count=len(selected) - len(predictions), threshold=identity['threshold'],
                  manifest_sha256=identity['manifest_sha256'], updated_utc=utc_now())
    write_json(output / 'progress.json', progress)
    write_json(output / 'summary.json', report)
    if report['pairs']:
        write_csv(output / 'pairwise_metrics.csv', tuple(report['pairs'][0]), report['pairs'])
    if not predictions or len(predictions) % 250 == 0 or status != 'running':
        try:
            from export_tables import export_run
            export_run(output, Path(identity['manifest_path']))
            write_json(output / 'table_export_status.json', {'status': 'ok', 'updated_utc': utc_now(), 'prediction_count': len(predictions)})
        except Exception as error:
            write_json(output / 'table_export_status.json', {'status': 'failed', 'updated_utc': utc_now(), 'prediction_count': len(predictions), 'error': str(error)})
            print(f'Table export failed; predictions are retained: {error}', file=sys.stderr, flush=True)
            return False
    return True


def verify_image(root, item):
    path = child_path(root, item['relative_path'])
    if file_hash(path) != item['sha256']:
        raise ValueError(f"Image SHA256 differs from manifest: {item['relative_path']}")
    return path


def main():
    parser = argparse.ArgumentParser(description='run validation')
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'Results' / 'zeroshot_20261106')
    parser.add_argument('--weights-dir', type=Path, default=WEIGHTS_ROOT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit-per-source', type=int, help='Smoke run only; omit for all available data')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--checkpoint-every', type=int, default=25)
    args = parser.parse_args()
    if args.threads < 1 or args.checkpoint_every < 1:
        parser.error('Thread and checkpoint counts must be positive')
    if args.retry_errors and not args.resume:
        parser.error('--retry-errors requires --resume')
    if not args.device.startswith('cuda'):
        parser.error('This experiment explicitly requires a local CUDA GPU')
    root = args.data_root.resolve()
    manifest = (args.manifest or root / 'manifests' / 'eval_unique.csv').resolve()
    output = args.output_dir.resolve()
    if not args.resume and output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty; use --resume or a new directory')
    import torch
    from fsd import FSDDetector
    from feature_cache import FeatureCache
    from feature_pipeline import score_and_cache
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; no silent CPU fallback')
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    print('Validating manifest; each image SHA256 is checked immediately before inference...', flush=True)
    rows = load_manifest(manifest, root, verify_hashes=False)
    selected = select_rows(rows, args.limit_per_source)
    weights, config = weights_identity(args.weights_dir.resolve())
    threshold = float(config['scoring']['default_threshold'])
    code_paths = [Path(__file__), HERE / 'export_tables.py', HERE / 'feature_cache.py',
                  HERE / 'feature_pipeline.py', EXTENSION_ROOT / 'zeroshot_common.py',
                  EXTENSION_ROOT / 'evaluate_zeroshot.py', *sorted((CODE / 'fsd').glob('*.py'))]
    dimensions = int(config['fre']['out_channels']) * (int(config['fsd']['kernel_size']) ** 2 - 1)
    extraction_identity = {
        'representation': 'raw_fsd_before_projections', 'dimensions': dimensions, 'dtype': '<f8',
        'fre_config': config['fre'], 'fsd_config': config['fsd'],
        'fre_weights_sha256': weights[config['fre']['weights_file']],
        'extractor_code_sha256': {name: file_hash(CODE / 'fsd' / name)
                                  for name in ('detector.py', 'fre.py', 'fsd_computation.py')},
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'Pillow')},
        'device': args.device, 'gpu': torch.cuda.get_device_name(args.device),
        'tf32_matmul': False, 'tf32_cudnn': False, 'cudnn_benchmark': False,
        'torch_threads': args.threads, 'seed': 2026,
    }
    identity = {
        'schema_version': 2, 'manifest_sha256': file_hash(manifest), 'manifest_path': str(manifest),
        'manifest_count': len(rows), 'selected_count': len(selected), 'data_root': str(root),
        'weights_sha256': weights, 'model_config': config,
        'paper_reference_sha256': file_hash(REFERENCE_PATH),
        'code_sha256': {p.relative_to(CODE).as_posix(): file_hash(p) for p in code_paths},
        'threshold': threshold, 'decision_rule': 'z_score < threshold', 'ranking_score': '-z_score',
        'device': args.device, 'gpu': torch.cuda.get_device_name(args.device),
        'gpu_total_memory_bytes': torch.cuda.get_device_properties(args.device).total_memory,
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'tf32_matmul': False, 'tf32_cudnn': False, 'torch_threads': args.threads,
        'python_version': platform.python_version(), 'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'Pillow', 'scipy', 'scikit-learn')},
        'limit_per_source': args.limit_per_source, 'seed': 2026,
        'all_manifest_hashes_preflight_verified': False,
        'image_integrity': 'SHA256 verified by bounded prefetch and rechecked immediately before every inference',
        'prefetch_workers': 4, 'prefetch_images': 16,
        'exact_paper_reproduction': False,
        'feature_cache': {'relative_path': 'feature_cache/raw_fsd.sqlite',
                          'commit_before_prediction': True, 'extraction_identity': extraction_identity},
    }
    output.mkdir(parents=True, exist_ok=True)

    import msvcrt
    lock_stream = (output / 'run.lock').open('a+b')
    if lock_stream.tell() == 0:
        lock_stream.write(b'0')
        lock_stream.flush()
    lock_stream.seek(0)
    msvcrt.locking(lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
    metadata_path = output / 'run_metadata.json'
    predictions_path = output / 'predictions.csv'
    if args.resume:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        check_resume_metadata(metadata['identity'], identity)
        previous = read_previous_predictions(predictions_path, selected, threshold)
    else:
        metadata = {'created_utc': utc_now(), 'identity': identity,
                    'note': 'Released FSD v1.2.0; available-data validation of paper zero-shot evaluation protocol. Missing sources excluded. No training or threshold tuning. Every new score uses a durably cached raw FSD vector; fresh run does not reuse earlier predictions.'}
        write_json(metadata_path, metadata)
        previous = []
    if args.retry_errors:
        previous = [p for p in previous if p['status'] == 'ok']
    cache = FeatureCache(output / 'feature_cache', extraction_identity, dimensions=dimensions)
    try:


        cache.integrity_check()
        cache.coverage(selected, verify_payloads=True)
        successful_ids = {p['image_id'] for p in previous if p['status'] == 'ok'}
        required = [item for item in selected if item['image_id'] in successful_ids]
        if cache.coverage(required)['missing_count']:
            raise ValueError('Successful predictions have missing raw features; restore the matching cache before resuming')
    except BaseException:
        cache.close()
        raise
    write_csv(predictions_path, PREDICTION_FIELDS, previous)
    done_ids = {p['image_id'] for p in previous}
    todo = [r for r in selected if r['image_id'] not in done_ids]
    predictions = list(previous)
    started = time.perf_counter()
    print(json.dumps({'selected_count': len(selected), 'remaining': len(todo), 'gpu': identity['gpu'], 'sources': dict(Counter(r['source'] for r in selected))}), flush=True)
    consecutive_errors = 0
    status = 'running'
    pool = None
    import ctypes
    previous_power_state = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        snapshot(output, predictions, rows, selected, identity, started, status, cache)
        detector = FSDDetector.load(weights_dir=args.weights_dir.resolve(), device=args.device, attribution=False) if todo else None
        pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='image-prefetch')
        pending = {i: pool.submit(verify_image, root, item) for i, item in enumerate(todo[:16])}
        with predictions_path.open('a', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
            for offset, item in enumerate(todo):
                tick = time.perf_counter()
                prediction = {**item, 'status': 'error', 'z_score': '', 'raw_score': '', 'fake_score': '',
                              'predicted_fake': '', 'threshold': repr(threshold), 'elapsed_seconds': '', 'error': ''}
                try:
                    pending.pop(offset).result()
                    path = verify_image(root, item)
                    result, cache_hit = score_and_cache(detector, path, item, cache)
                    z, raw = float(result.z_score), float(result.raw_score)
                    if not all(math.isfinite(x) for x in (z, raw)):
                        raise ValueError('Nonfinite model output')
                    if float(result.threshold) != threshold or bool(result.is_fake) != (z < threshold):
                        raise ValueError('Detector disagrees with fixed threshold rule')
                    prediction.update(status='ok', z_score=repr(z), raw_score=repr(raw), fake_score=repr(-z), predicted_fake=str(int(z < threshold)))
                    consecutive_errors = 0
                except Exception as error:
                    prediction['error'] = f'{type(error).__name__}: {error}'.replace('\n', ' ').replace('\r', ' ')
                    consecutive_errors += 1
                    torch.cuda.empty_cache()
                next_offset = offset + 16
                if next_offset < len(todo):
                    pending[next_offset] = pool.submit(verify_image, root, todo[next_offset])
                prediction['elapsed_seconds'] = f'{time.perf_counter() - tick:.6f}'
                writer.writerow(prediction)
                stream.flush()
                predictions.append(prediction)
                print(f"[{len(predictions)}/{len(selected)}] {item['source']} {prediction['status']} {prediction['elapsed_seconds']}s {prediction['error']}", flush=True)
                if len(predictions) % args.checkpoint_every == 0 or prediction['status'] == 'error' or len(predictions) == 1:
                    os.fsync(stream.fileno())
                    snapshot(output, predictions, rows, selected, identity, started, 'running', cache)
                if consecutive_errors >= 3:
                    raise RuntimeError('Stopped after 3 consecutive inference errors; fix before --resume --retry-errors')
        status = 'completed_with_errors' if any(p['status'] == 'error' for p in predictions) else 'completed'
    except BaseException:
        status = 'interrupted'
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        if previous_power_state:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        try:
            report_ok = snapshot(output, predictions, rows, selected, identity, started, status, cache)
            write_json(output / 'gpu_memory.json', {'peak_allocated_bytes': torch.cuda.max_memory_allocated(args.device),
                                                   'peak_reserved_bytes': torch.cuda.max_memory_reserved(args.device)})
        finally:
            cache.close()
            lock_stream.close()
    print(json.dumps({'status': status, 'output': str(output)}, ensure_ascii=False), flush=True)
    return 0 if status == 'completed' and report_ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
