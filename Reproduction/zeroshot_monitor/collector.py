from __future__ import annotations

from collections import Counter, deque
import csv
import ctypes
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time

REAL_SOURCES = ('COCO2017', 'ImageNet-1k', 'ImageNet-22k', 'MIDB')
FAKE_SOURCES = ('BigGAN', 'DALLE 2', 'DALLE 3', 'DALLE M', 'EG3D', 'FireFly', 'GigaGAN', 'GLIDE', 'Guided Dif', 'Latent Dif', 'MJ v5', 'MJ v6', 'ProGAN', 'Proj.GAN', 'SD1.3', 'SD1.4', 'SD1.5', 'SD2.1', 'SD3.0', 'SDXL', 'StyleGAN', 'StyleGAN2', 'StyleGAN3', 'Tam.Xformer')


def iso(value=None):
    return datetime.fromtimestamp(time.time() if value is None else value, timezone.utc).isoformat()


def timestamp(value):
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()
    except (ValueError, TypeError):
        return None


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return None
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return None
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
    kernel.GetExitCodeProcess.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False if ctypes.get_last_error() == 87 else None
    try:
        code = ctypes.c_ulong()
        return code.value == 259 if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) else None
    finally:
        kernel.CloseHandle(handle)


def gpu_info():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader,nounits', '--id=0'],
            capture_output=True, text=True, timeout=4,
            creationflags=0x08000000 if os.name == 'nt' else 0,
        )
        if result.returncode:
            return None
        values = next(csv.reader(io.StringIO(result.stdout)))
        if len(values) != 6:
            return None
        return dict(zip(('name', 'utilization_percent', 'memory_used_mib', 'memory_total_mib', 'temperature_c', 'power_w'), [values[0].strip(), *[finite(v.strip()) for v in values[1:]]]))
    except (OSError, subprocess.SubprocessError, StopIteration):
        return None


class LivePredictions:

    def __init__(self, path):
        self.path = Path(path)
        self.reset()

    def reset(self):
        self.offset = 0
        self.identity = None
        self.mtime_ns = None
        self.header = None
        self.count = self.ok = self.errors = 0
        self.seconds = 0.0
        self.by_source = Counter()
        self.ok_by_source = Counter()
        self.error_by_source = Counter()
        self.history = deque(maxlen=60)
        self.recent = deque(maxlen=8)
        self.seen_ids = set()
        self.last_result_at = None

    def update(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return
        identity = (stat.st_dev, stat.st_ino)
        if (self.identity is not None and identity != self.identity) or stat.st_size < self.offset or (stat.st_size == self.offset and self.mtime_ns is not None and stat.st_mtime_ns != self.mtime_ns):
            self.reset()
        self.identity = identity
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            chunk = stream.read()
        newline = chunk.rfind(b'\n')
        if newline < 0:
            return
        complete = chunk[:newline + 1]
        records = list(csv.reader(io.StringIO(complete.decode('utf-8-sig')), strict=True))
        if self.header is None:
            if not records:
                return
            self.header = records.pop(0)
            required = {'image_id', 'source', 'status', 'elapsed_seconds', 'z_score'}
            if not required.issubset(self.header):
                raise ValueError('Unexpected prediction schema')
        for values in records:
            if len(values) != len(self.header):
                raise ValueError('Incomplete prediction record')
            row = dict(zip(self.header, values))
            if row['image_id'] in self.seen_ids:
                raise ValueError('Duplicate prediction record')
            if row['status'] not in ('ok', 'error'):
                raise ValueError('Unknown prediction state')
            seconds = finite(row['elapsed_seconds'])
            if seconds is None or seconds < 0:
                raise ValueError('Invalid prediction elapsed time')
            self.seen_ids.add(row['image_id'])
            self.count += 1
            self.seconds += seconds
            self.by_source[row['source']] += 1
            if row['status'] == 'ok':
                self.ok += 1
                self.ok_by_source[row['source']] += 1
            else:
                self.errors += 1
                self.error_by_source[row['source']] += 1
            item = {'index': self.count, 'source': row['source'], 'status': row['status'], 'seconds': seconds, 'z_score': finite(row['z_score'])}
            if row['status'] == 'error':
                item['error'] = '推理失败，请在本机查看日志'
            self.history.append({'index': self.count, 'seconds': seconds, 'source': row['source']})
            self.recent.append(item)
        self.offset += len(complete)
        self.mtime_ns = stat.st_mtime_ns
        if records:
            self.last_result_at = stat.st_mtime


def selected_tables(report):
    if not report:
        return None

    allowed1 = {'method', 'value_kind', 'run_status', 'Average', 'Average_pairs', *REAL_SOURCES, *(x + '_pairs' for x in REAL_SOURCES)}
    allowed2 = {'real_source', 'measured_worst_auc', 'measured_generators', 'evaluated_generator_count', 'paper_worst_auc', 'paper_generator'}
    return {
        'main_table_1': [{k: v for k, v in row.items() if k in allowed1} for row in report.get('main_table_1', [])],
        'main_table_2': [{k: v for k, v in row.items() if k in allowed2} for row in report.get('main_table_2', [])],
    }


class Collector:
    def __init__(self, run_dir, launcher_path=None):
        self.run_dir = Path(run_dir)
        self.launcher_path = Path(launcher_path) if launcher_path else self.run_dir.parent / (self.run_dir.name + '_launcher.json')
        self.predictions = LivePredictions(self.run_dir / 'predictions.csv')
        self.expected = None

    def snapshot(self, *, now=None, alive_check=process_alive, gpu_provider=gpu_info):
        now = time.time() if now is None else now
        progress = read_json(self.run_dir / 'progress.json')
        metadata = read_json(self.run_dir / 'run_metadata.json')
        launcher = read_json(self.launcher_path)
        identity = metadata.get('identity', {})
        if self.expected is None:
            expected = progress.get('expected_by_source', {})
            self.expected = {str(k): int(v) for k, v in expected.items()} if expected else None
            if self.expected is None and identity.get('manifest_path'):
                with Path(identity['manifest_path']).open(encoding='utf-8-sig', newline='') as stream:
                    self.expected = dict(Counter(r['source'] for r in csv.DictReader(stream)))
        expected = self.expected or {}
        self.predictions.update()
        p = self.predictions
        total = sum(expected.values()) or int(identity.get('selected_count', 0))
        if total and p.count > total:
            raise ValueError('More predictions than selected images')
        started_at = metadata.get('created_utc') or launcher.get('started_utc')
        started = timestamp(started_at)
        age = max(0.0, now - p.last_result_at) if p.last_result_at else None
        alive = alive_check(progress.get('pid') or launcher.get('pid'))
        mean = p.seconds / p.count if p.count else None
        recent_mean = sum(x['seconds'] for x in p.history) / len(p.history) if p.history else None
        remaining = max(0, total - p.count)
        if total and p.count == total:
            status = 'completed_with_errors' if p.errors else 'completed'
        elif alive is False:
            status = 'stopped'
        elif age is not None and age > max(180.0, (mean or 0) * 8):
            status = 'stalled' if alive else 'stopped'
        else:
            status = 'running' if p.count or alive else 'starting'
        sources = []
        for name in (*REAL_SOURCES, *FAKE_SOURCES):
            n = expected.get(name, 0)
            done, error, ok = p.by_source[name], p.error_by_source[name], p.ok_by_source[name]
            state = 'missing' if not n else ('error' if error else ('complete' if done == n else ('running' if done else 'pending')))
            sources.append({'name': name, 'label': 'real' if name in REAL_SOURCES else 'fake', 'total': n, 'completed': done, 'ok': ok, 'errors': error, 'status': state})
        done_reals = sum(s['label'] == 'real' and s['status'] == 'complete' for s in sources)
        done_fakes = sum(s['label'] == 'fake' and s['status'] == 'complete' for s in sources)
        available_reals = sum(s['label'] == 'real' and s['total'] > 0 for s in sources)
        available_fakes = sum(s['label'] == 'fake' and s['total'] > 0 for s in sources)
        report = read_json(self.run_dir / 'tables' / 'table_report.json')
        gpu = gpu_provider()
        return {
            'schema_version': 1, 'experiment': 'FSD zero-shot', 'run_id': self.run_dir.name,
            'captured_at': iso(now), 'status': status, 'process_alive': alive,
            'total': total, 'completed': p.count, 'ok': p.ok, 'errors': p.errors, 'remaining': remaining,
            'percent': p.count / total * 100 if total else 0,
            'elapsed_seconds': max(0, now - started) if started else 0,
            'mean_seconds_per_image': mean, 'recent_seconds_per_image': recent_mean,
            'eta_seconds': remaining * recent_mean if recent_mean and status == 'running' else (0 if not remaining and total else None),
            'started_at': started_at, 'last_result_at': iso(p.last_result_at) if p.last_result_at else None,
            'seconds_since_last_result': age,
            'current_source': p.recent[-1]['source'] if p.recent else None,
            'sources': sources, 'gpu': gpu, 'history': list(p.history), 'recent_results': list(reversed(p.recent)),
            'missing_sources': [s['name'] for s in sources if not s['total']],
            'expected_pairs': 96, 'available_pairs': available_reals * available_fakes,
            'completed_pairs': done_reals * done_fakes,
            'table_status': report.get('run_status', 'PENDING'), 'tables': selected_tables(report),
        }
