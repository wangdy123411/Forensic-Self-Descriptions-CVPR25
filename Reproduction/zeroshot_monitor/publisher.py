from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from collector import Collector, iso


class NoRedirect(__import__('urllib.request', fromlist=['HTTPRedirectHandler']).HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):

        return None


def publish(url, token, snapshot):
    from urllib.request import build_opener
    data = json.dumps(snapshot, ensure_ascii=False, allow_nan=False).encode('utf-8')
    request = Request(url, data=data, method='POST', headers={
        'Content-Type': 'application/json', 'OAI-Sites-Authorization': 'Bearer ' + token,
        'User-Agent': 'FSD-Progress-Monitor/1.0',
    })
    with build_opener(NoRedirect).open(request, timeout=20) as response:
        body = response.read(16384)
        if response.status not in (200, 201, 202, 204):
            raise RuntimeError('Unexpected response status')
        if body:
            result = json.loads(body)
            if not isinstance(result, dict) or result.get('ok') is not True:
                raise RuntimeError('Server did not acknowledge progress')


def write_status(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1 * (attempt + 1))


def can_stop_after_upload(snapshot):

    return (snapshot['status'] in ('completed', 'completed_with_errors')
            and snapshot.get('process_alive') is False
            and snapshot.get('table_status') in ('COMPLETE_AVAILABLE_DATA', 'COMPLETED_WITH_ERRORS'))


def main():
    parser = argparse.ArgumentParser(description='publisher')
    parser.add_argument('--config-stdin', action='store_true', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    config = json.loads(sys.stdin.readline())
    token = config.pop('token')
    origin = config['url'].rstrip('/')
    parsed = urlsplit(origin)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('A plain HTTPS Site origin is required')
    if parsed.path not in ('', '/'):
        raise ValueError('Pass the Site origin without an API path')
    if not isinstance(token, str) or not token:
        raise ValueError('A service token is required')
    interval = max(10, min(60, float(config.get('interval_seconds', 15))))
    collector = Collector(config['run_dir'], config.get('launcher_path'))
    status_path = Path(config.get('status_path') or Path(config['run_dir']).parent / 'zeroshot_monitor' / 'publisher_status.json')
    failures = 0
    last_success = None
    while True:
        started = time.monotonic()
        try:
            snapshot = collector.snapshot()
            publish(origin + '/api/ingest', token, snapshot)
            failures = 0
            last_success = iso()
            final = can_stop_after_upload(snapshot)
            status = {'status': 'completed' if final else 'syncing', 'pid': __import__('os').getpid(), 'updated_utc': iso(), 'last_success_utc': last_success, 'site_url': origin, 'run_status': snapshot['status'], 'completed': snapshot['completed'], 'total': snapshot['total'], 'consecutive_failures': 0, 'interval_seconds': interval}
            write_status(status_path, status)
            print(f"{iso()} synced {snapshot['completed']}/{snapshot['total']} {snapshot['status']}", flush=True)
            if final or args.once:
                return 0
        except Exception as error:
            failures += 1

            error_kind = type(error).__name__
            http_status = error.code if isinstance(error, HTTPError) else None
            write_status(status_path, {'status': 'retrying', 'pid': __import__('os').getpid(), 'updated_utc': iso(), 'last_success_utc': last_success, 'site_url': origin, 'consecutive_failures': failures, 'error_type': error_kind, 'http_status': http_status})
            print(f'{iso()} sync retry ({error_kind}, HTTP {http_status})', file=sys.stderr, flush=True)
            if args.once:
                return 2
        delay = min(60, interval * (2 ** min(failures, 2)))
        time.sleep(max(.1, delay - (time.monotonic() - started)))


if __name__ == '__main__':
    raise SystemExit(main())
