import contextlib
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample_ossia_tar import BoundedHTTPReader, Budget, TARGETS, TransferLimit, retry_checkpoint, run


class OssiaSamplerTests(unittest.TestCase):
    def test_transient_windows_checkpoint_lock_retries(self):
        writer = mock.Mock(side_effect=[PermissionError("busy"), PermissionError("busy"), None])
        with mock.patch("sample_ossia_tar.time.sleep"):
            retry_checkpoint(writer, "path", {"status": "partial"})
        self.assertEqual(writer.call_count, 3)

    def test_indexed_original_stream_selection_resume_and_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index = root / "test.csv"
            compressed = io.BytesIO()
            payloads = {}
            rows = []
            with tarfile.open(fileobj=compressed, mode="w:gz") as archive:
                for arch in TARGETS:
                    for i in range(2):
                        name = f"{arch}/fixture/uncompressed/{i}.png"
                        payload = b"\x89PNG\r\n\x1a\n" + (arch + str(i)).encode()
                        payloads[name] = payload
                        member = tarfile.TarInfo(name)
                        member.size = len(payload)
                        archive.addfile(member, io.BytesIO(payload))
                        rows.append(dict(arch=arch, dataset="fixture", compression="", width=1, height=1, path=name))
            with index.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            data = compressed.getvalue()
            requests = []
            completed_requests = []
            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass
                def do_GET(self):
                    requests.append(self.headers.get("Range"))
                    start, end = map(int, self.headers["Range"][6:].split("-"))
                    if start == 0:
                        time.sleep(0.08)
                    body = data[start:end+1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    completed_requests.append(start)
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            url = f"http://127.0.0.1:{server.server_port}/test.0.tar.gz"
            archives = [dict(path="data/test.0.tar.gz", url=url, size=len(data), shard=0)]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    code = run(index, archives, "a" * 40, root / "samples", root, per_source=1, max_transfer=1024*1024, chunk_bytes=73, opener=opener)
                self.assertEqual(code, 0)
                state = json.loads((root / "samples" / "sample_state.json").read_text())
                self.assertEqual(state["counts"], {source: 1 for source in TARGETS.values()})
                self.assertFalse(state["uniform_random_sample"])
                self.assertNotEqual(completed_requests[0], 0)
                self.assertLessEqual(state["reserved_bytes_this_run"], 1024 * 1024)
                for image in state["images"]:
                    self.assertEqual((root / image["relative_path"]).read_bytes(), payloads[image["member"]])
                count = len(requests)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run(index, archives, "a"*40, root/"samples", root, per_source=1, max_transfer=1024*1024, resume=True, opener=opener), 0)
                self.assertEqual(len(requests), count)
                with BoundedHTTPReader(url, len(data), Budget(10), chunk_bytes=73, opener=opener) as reader:
                    with self.assertRaises(TransferLimit):
                        reader.read(1)
                self.assertEqual(len(requests), count)
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(TransferLimit):
                    run(index, archives, "a"*40, root/"partial", root, per_source=1, max_transfer=100, chunk_bytes=73, opener=opener)
                partial = json.loads((root / "partial" / "sample_state.json").read_text())
                self.assertEqual(partial["status"], "partial")
                self.assertLessEqual(partial["reserved_bytes_this_run"], 100)
                self.assertLessEqual(partial["transferred_bytes_this_run"], 100)
                (root / state["images"][0]["relative_path"]).write_bytes(b"modified")
                with self.assertRaisesRegex(ValueError, "bytes changed"):
                    run(index, archives, "a"*40, root/"samples", root, per_source=1, max_transfer=1024*1024, resume=True, opener=opener)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
