import io
import json
from pathlib import Path
import re
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample_remote_zip import RangeFile, TransferBudget, run


class LocalServer:
    def __init__(self, payload, mode="ok"):
        self.requests = []
        self.payload, self.mode = payload, mode
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                value = self.headers.get("Range", "")
                owner.requests.append(value)
                if owner.mode == "retry" and len(owner.requests) == 1:
                    self.send_error(503)
                    return
                if owner.mode == "whole":
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(owner.payload)))
                    self.end_headers()
                    return
                match = re.fullmatch(r"bytes=(\d+)-(\d+)", value)
                if not match:
                    self.send_error(400)
                    return
                start, end = map(int, match.groups())
                body = owner.payload[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(owner.payload) + (1 if owner.mode == 'wrong' else 0)}")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/data.zip"

    def __enter__(self): return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def fixture():
    stream, originals = io.BytesIO(), {}
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for source in ("real", "fake"):
            for i in range(5):
                name = f"dataset/{source}/image{i}.png"
                originals[name] = b"original-image-bytes:" + name.encode() + bytes(range(256))
                archive.writestr(name, originals[name])
        archive.writestr("dataset/fake/readme.txt", "excluded")
    return stream.getvalue(), originals


def config(url, size):
    return {"archives": [{"id": "fixture", "url": url, "size": size, "groups": [
        {"source": "COCO2017", "label": 0, "prefixes": ["dataset/real/"]},
        {"source": "GLIDE", "label": 1, "prefixes": ["dataset/fake/"]}]}]}


class SamplingTests(unittest.TestCase):
    def test_original_bytes_resume_and_seed(self):
        payload, originals = fixture()
        with LocalServer(payload) as server, tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "samples"
            cfg = config(server.url, len(payload))
            state = run(cfg, root, per_source=2)
            self.assertEqual(state["status"], "complete")
            self.assertEqual(len(state["images"]), 4)
            for row in state["images"]:
                self.assertEqual((root / row["relative_path"]).read_bytes(), originals[row["member"]])
            resumed = run(cfg, root, per_source=2, resume=True)
            self.assertEqual(state["images"], resumed["images"])
            second = run(cfg, Path(temp) / "second", per_source=2)
            self.assertEqual(state["selection"], second["selection"])
            with self.assertRaisesRegex(ValueError, "config or seed"):
                run(cfg, root, per_source=2, seed=1, resume=True)
            before = (root / "sample_state.json").read_bytes()
            (root / state["images"][-1]["relative_path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "Resume"):
                run(cfg, root, per_source=2, resume=True)
            self.assertEqual(before, (root / "sample_state.json").read_bytes())

    def test_plan_has_no_output_mutations(self):
        payload, _ = fixture()
        with LocalServer(payload) as server, tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "never-created"
            state = run(config(server.url, len(payload)), root, per_source=1, plan=True)
            self.assertEqual(state["status"], "plan")
            self.assertEqual(len(state["images"]), 2)
            self.assertFalse(root.exists())

    def test_resume_failure_preserves_all_previously_verified_records(self):
        payload, _ = fixture()
        with LocalServer(payload) as server, tempfile.TemporaryDirectory() as temp:
            root, cfg = Path(temp) / "samples", config(server.url, len(payload))
            state = run(cfg, root, per_source=2)
            retained = state["images"][-1]
            for row in state["images"][:-1]:
                (root / row["relative_path"]).unlink()
            state["images"] = [retained]
            state["status"] = "partial"
            (root / "sample_state.json").write_text(json.dumps(state), encoding="utf-8")
            original_read, calls = zipfile.ZipFile.read, []

            def fail_second(archive, name, *args, **kwargs):
                calls.append(name)
                if len(calls) == 2: raise OSError("simulated interrupted resume")
                return original_read(archive, name, *args, **kwargs)

            with patch.object(zipfile.ZipFile, "read", fail_second), self.assertRaisesRegex(OSError, "interrupted"):
                run(cfg, root, per_source=2, resume=True)
            interrupted = json.loads((root / "sample_state.json").read_text(encoding="utf-8"))
            self.assertIn(retained, interrupted["images"])
            self.assertEqual(len(interrupted["images"]), 2)
            self.assertEqual(len(run(cfg, root, per_source=2, resume=True)["images"]), 4)

    def test_preserved_member_destination_and_escape_rejection(self):
        payload, originals = fixture()
        with LocalServer(payload) as server, tempfile.TemporaryDirectory() as temp:
            root, cfg = Path(temp) / "samples", config(server.url, len(payload))
            cfg["archives"][0]["destination"] = "Synthbuster/images"
            state = run(cfg, root, per_source=1)
            for row in state["images"]:
                self.assertEqual(row["relative_path"], "Synthbuster/images/" + row["member"])
                self.assertEqual((root / row["relative_path"]).read_bytes(), originals[row["member"]])
            cfg["archives"][0]["destination"] = "../outside"
            with self.assertRaisesRegex(ValueError, "escapes"):
                run(cfg, Path(temp) / "escape", per_source=1, plan=True)
            self.assertFalse((Path(temp) / "escape").exists())

    def test_reject_whole_response_and_wrong_range(self):
        payload, _ = fixture()
        for mode in ("whole", "wrong"):
            with self.subTest(mode=mode), LocalServer(payload, mode) as server:
                remote = RangeFile(server.url, len(payload), TransferBudget(4096), retries=0)
                with self.assertRaisesRegex(ValueError, "exact Range"):
                    remote.read(4)
                self.assertEqual(remote.budget.used, 0)

    def test_retry_and_request_bounds_and_budget(self):
        payload, _ = fixture()
        with LocalServer(payload, "retry") as server, patch("sample_remote_zip.time.sleep"):
            remote = RangeFile(server.url, len(payload), TransferBudget(100), chunk_size=8)
            self.assertEqual(remote.read(20), payload[:20])
            self.assertEqual(server.requests, ["bytes=0-7", "bytes=0-7", "bytes=8-15", "bytes=16-19"])
        with LocalServer(payload) as server:
            remote = RangeFile(server.url, len(payload), TransferBudget(4), retries=0)
            with self.assertRaisesRegex(ValueError, "budget"):
                remote.read(4)
            self.assertEqual(server.requests, [])

    def test_crc_failure_is_not_accepted(self):
        payload, originals = fixture()

        bad = payload.replace(originals["dataset/real/image0.png"], b"x" * len(originals["dataset/real/image0.png"]))
        with LocalServer(bad) as server, tempfile.TemporaryDirectory() as temp:
            cfg = config(server.url, len(bad))
            with self.assertRaises(zipfile.BadZipFile):
                run(cfg, Path(temp) / "samples", per_source=5)


if __name__ == "__main__":
    unittest.main()
