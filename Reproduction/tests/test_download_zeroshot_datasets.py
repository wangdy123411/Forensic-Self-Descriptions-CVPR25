import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest
from unittest import mock
import urllib.request
import zipfile

MODULE_PATH = Path(__file__).resolve().parents[1] / "download_zeroshot_datasets.py"
spec = importlib.util.spec_from_file_location("downloader", MODULE_PATH)
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)
PAYLOAD = b"original-image-archive-bytes\x00\xff" * 1000
SHA = hashlib.sha256(PAYLOAD).hexdigest()


class Handler(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.requests.append((self.path, self.headers.get("Range")))
        if self.path == "/html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html>please sign in</html>")
            return
        offset = int(self.headers.get("Range", "bytes=0-")[6:-1])
        if self.path == "/ignore":
            offset = 0
        data = PAYLOAD[offset:]
        self.send_response(206 if offset else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("ETag", '"fixture-v1"')
        if offset:
            start = offset + 1 if self.path == "/bad-range" else offset
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD)-1}/{len(PAYLOAD)}")
        self.end_headers()
        self.wfile.write(data)


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.target = self.root / "data.zip"

    def tearDown(self):
        self.tmp.cleanup()

    def source(self, endpoint="/range"):
        return {"url": self.base + endpoint, "filename": "data.zip", "expected_bytes": len(PAYLOAD), "sha256": SHA}

    def seed(self, source):
        self.target.with_name("data.zip.part").write_bytes(PAYLOAD[:1234])
        downloader.atomic_json(self.target.with_name("data.zip.part.json"), {"url": source["url"], "etag": '"fixture-v1"'})

    def test_range_resume_and_hash_verified_skip(self):
        source = self.source()
        self.seed(source)
        result = downloader.download_source(source, self.target, retries=1, opener=self.opener)
        self.assertEqual(result["sha256"], SHA)
        self.assertEqual(self.target.read_bytes(), PAYLOAD)
        self.assertEqual(Handler.requests[-1], ("/range", "bytes=1234-"))
        count = len(Handler.requests)
        result = downloader.download_source(source, self.target, retries=1, opener=self.opener)
        self.assertEqual(result["action"], "verified_existing")
        self.assertEqual(len(Handler.requests), count)

    def test_server_ignoring_range_restarts(self):
        source = self.source("/ignore")
        self.seed(source)
        downloader.download_source(source, self.target, retries=1, opener=self.opener)
        self.assertEqual(self.target.read_bytes(), PAYLOAD)

    def test_bad_content_range_rejected(self):
        source = self.source("/bad-range")
        self.seed(source)
        with self.assertRaisesRegex(RuntimeError, "Content-Range"):
            downloader.download_source(source, self.target, retries=1, opener=self.opener)
        self.assertFalse(self.target.exists())

    def test_bad_digest_is_not_committed(self):
        source = self.source()
        source["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
            downloader.download_source(source, self.target, retries=1, opener=self.opener)
        self.assertFalse(self.target.exists())

    def test_html_and_missing_access(self):
        source = {"url": self.base + "/html", "filename": "data.zip"}
        with self.assertRaisesRegex(RuntimeError, "error/login/document"):
            downloader.download_source(source, self.target, retries=1, opener=self.opener)
        result = downloader.download_source({"url": None, "filename": "data.zip"}, self.target)
        self.assertEqual(result["status"], "needs_access")
        self.assertFalse(self.target.exists())

    def test_concurrent_cli_status_and_blocked(self):
        config = {"datasets": [{"name": "Fixture", "sources": [self.source(), {**self.source(), "filename": "second.zip"}]}, {"name": "Gated", "sources": [], "blocked_reason": "Authorization required"}]}
        path = self.root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            result = downloader.main(["--config", str(path), "--root", str(self.root / "datasets"), "--workers", "2", "--no-proxy"])
        self.assertEqual(result, 2)
        state = json.loads((self.root / "datasets" / "download_status.json").read_text())
        self.assertEqual(state["datasets"]["Fixture"]["status"], "complete")
        self.assertEqual(state["datasets"]["Gated"]["status"], "blocked")


class ArchiveTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows extended paths")
    def test_windows_concurrent_resolution_prefix(self):
        original = Path.resolve
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "not_created_yet"
            def mixed_prefix(path, *args, **kwargs):
                resolved = original(path, *args, **kwargs)
                if path.name == "image.jpg":
                    return Path("\\\\?\\" + str(resolved))
                return resolved
            with mock.patch.object(Path, "resolve", mixed_prefix):
                self.assertEqual(downloader.contained_path(root, "images/image.jpg"), root / "images" / "image.jpg")

    def test_portable_path_rejection(self):
        for name in ("../bad", "/bad", "C:/bad", "a/../bad", "a\\..\\bad", "NUL.png", "a/file:stream", "file."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                downloader.safe_relative(name)

    def test_original_bytes_deterministic_subset_and_existing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr("b.png", b"exact-b-bytes")
                target.writestr("a.png", b"exact-a-bytes")
                target.writestr("notes.txt", b"metadata")
            result = downloader.extract_images(archive, root / "images", max_images=1)
            self.assertEqual(result["available_images"], 2)
            self.assertEqual(result["images"][0]["member"], "a.png")
            self.assertEqual((root / "images" / "a.png").read_bytes(), b"exact-a-bytes")
            self.assertFalse((root / "images" / "b.png").exists())
            downloader.extract_images(archive, root / "images", max_images=1)
            (root / "images" / "a.png").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "different existing"):
                downloader.extract_images(archive, root / "images", max_images=1)

    def test_zip_traversal_and_symlinks_rejected_before_write(self):
        for name, mode in (("../escape.png", 0), ("link.png", 0o120777 << 16)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                archive = root / "bad.zip"
                with zipfile.ZipFile(archive, "w") as target:
                    member = zipfile.ZipInfo(name)
                    member.external_attr = mode
                    target.writestr(member, b"payload")
                with self.assertRaises(ValueError):
                    downloader.extract_images(archive, root / "images")
                self.assertFalse((root / "images").exists())

    def test_tar_hardlinks_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "bad.tar"
            with tarfile.open(archive, "w") as target:
                member = tarfile.TarInfo("link.png")
                member.type = tarfile.LNKTYPE
                member.linkname = "outside.png"
                target.addfile(member)
            with self.assertRaisesRegex(ValueError, "link or special"):
                downloader.extract_images(archive, root / "images")


if __name__ == "__main__":
    unittest.main()
