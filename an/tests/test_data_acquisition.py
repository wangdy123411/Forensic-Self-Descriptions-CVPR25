"""Offline checks for bounded range downloads and native ZIP member integrity."""

from __future__ import annotations

import io
from pathlib import Path
import re
import struct
import sys
import unittest
from unittest.mock import patch
import zipfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robustness.data import HTTPRangeFile, RemoteZip


class Response(io.BytesIO):
    def __init__(self, payload, status, content_range):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Range": content_range}


class SparseHTTP:
    """Serve bounded pieces of a virtual file without allocating its gaps."""

    def __init__(self, segments, total):
        self.segments = segments
        self.total = total
        self.requests = []

    def __call__(self, request, timeout=None):
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        start, requested_end = map(int, match.groups())
        end = min(requested_end, self.total - 1)
        self.requests.append((start, end))
        payload = bytearray(end - start + 1)
        for offset, block in self.segments:
            left, right = max(start, offset), min(end + 1, offset + len(block))
            if left < right:
                payload[left - start : right - start] = block[left - offset : right - offset]
        return Response(bytes(payload), 206, f"bytes {start}-{end}/{self.total}")


def ordinary_zip(compression=zipfile.ZIP_DEFLATED, force_zip64=False):
    payload = bytes(range(256)) * 900
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        if force_zip64:
            info = zipfile.ZipInfo("images/native.png")
            info.compress_type = compression
            with archive.open(info, "w", force_zip64=True) as member:
                member.write(payload)
        else:
            archive.writestr("images/native.png", payload)
    raw = output.getvalue()
    return SparseHTTP([(0, raw)], len(raw)), payload


def sparse_zip64():
    """An actual ZIP64 directory with 64-bit member and directory offsets."""
    name = b"images/large-offset.png"
    payload = b"native-image-fixture" * 31
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    offset = 2**32 + 4096
    local = (
        struct.pack(
            "<4s5H3I2H",
            b"PK\x03\x04",
            45,
            0,
            0,
            0,
            0,
            crc,
            len(payload),
            len(payload),
            len(name),
            0,
        )
        + name
        + payload
    )
    directory_offset = offset + len(local)
    extra = struct.pack("<HHQ", 1, 8, offset)
    central = (
        struct.pack(
            "<4s6H3I5H2I",
            b"PK\x01\x02",
            45,
            45,
            0,
            0,
            0,
            0,
            crc,
            len(payload),
            len(payload),
            len(name),
            len(extra),
            0,
            0,
            0,
            0,
            0xFFFFFFFF,
        )
        + name
        + extra
    )
    zip64_offset = directory_offset + len(central)
    zip64_end = struct.pack(
        "<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, 1, 1, len(central), directory_offset
    )
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, zip64_offset, 1)
    end = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 1, 1, len(central), 0xFFFFFFFF, 0)
    tail = central + zip64_end + locator + end
    server = SparseHTTP([(offset, local), (directory_offset, tail)], directory_offset + len(tail))
    return server, payload, offset


class RangeAndZIPTests(unittest.TestCase):
    def test_native_member_roundtrip_supported_compressions(self):
        for compression in [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2]:
            with self.subTest(compression=compression):
                server, expected = ordinary_zip(compression)
                with patch("robustness.data.urllib.request.urlopen", server):
                    archive = RemoteZip("https://example.invalid/native.zip")
                    members = archive.members()
                    self.assertEqual(len(members), 1)
                    self.assertEqual(archive.extract_bytes(members[0]), expected)

    def test_forced_zip64_local_extra_field(self):
        server, expected = ordinary_zip(force_zip64=True)
        with patch("robustness.data.urllib.request.urlopen", server):
            archive = RemoteZip("https://example.invalid/zip64.zip")
            self.assertEqual(archive.extract_bytes(archive.members()[0]), expected)

    def test_zip64_offsets_beyond_four_gib_without_large_allocation(self):
        server, expected, offset = sparse_zip64()
        with patch("robustness.data.urllib.request.urlopen", server):
            archive = RemoteZip("https://example.invalid/sparse.zip")
            self.assertGreater(archive.remote.total, 2**32)
            self.assertEqual(archive.members()[0].header_offset, offset)
            self.assertEqual(archive.extract_bytes(archive.members()[0]), expected)
        self.assertLess(sum(end - start + 1 for start, end in server.requests), 1_000_000)

    def test_crc_rejects_modified_native_bytes(self):
        server, _ = ordinary_zip(zipfile.ZIP_STORED)
        raw = bytearray(server.segments[0][1])
        name_length = struct.unpack_from("<H", raw, 26)[0]
        raw[30 + name_length + 17] ^= 1
        server = SparseHTTP([(0, bytes(raw))], len(raw))
        with patch("robustness.data.urllib.request.urlopen", server):
            archive = RemoteZip("https://example.invalid/corrupt.zip")
            with self.assertRaisesRegex(ValueError, "CRC/size check failed"):
                archive.extract_bytes(archive.members()[0])

    def test_rejects_full_response_misaligned_and_truncated_ranges(self):
        cases = [
            (200, "bytes 0-0/10", b"x"),
            (206, "bytes 1-1/10", b"x"),
            (206, "bytes 0-0/10", b""),
        ]
        for status, content_range, body in cases:
            with self.subTest(status=status, content_range=content_range, body=body):
                with patch(
                    "robustness.data.urllib.request.urlopen",
                    side_effect=lambda *a, **k: Response(body, status, content_range),
                ):
                    with self.assertRaises(RuntimeError):
                        HTTPRangeFile("https://example.invalid/file", retries=1)

    def test_valid_http_ranges_do_not_make_html_a_zip_archive(self):
        html = b"<html><body>This is not a dataset archive.</body></html>"
        server = SparseHTTP([(0, html)], len(html))
        with patch("robustness.data.urllib.request.urlopen", server):
            with self.assertRaises(zipfile.BadZipFile):
                RemoteZip("https://example.invalid/mirror.zip")


if __name__ == "__main__":
    unittest.main()
