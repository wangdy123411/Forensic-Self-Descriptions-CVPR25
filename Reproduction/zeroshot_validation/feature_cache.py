from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _item_identity(item: dict) -> tuple[str, str, str, int]:
    key, digest, source = item.get("image_id"), item.get("sha256"), item.get("source")
    if not isinstance(key, str) or not key or not isinstance(source, str) or not source.strip():
        raise ValueError("Feature cache requires a nonempty image_id and source")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Feature cache requires a lowercase SHA256 image digest")
    if item.get("label") not in (0, 1, "0", "1"):
        raise ValueError("Feature cache requires a binary image label")
    return key, digest, source, int(item["label"])


class FeatureCache:


    def __init__(self, directory: Path, extraction_identity: dict, dimensions: int = 960):
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("Feature dimensions must be a positive integer")
        if not isinstance(extraction_identity, dict) or not extraction_identity:
            raise ValueError("An explicit extraction identity is required")
        self.directory = Path(directory)
        self.path = self.directory / "raw_fsd.sqlite"
        self.dimensions = dimensions
        identity_json = _canonical(extraction_identity)
        self.signature = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, timeout=30)
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA journal_mode=DELETE")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA busy_timeout=30000")
            with self._connection:
                self._connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                self._connection.execute("""CREATE TABLE IF NOT EXISTS features (
                    image_id TEXT PRIMARY KEY,
                    image_sha256 TEXT NOT NULL,
                    source TEXT NOT NULL,
                    label INTEGER NOT NULL,
                    dimensions INTEGER NOT NULL,
                    dtype TEXT NOT NULL,
                    feature BLOB NOT NULL,
                    feature_sha256 TEXT NOT NULL,
                    extraction_signature TEXT NOT NULL,
                    created_utc TEXT NOT NULL,
                    phase TEXT NOT NULL
                )""")
                expected = {"schema_version": "1", "representation": "raw_fsd_before_projections",
                            "dtype": "<f8", "dimensions": str(dimensions),
                            "extraction_identity": identity_json, "extraction_signature": self.signature}
                actual = dict(self._connection.execute("SELECT key,value FROM metadata"))
                if actual:
                    if actual != expected:
                        raise ValueError("Feature cache extraction identity or schema differs")
                elif self._connection.execute("SELECT COUNT(*) FROM features").fetchone()[0]:
                    raise ValueError("Feature cache has vectors without extraction metadata")
                else:
                    self._connection.executemany("INSERT INTO metadata(key,value) VALUES (?,?)", expected.items())
        except BaseException:
            self._connection.close()
            raise

    def _validate_header(self, item: dict, row) -> None:
        key, digest, source, label = _item_identity(item)
        if (row["image_id"], row["image_sha256"], row["source"], row["label"]) != (key, digest, source, label):
            raise ValueError(f"Cached feature no longer matches image identity: {key}")
        if row["extraction_signature"] != self.signature:
            raise ValueError(f"Cached extraction signature differs: {key}")
        if row["dimensions"] != self.dimensions or row["dtype"] != "<f8":
            raise ValueError(f"Cached vector shape or dtype differs: {key}")
        if row["phase"] not in ("online", "backfill"):
            raise ValueError(f"Invalid cached provenance phase: {key}")
        digest = row["feature_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid feature digest: {key}")

    def _decode(self, item: dict, row) -> np.ndarray:
        self._validate_header(item, row)
        blob = row["feature"]
        if not isinstance(blob, bytes) or len(blob) != self.dimensions * 8:
            raise ValueError(f"Invalid cached vector byte length: {item['image_id']}")
        if hashlib.sha256(blob).hexdigest() != row["feature_sha256"]:
            raise ValueError(f"Cached feature checksum mismatch: {item['image_id']}")
        vector = np.frombuffer(blob, dtype="<f8")
        if not np.isfinite(vector).all():
            raise ValueError(f"Cached feature contains nonfinite values: {item['image_id']}")
        return vector.copy()

    def get(self, item: dict) -> np.ndarray | None:
        key, _, _, _ = _item_identity(item)
        row = self._connection.execute("SELECT * FROM features WHERE image_id=?", (key,)).fetchone()
        return None if row is None else self._decode(item, row)

    def put(self, item: dict, vector, phase: str = "online") -> bool:

        key, digest, source, label = _item_identity(item)
        if phase not in ("online", "backfill"):
            raise ValueError("Feature phase must be online or backfill")
        array = np.asarray(vector)
        if array.shape != (self.dimensions,) or array.dtype.kind != "f" or array.dtype.itemsize != 8:
            raise ValueError(f"Expected a float64 ({self.dimensions},) raw FSD vector")
        array = np.asarray(array, dtype="<f8", order="C")
        if not np.isfinite(array).all():
            raise ValueError("Cannot cache nonfinite FSD features")
        blob = array.tobytes(order="C")
        old = self.get(item)
        if old is not None:
            if old.tobytes(order="C") != blob:
                raise ValueError(f"Refusing to overwrite a different cached feature: {key}")
            return False
        with self._connection:
            self._connection.execute("""INSERT INTO features
                (image_id,image_sha256,source,label,dimensions,dtype,feature,
                 feature_sha256,extraction_signature,created_utc,phase)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (key, digest, source, label, self.dimensions, "<f8", sqlite3.Binary(blob),
                 hashlib.sha256(blob).hexdigest(), self.signature,
                 datetime.now(timezone.utc).isoformat(), phase))
        return True

    def coverage(self, rows: list[dict], verify_payloads: bool = False) -> dict:

        expected = {}
        for item in rows:
            key, _, _, _ = _item_identity(item)
            if key in expected:
                raise ValueError(f"Duplicate image ID in coverage request: {key}")
            expected[key] = item
        present, phases = set(), Counter()
        columns = "*" if verify_payloads else "image_id,image_sha256,source,label,dimensions,dtype,feature_sha256,extraction_signature,created_utc,phase,length(feature) AS byte_length"
        for row in self._connection.execute(f"SELECT {columns} FROM features"):
            key = row["image_id"]
            if key not in expected:
                continue
            if verify_payloads:
                self._decode(expected[key], row)
            else:
                self._validate_header(expected[key], row)
                if row["byte_length"] != self.dimensions * 8:
                    raise ValueError(f"Invalid cached vector byte length: {key}")
            present.add(key)
            phases[row["phase"]] += 1
        missing = [key for key in expected if key not in present]
        return {"total_count": len(rows), "cached_count": len(present),
                "missing_count": len(missing), "missing_image_ids": missing,
                "by_phase": dict(phases), "payloads_verified": verify_payloads,
                "extraction_signature": self.signature, "dimensions": self.dimensions,
                "dtype": "<f8", "representation": "raw_fsd_before_projections"}

    def integrity_check(self) -> bool:

        try:
            result = [row[0] for row in self._connection.execute("PRAGMA integrity_check")]
        except sqlite3.DatabaseError as error:
            raise ValueError("Feature cache database integrity check failed") from error
        if result != ["ok"]:
            raise ValueError("Feature cache database integrity check failed")
        return True

    def close(self) -> None:
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
