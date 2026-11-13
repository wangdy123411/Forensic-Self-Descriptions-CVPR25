from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

REAL_SOURCES = ("COCO2017", "ImageNet-1k", "ImageNet-22k", "MIDB")
FAKE_SOURCES = (
    "BigGAN", "DALLE 2", "DALLE 3", "DALLE M", "EG3D", "FireFly",
    "GigaGAN", "GLIDE", "Guided Dif", "Latent Dif", "MJ v5", "MJ v6",
    "ProGAN", "Proj.GAN", "SD1.3", "SD1.4", "SD1.5", "SD2.1",
    "SD3.0", "SDXL", "StyleGAN", "StyleGAN2", "StyleGAN3", "Tam.Xformer",
)
PAPER_COUNTS = {source: 1000 for source in (*REAL_SOURCES, *FAKE_SOURCES)}
PAPER_COUNTS.update({source: 2000 for source in ("DALLE 2", "DALLE 3", "GLIDE", "SD1.4", "SDXL")})
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
MANIFEST_FIELDS = ("image_id", "relative_path", "label", "source", "split", "sha256")
PREDICTION_FIELDS = (*MANIFEST_FIELDS, "status", "z_score", "raw_score", "fake_score", "predicted_fake", "threshold", "elapsed_seconds", "error")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_checkpoint(temporary: Path, path: Path) -> None:

    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    replace_checkpoint(temporary, path)


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    replace_checkpoint(temporary, path)


def child_path(root: Path, relative: str) -> Path:

    portable = relative.replace("\\", "/")
    if not portable or PureWindowsPath(relative).drive or portable.startswith("/"):
        raise ValueError(f"Expected a relative path: {relative!r}")
    result = (root / portable).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes data root: {relative!r}")
    return result


def image_id(source: str, relative: str, digest: str) -> str:
    return hashlib.sha256(f"{source}\0{relative}\0{digest}".encode("utf-8")).hexdigest()[:24]


def load_manifest(path: Path, root: Path, *, verify_hashes=True) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(MANIFEST_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest needs columns: {', '.join(MANIFEST_FIELDS)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    ids, paths, hashes, within_source = set(), set(), {}, set()
    source_labels = {}
    for number, row in enumerate(rows, start=2):
        if any(not isinstance(row.get(field), str) for field in MANIFEST_FIELDS):
            raise ValueError(f"Row {number}: incomplete manifest row")
        if row["label"] not in ("0", "1") or row["split"] != "test" or not row["source"].strip():
            raise ValueError(f"Row {number}: label must be 0/1, split must be test, and source must be nonempty")
        label = int(row["label"])
        source = row["source"]
        if source in source_labels and source_labels[source] != label:
            raise ValueError(f"Source has conflicting labels: {source}")
        if source in REAL_SOURCES and label != 0 or source in FAKE_SOURCES and label != 1:
            raise ValueError(f"Label conflicts with paper source: {source}")
        source_labels[source] = label
        item_path = child_path(root, row["relative_path"])
        if not item_path.is_file():
            raise FileNotFoundError(f"Manifest image missing: {item_path}")
        path_key = str(item_path)
        if row["image_id"] in ids or not row["image_id"] or path_key in paths:
            raise ValueError(f"Duplicate image ID or path: {row['relative_path']}")
        digest = row["sha256"].lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid SHA256 at row {number}")
        if verify_hashes and file_hash(item_path) != digest:
            raise ValueError(f"SHA256 mismatch: {item_path}")
        if digest in hashes and hashes[digest] != label:
            raise ValueError(f"Identical image bytes have conflicting real/fake labels: {item_path}")
        if (source, digest) in within_source:
            raise ValueError(f"Duplicate image content within source: {source}: {item_path}")
        ids.add(row["image_id"])
        paths.add(path_key)
        hashes[digest] = label
        within_source.add((source, digest))
        row["sha256"] = digest
        row["label"] = str(label)
    return rows


def binary_metrics(labels, scores, predicted) -> dict:

    if not (len(labels) == len(scores) == len(predicted)) or not labels:
        raise ValueError("Metric inputs must be nonempty and have equal lengths")
    if any(y not in (0, 1) for y in labels) or any(p not in (0, 1) for p in predicted):
        raise ValueError("Labels and predictions must be binary")
    if any(not math.isfinite(float(s)) for s in scores):
        raise ValueError("Scores must be finite")
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predicted))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predicted))
    positive, negative = tp + fn, tn + fp
    auc = ap = None
    if positive and negative:
        by_score = defaultdict(lambda: [0, 0])
        for label, score in zip(labels, scores):
            by_score[float(score)][label] += 1
        preceding_negative = wins = 0.0
        for score in sorted(by_score):
            negatives, positives = by_score[score]
            wins += positives * (preceding_negative + 0.5 * negatives)
            preceding_negative += negatives
        auc = wins / (positive * negative)
        seen_positive = seen_total = 0
        ap = 0.0
        for score in sorted(by_score, reverse=True):
            negatives, positives = by_score[score]
            seen_positive += positives
            seen_total += positives + negatives
            ap += (positives / positive) * (seen_positive / seen_total)
    recall = tp / positive if positive else None
    specificity = tn / negative if negative else None
    return {
        "n": len(labels), "n_real": negative, "n_fake": positive,
        "roc_auc": auc, "average_precision": ap,
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (recall + specificity) / 2 if positive and negative else None,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": recall, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "false_positive_rate": fp / negative if negative else None,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def metrics_for(rows) -> dict:
    return binary_metrics(
        [int(r["label"]) for r in rows],
        [float(r["fake_score"]) for r in rows],
        [int(r["predicted_fake"]) for r in rows],
    )


def summarize(predictions: list[dict], manifest: list[dict], *, limited=False) -> dict:
    successful = [r for r in predictions if r["status"] == "ok"]
    failed = [r for r in predictions if r["status"] == "error"]
    groups = defaultdict(list)
    for row in successful:
        groups[(int(row["label"]), row["source"])].append(row)
    reals = sorted(source for label, source in groups if label == 0)
    fakes = sorted(source for label, source in groups if label == 1)
    pairs = [
        {"real_source": real, "fake_source": fake, **metrics_for(groups[(0, real)] + groups[(1, fake)])}
        for real in reals for fake in fakes
    ]
    by_real = {}
    for real in reals:
        aucs = [r["roc_auc"] for r in pairs if r["real_source"] == real]
        paper_aucs = [r["roc_auc"] for r in pairs if r["real_source"] == real and r["fake_source"] in FAKE_SOURCES]
        by_real[real] = {
            "n_real": len(groups[(0, real)]), "available_generator_count": len(aucs),
            "macro_auc": sum(aucs) / len(aucs) if aucs else None,
            "macro_auc_scope": "All available generators, including sources outside the paper",
            "paper_generator_count": len(paper_aucs),
            "paper_generator_macro_auc": sum(paper_aucs) / len(paper_aucs) if paper_aucs else None,
            "worst_auc": min(aucs) if aucs else None,
            "real_only_metrics": metrics_for(groups[(0, real)]),
        }
    missing_real = [x for x in REAL_SOURCES if x not in reals]
    missing_fake = [x for x in FAKE_SOURCES if x not in fakes]
    expected_pairs = [r for r in pairs if r["real_source"] in REAL_SOURCES and r["fake_source"] in FAKE_SOURCES]
    all_counts = Counter(r["source"] for r in successful)
    shortages = {s: {"observed": all_counts[s], "paper_test_count": n} for s, n in PAPER_COUNTS.items() if all_counts[s] < n}
    coverage_complete = not (missing_real or missing_fake or shortages or failed or limited)
    return {
        "protocol": "FSD zero-shot; fake_score=-z_score; pairwise ROC AUC then unweighted macro average",
        "paper_reference": "https://arxiv.org/abs/2503.21003v1",
        "exact_paper_reproduction": False,
        "reproduction_note": "Original paper sample IDs and train/test provenance are not supplied. Coverage does not establish exact reproduction.",
        "coverage_status": "all_paper_sources_and_counts_covered" if coverage_complete else "partial",
        "manifest_count": len(manifest), "scored_count": len(successful), "failed_count": len(failed),
        "selected_count": len(predictions), "limit_per_source_applied": limited,
        "missing_real_sources": missing_real, "missing_fake_sources": missing_fake,
        "short_sources": shortages,
        "expected_pair_count": 96, "available_expected_pair_count": len(expected_pairs),
        "available_pair_count_including_extra_sources": len(pairs),
        "macro_auc_available_pairs": sum(p["roc_auc"] for p in pairs) / len(pairs) if pairs else None,
        "macro_auc_available_paper_pairs": sum(p["roc_auc"] for p in expected_pairs) / len(expected_pairs) if expected_pairs else None,
        "paper_96_pair_macro_auc": sum(p["roc_auc"] for p in expected_pairs) / 96 if coverage_complete else None,
        "auc_note": "No real/fake pairs available; real-only runs report false-positive rates." if not pairs else "Each available real source is paired with each available generator; pooled AUC is not the paper macro AUC.",
        "per_real_source": by_real, "pairs": pairs,
        "failed_images": [{"image_id": r["image_id"], "relative_path": r["relative_path"], "error": r["error"]} for r in failed],
    }


def check_resume_metadata(stored: dict, expected: dict) -> None:
    if stored != expected:
        changed = sorted(k for k in set(stored) | set(expected) if stored.get(k) != expected.get(k))
        raise ValueError(f"Cannot resume: run identity changed ({', '.join(changed)}). Use a new output directory.")
