from __future__ import annotations

from project_paths import ARCHIVE_ROOT


import argparse
from collections import Counter
import json
from pathlib import Path
import random

from zeroshot_common import (
    FAKE_SOURCES, IMAGE_SUFFIXES, MANIFEST_FIELDS, PAPER_COUNTS, REAL_SOURCES, child_path,
    file_hash, image_id, load_manifest, write_csv, write_json,
)


def build_manifest(config: dict, root: Path, *, per_source=1000, seed=2026, allow_missing=False, allow_short=False, paper_counts=False):
    if per_source <= 0:
        raise ValueError("per_source must be positive")
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Config must contain a nonempty sources list")
    rows, statuses, names, path_keys, content_labels = [], [], set(), set(), {}
    for spec in sources:
        source, label = spec["source"], spec["label"]
        if not isinstance(source, str) or not source.strip() or source in names:
            raise ValueError(f"Duplicate or empty source name: {source!r}")
        if type(label) is not int or label not in (0, 1) or spec.get("split") != "test":
            raise ValueError(f"{source}: require integer label 0/1 and explicit split=test")
        if source in REAL_SOURCES and label != 0 or source in FAKE_SOURCES and label != 1:
            raise ValueError(f"{source}: label does not match paper source")
        names.add(source)
        default_count = PAPER_COUNTS.get(source, per_source) if paper_counts else per_source
        count = spec.get("sample_count", spec.get("count", default_count))
        if "sample_count" in spec and "count" in spec and spec["sample_count"] != spec["count"]:
            raise ValueError(f"{source}: sample_count and count disagree")
        if type(count) is not int or count <= 0:
            raise ValueError(f"{source}: sample_count must be a positive integer")
        if "directory" in spec and "directories" in spec:
            raise ValueError(f"{source}: use directory or directories, not both")
        directory_names = spec.get("directories", [spec.get("directory")])
        if not isinstance(directory_names, list) or not directory_names or any(not isinstance(d, str) or not d for d in directory_names):
            raise ValueError(f"{source}: require directory or a nonempty directories list")
        directories, missing_directories = [], []
        for name in directory_names:
            directory = child_path(root, name)
            if directory.is_dir():
                directories.append(directory)
                continue
            if not allow_missing:
                raise FileNotFoundError(f"Source directory missing: {directory}; use --allow-missing for an explicitly partial manifest")
            missing_directories.append(name)
        if not directories:
            statuses.append({"source": source, "status": "missing", "directories": directory_names, "missing_directories": missing_directories, "requested": count, "selected": 0})
            continue
        candidates = sorted({p for directory in directories for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}, key=lambda p: p.relative_to(root).as_posix())
        rng = random.Random(f"{seed}:{source}")
        rng.shuffle(candidates)
        selected, seen_hashes, duplicate_count = [], set(), 0
        for path in candidates:
            if len(selected) == count:
                break
            path = path.resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError(f"Image symlink escapes data root: {path}")
            digest = file_hash(path)
            if digest in content_labels and content_labels[digest] != label:
                raise ValueError(f"Conflicting real/fake labels for identical image: {path}")
            if digest in seen_hashes:
                duplicate_count += 1
                continue
            if str(path) in path_keys:
                raise ValueError(f"Same image path appears in more than one source: {path}")
            seen_hashes.add(digest)
            content_labels[digest] = label
            path_keys.add(str(path))
            relative = path.relative_to(root.resolve()).as_posix()
            selected.append({"image_id": image_id(source, relative, digest), "relative_path": relative, "label": str(label), "source": source, "split": "test", "sha256": digest})
        if len(selected) < count and not allow_short:
            raise ValueError(f"{source}: only {len(selected)} unique images available, requested {count}; use --allow-short for partial data")
        if not selected and not allow_missing:
            raise ValueError(f"No images found for {source}")
        rows.extend(sorted(selected, key=lambda r: r["relative_path"]))
        statuses.append({"source": source, "status": "ready" if len(selected) == count and not missing_directories else "partial", "directories": directory_names, "missing_directories": missing_directories, "candidate_count": len(candidates), "requested": count, "selected": len(selected), "duplicates_skipped": duplicate_count})
    if not rows:
        raise ValueError("No images selected; no manifest will be written")
    report = {
        "seed": seed, "default_requested_per_source": per_source, "paper_counts_default": paper_counts,
        "selection": "Per-source seeded shuffle of sorted original paths; duplicate bytes within source removed; selected paths sorted for inference",
        "exact_paper_sample_list": False,
        "provenance_note": "User-configured evaluation directories; original paper sample IDs and training exclusion are not established.",
        "data_root": str(root.resolve()), "image_count": len(rows), "sources": statuses,
        "counts": dict(Counter(r["source"] for r in rows)),
    }
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description='prepare zeroshot manifest')
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=1000)
    parser.add_argument("--paper-counts", action="store_true", help="Use Table 8/9 counts as defaults (five generators need 2000); source sample_count/count overrides this")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--allow-short", action="store_true")
    args = parser.parse_args()
    report_path = args.output.with_suffix(".selection.json")
    if args.output.exists() or report_path.exists():
        parser.error("Output already exists; choose a new --output to preserve the selection record")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        rows, report = build_manifest(config, args.data_root.resolve(), per_source=args.per_source, seed=args.seed, allow_missing=args.allow_missing, allow_short=args.allow_short, paper_counts=args.paper_counts)
        write_csv(args.output, MANIFEST_FIELDS, rows)
        load_manifest(args.output, args.data_root.resolve())
        report["manifest_sha256"] = file_hash(args.output)
        report["config_sha256"] = file_hash(args.config)
        write_json(report_path, report)
        print(json.dumps({"manifest": str(args.output.resolve()), **report}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Manifest preparation failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
