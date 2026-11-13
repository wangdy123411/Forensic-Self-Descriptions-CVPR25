"""Package the AN reports, evidence, and source without copying large runtime data.

Run from the repository root with ``an/.venv/Scripts/python.exe an/export_reports.py``.
The output stays under ``an/reports``. Existing experiment artifacts are read only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

AN = Path(__file__).resolve().parent
OUTPUT = AN / "reports" / "AN_Robustness.zip"
DIRECTORIES = (
    "docs",
    "robustness",
    "tests",
    "paper_cvpr2026",
    "baseline_comparison",
    "results",
    "provenance",
    "reports",
)
EXCLUDED_PARTS = {"__pycache__", "build", "features", "examples", "previews"}
SUFFIXES = {
    ".py",
    ".md",
    ".html",
    ".txt",
    ".json",
    ".jsonl",
    ".csv",
    ".png",
    ".pdf",
    ".zip",
    ".tex",
    ".sty",
    ".bst",
    ".bib",
    ".sto",
    ".tlpobj",
    ".log",
    ".patch",
}


def report_files() -> list[Path]:
    """Select source and reporting artifacts while excluding images and feature caches."""
    candidates = {p for p in AN.iterdir() if p.is_file() and p.suffix in SUFFIXES}
    candidates.update(p for p in (AN / "data").iterdir() if p.suffix in {".csv", ".json"})
    candidates.update((AN / "logs").glob("*.log"))
    for directory in DIRECTORIES:
        for path in (AN / directory).rglob("*"):
            relative = path.relative_to(AN)
            if (
                path.is_file()
                and path.suffix in SUFFIXES
                and not EXCLUDED_PARTS.intersection(relative.parts)
                and path != OUTPUT
                and relative.as_posix() != "provenance/delivery_verification.json"
            ):
                candidates.add(path)
    return sorted(candidates, key=lambda path: path.relative_to(AN).as_posix())


def main() -> None:
    OUTPUT.parent.mkdir(exist_ok=True)
    temporary = OUTPUT.with_suffix(".zip.tmp")
    inventory = []
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in report_files():
            relative = path.relative_to(AN).as_posix()
            content = path.read_bytes()
            archive.writestr(f"AN_Robustness/{relative}", content)
            inventory.append(
                {
                    "path": relative,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        archive.writestr(
            "AN_Robustness/START_HERE.html",
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            "<title>AN robustness reports</title>"
            '<meta http-equiv="refresh" content="0;url=reports/index.html">'
            '<p><a href="reports/index.html">Open the AN report collection</a>.</p></html>',
        )
        archive.writestr(
            "AN_Robustness/MANIFEST.json",
            json.dumps(
                {
                    "description": "Report package; paths and hashes are relative to AN_Robustness/.",
                    "excludes": [
                        "original images",
                        "download archives",
                        "weights",
                        "virtual environment",
                        "per-image feature arrays",
                        "temporary build and render outputs",
                    ],
                    "files": inventory,
                },
                indent=2,
            )
            + "\n",
        )
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Report package failed its ZIP integrity check")
        for item in inventory:
            content = archive.read(f"AN_Robustness/{item['path']}")
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError(f"Report package hash mismatch: {item['path']}")
    temporary.replace(OUTPUT)
    print(
        json.dumps({"output": str(OUTPUT), "files": len(inventory), "bytes": OUTPUT.stat().st_size})
    )


if __name__ == "__main__":
    main()
