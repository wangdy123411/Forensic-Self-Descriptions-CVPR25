"""Download official release weights to this experiment, with provenance."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
REPO = "ductai199x/Forensic-Self-Descriptions-CVPR25"
TAG = "v1.2.0"
FILES = [
    "config.json",
    "fre.pt",
    "gmm.pt",
    "fsd_transforms.pt",
    "attribution_transforms.pt",
    "source_gmms.pt",
]


def main():
    dest = ROOT / "weights" / TAG
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for name in FILES:
        path = dest / name
        url = f"https://github.com/{REPO}/releases/download/{TAG}/{name}"
        if not path.exists():
            tmp = path.with_suffix(".partial")
            for attempt in range(5):
                try:
                    req = Request(url, headers={"User-Agent": "AN-FSD-robustness-research"})
                    with urlopen(req, timeout=120) as response, tmp.open("wb") as f:
                        expected = int(response.headers.get("Content-Length", 0))
                        while chunk := response.read(1024 * 1024):
                            f.write(chunk)
                    if expected and tmp.stat().st_size != expected:
                        raise IOError("Incomplete weight download")
                    tmp.replace(path)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(dict(file=name, url=url, bytes=path.stat().st_size, sha256=digest))
        print(f"{name}: {path.stat().st_size:,} bytes sha256={digest}", flush=True)
    (ROOT / "weights_provenance.json").write_text(
        json.dumps({"repository": REPO, "release": TAG, "files": records}, indent=2)
    )
    print((dest / "config.json").read_text())


if __name__ == "__main__":
    main()
