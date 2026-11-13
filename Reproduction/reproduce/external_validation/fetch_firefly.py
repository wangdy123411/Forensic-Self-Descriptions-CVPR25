from project_paths import DATA_ROOT

import csv
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from PIL import Image


PRIOR = DATA_ROOT
DEST = Path(__file__).resolve().parent / "input" / "firefly"
REVISION = "52c83b352baf31b77dca7a8ee569e6cffc09c156"
REPO = "microsoft/MNW"
DIRECTORY = "AI_Images/Adobe_Firefly_v2"
BYTE_LIMIT = 8 * 1024 * 1024
SEED = "ce7454-mnw-firefly-v2-external-2026"


def fetch(url):
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "CE7454-academic-evaluation"})
            with urlopen(req, timeout=35) as response:
                data = response.read(BYTE_LIMIT + 1)
            if len(data) > BYTE_LIMIT:
                raise ValueError("Source response exceeds 8 MiB.")
            return data
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == 2 or (isinstance(error, HTTPError) and error.code in {401, 403, 404}):
                raise
            time.sleep(1 + attempt)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def acquire(item, excluded):
    name = item["name"]
    path = DEST / name
    url = f"https://media.githubusercontent.com/media/{REPO}/{REVISION}/{quote(item['path'])}"
    data = path.read_bytes() if path.exists() else fetch(url)
    digest = hashlib.sha256(data).hexdigest()
    size = len(data)
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {size}\n".encode()
    blob_header = b"blob " + str(len(pointer)).encode() + b"\x00"
    if hashlib.sha1(blob_header + pointer).hexdigest() != item["sha"]:
        raise ValueError(f"Git LFS pointer does not match official manifest: {name}")
    if digest in excluded:
        raise ValueError(f"Exact image already present in project: {name}")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.format != "JPEG":
            raise ValueError(f"Expected JPEG: {name}")
        width, height = image.size
    if not path.exists():
        temp = path.with_suffix(".tmp")
        temp.write_bytes(data)
        temp.replace(path)
    return dict(image_id="mnw_" + digest[:24], source="FireFly", path=name, sha256=digest,
                package="MNW", generator="Adobe Firefly", version="v2", url=url, revision=REVISION,
                git_lfs_pointer_blob=item["sha"], width=width, height=height, file_bytes=size)


def main():
    old = read_rows(PRIOR / "supplemental" / "MNW_Firefly_v2_3" / "manifest.csv")
    if len(old) != 3 or {row["revision"] for row in old} != {REVISION}:
        raise ValueError("Prior three-image Firefly supplement differs from the expected revision.")
    excluded = {row["sha256"] for row in old}
    inventory = read_rows(PRIOR / "manifests" / "inventory_all.csv")
    excluded.update(row["sha256"] for row in inventory)
    old_names = {row["path"] for row in old}
    url = f"https://api.github.com/repos/{REPO}/contents/{DIRECTORY}?ref={REVISION}"
    listing = json.loads(fetch(url))
    pool = [item for item in listing if item["name"].endswith(".jpg") and item["name"] not in old_names]
    selected = sorted(pool, key=lambda item: hashlib.sha256((SEED + "/" + item["path"]).encode()).digest())[:100]
    if len(selected) != 100:
        raise ValueError("100 eligible samples required.")
    DEST.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(acquire, item, excluded) for item in selected]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 10 == 0:
                print(f"Verified {len(rows)} / 100", flush=True)
    rows.sort(key=lambda row: row["path"])
    if len({row["sha256"] for row in rows}) != 100:
        raise ValueError("Duplicate within cohort.")
    tmp = DEST / "selection.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(DEST / "selection.csv")
    print(f"{len(rows)} held-out JPEGs, {sum(row['file_bytes'] for row in rows)} bytes; "
          f"checked against {len(excluded)} SHA-256 hashes.", flush=True)


if __name__ == "__main__":
    main()
