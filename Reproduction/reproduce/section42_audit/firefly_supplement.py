import base64
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

REPOSITORY = "microsoft/MNW"
PUBLIC_DIRECTORY = "AI_Images/Adobe_Firefly_v2"
ENDPOINT = f"https://api.github.com/repos/{REPOSITORY}"


def fetch(url):
    request = Request(url, headers={"User-Agent": "CE7454-attribution-evaluation"})
    with urlopen(request, timeout=40) as response:
        return response.read()


def main(destination):
    destination.mkdir(parents=True, exist_ok=True)
    version = json.loads(fetch(ENDPOINT + "/commits/main"))["sha"]
    listing = json.loads(fetch(f"{ENDPOINT}/contents/{PUBLIC_DIRECTORY}?ref={version}"))
    entries = sorted((item for item in listing if item["name"].endswith(".jpg")), key=lambda item: item["name"])
    rows = []
    for item in entries[:3]:
        blob = json.loads(fetch(item["git_url"]))
        pointer = base64.b64decode(blob["content"]).decode("ascii")
        expected = re.search(r"oid sha256:([0-9a-f]{64})", pointer).group(1)
        expected_size = int(re.search(r"^size (\d+)", pointer, re.MULTILINE).group(1))
        url = f"https://media.githubusercontent.com/media/{REPOSITORY}/{version}/{item['path']}"
        image_bytes = fetch(url)
        if len(image_bytes) != expected_size or hashlib.sha256(image_bytes).hexdigest() != expected:
            raise ValueError(f"Git LFS size or digest differs for {item['name']}")
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            assert image.format == "JPEG"
            width, height = image.size
        (destination / item["name"]).write_bytes(image_bytes)
        rows.append(dict(image_id="mnw_"+expected[:24], source="FireFly", path=item["name"],
                         sha256=expected, package="MNW", generator="Adobe Firefly v2", url=url,
                         revision=version, width=width, height=height, file_bytes=expected_size))
    if len({r["sha256"] for r in rows}) != 3:
        raise ValueError("Three distinct Firefly images are required")
    with (destination / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(version)
    for row in rows:
        print(row["path"],row["width"],row["height"],row["file_bytes"],row["sha256"])


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
