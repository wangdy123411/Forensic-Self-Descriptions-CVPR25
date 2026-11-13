from project_paths import DATA_ROOT, ARCHIVE_ROOT

import csv
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


HERE = Path(__file__).resolve().parent
OLD = DATA_ROOT
ORIGINAL = ARCHIVE_ROOT
OUT = HERE / "overlap"
RESAMPLE = Image.Resampling.LANCZOS
X = np.arange(32) + 0.5
DCT = np.cos(np.pi * np.outer(np.arange(8), X) / 32)
DCT[0] /= np.sqrt(2)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def pixel_features(image):
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    thumb = np.asarray(rgb.resize((32, 32), RESAMPLE), dtype=np.int16)
    gray = rgb.convert("L")
    d = np.asarray(gray.resize((9, 8), RESAMPLE))
    d_bits = d[:, 1:] > d[:, :-1]
    freq = (DCT @ np.asarray(gray.resize((32, 32), RESAMPLE)) @ DCT.T).flatten()[1:]
    p_bits = freq > np.median(freq)
    return dict(shape=rgb.size, pixel=digest, thumb=thumb,
                dh=int.from_bytes(np.packbits(d_bits).tobytes(), "big"),
                ph=int.from_bytes(np.packbits(p_bits).tobytes(), "big"))


def fingerprint(item):
    data = item["physical"].read_bytes()
    if hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise ValueError(f"File SHA differs: {item['physical']}")
    with Image.open(io.BytesIO(data)) as image:
        return {**item, **pixel_features(image), "metadata_keys": sorted(image.info)}


def distances(a, b):
    return ((a["ph"] ^ b["ph"]).bit_count(), (a["dh"] ^ b["dh"]).bit_count(),
            float(np.abs(a["thumb"] - b["thumb"]).mean()))


def judge(a, b):
    ph, dh, error = distances(a, b)
    if a["sha256"] == b["sha256"]:
        kind = "same_bytes"
    elif a["shape"] == b["shape"] and a["pixel"] == b["pixel"]:
        kind = "same_decoded_rgb"
    elif a["family"] and a["family"] == b["family"]:
        kind = "same_source_family_id"
    elif ((ph <= 8 or dh <= 6) and error <= 9.0):
        kind = "near_duplicate_candidate"
    else:
        kind = ""
    return kind, ph, dh, error


def family(path, stem):
    if "GigaGAN_cond_imagenet256" in path:
        return "GigaGAN/ImageNet/" + stem
    if "GigaGAN_t2i_coco256" in path:
        return "GigaGAN/COCO/" + stem
    if "Adobe_Firefly_v2" in path:
        return "MNW/v2/" + stem
    return ""


def prepare():
    inventory = read_csv(OLD / "manifests" / "inventory_all.csv")
    provenance = read_csv(OLD / "manifests" / "provenance.csv")
    old = []
    sources = {"FireFly": "firefly", "GigaGAN": "gigagan"}
    prior_hashes = {row["sha256"]: row["image_id"] for row in inventory}
    for row in provenance:
        source = sources.get(row["source"])
        if not source:
            continue
        origin = row["original_relative_path"]
        path = ORIGINAL / origin
        if not path.is_file():
            path = OLD / row["relative_path"]
        old.append(dict(source=source, image_id=row["image_id"], physical=path,
                        sha256=row["sha256"], family=family(origin, Path(origin).stem), scope="old"))
    prior_mnw = OLD / "supplemental" / "MNW_Firefly_v2_3"
    for row in read_csv(prior_mnw / "manifest.csv"):
        old.append(dict(source="firefly", image_id=row["image_id"], physical=prior_mnw / row["path"],
                        sha256=row["sha256"], family=family(row["path"], Path(row["path"]).stem), scope="old"))
        prior_hashes[row["sha256"]] = row["image_id"]
    new = []
    for source in sources.values():
        folder = HERE / "input" / source
        for row in read_csv(folder / "selection.csv"):
            path = folder / row["path"]
            src_family = family(row["path"], row.get("historical_family_stem", path.stem))
            new.append(dict(source=source, image_id=row["image_id"], physical=path, sha256=row["sha256"],
                            family=src_family, scope="new"))
    return old, new, prior_hashes, len(inventory)


def positive_controls(items):
    checks = []
    for source in sorted({item["source"] for item in items}):
        for item in [i for i in items if i["source"] == source][:5]:
            with Image.open(item["physical"]) as original:
                img = ImageOps.exif_transpose(original).convert("RGB")
                for variant in ["jpeg75", "resize_half_jpeg75"]:
                    copy = img if variant == "jpeg75" else img.resize((img.width // 2, img.height // 2), RESAMPLE)
                    buf = io.BytesIO()
                    copy.save(buf, format="JPEG", quality=75)
                    buf.seek(0)
                    with Image.open(buf) as changed:
                        ph, dh, error = distances(item, pixel_features(changed))
                    checks.append(dict(source=source, image_id=item["image_id"], transform=variant,
                                       phash_distance=ph, dhash_distance=dh, thumbnail_mae=round(error, 3),
                                       detected=(ph <= 8 or dh <= 6) and error <= 9.0))
    return checks


def main():
    old_rows, new_rows, old_hashes, inventory_count = prepare()
    with ThreadPoolExecutor(max_workers=6) as pool:
        old = list(pool.map(fingerprint, old_rows))
        new = list(pool.map(fingerprint, new_rows))
    OUT.mkdir(exist_ok=True)
    summary = []
    matches = []
    for i, item in enumerate(new):
        nearest = None
        nearest_prior = None
        flags = []
        if item["sha256"] in old_hashes:
            flags.append("same_bytes_full_inventory/" + old_hashes[item["sha256"]])
        for target in [*old, *new[:i]]:
            if target["scope"] == "old" and target["source"] != item["source"]:
                continue
            kind, ph, dh, error = judge(item, target)
            values = (ph, dh, round(error, 3), target["image_id"])
            if nearest is None or values < nearest:
                nearest = values
            if target["scope"] == "old" and (nearest_prior is None or values < nearest_prior):
                nearest_prior = values
            if kind:
                flags.append(kind + "/" + target["image_id"])
                matches.append(dict(image_id=item["image_id"], target_id=target["image_id"],
                                    scope=target["scope"], reason=kind, ph=ph, dh=dh, color_mae=round(error, 3)))
        summary.append(dict(source=item["source"], image_id=item["image_id"], sha256=item["sha256"],
                            eligible=int(not flags), overlap_flags="|".join(flags),
                            nearest_phash=nearest[0], nearest_dhash=nearest[1], nearest_color_mae=nearest[2],
                            nearest_id=nearest[3], prior_nearest_phash=nearest_prior[0],
                            prior_nearest_dhash=nearest_prior[1], prior_nearest_color_mae=nearest_prior[2],
                            prior_nearest_id=nearest_prior[3]))
    path = OUT / "overlap.csv"
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)
    tmp.replace(path)
    controls = positive_controls(new)
    result = dict(inventory_rows=inventory_count, distinct_inventory_and_supplement_hashes=len(old_hashes),
                  decoded_old_per_source={s: sum(row["source"] == s for row in old) for s in ("firefly", "gigagan")},
                  eligible_per_source={s: sum(row["eligible"] for row in summary if row["source"] == s)
                                       for s in ("firefly", "gigagan")},
                  decoded_new_per_source={s: sum(row["source"] == s for row in new) for s in ("firefly", "gigagan")},
                  possible_matches=matches, transforms_positive_control=controls,
                  firefly_jpeg_metadata_keys=sorted({k for row in new if row["source"] == "firefly" for k in row["metadata_keys"]}),
                  method="Verify file SHA-256 and same-generator source-family IDs; compare EXIF-oriented full RGB pixels, "
                         "63-bit pHash, 64-bit dHash and 32x32 RGB error. Compare all new-new pairs across either source, "
                         "all old-new same-source pairs. Near duplicate candidate if (pHash<=8 or dHash<=6) AND RGB MAE<=9/255. "
                         "Exact bytes checked against project-wide inventory plus prior 3 MNW. Perceptual screening does not "
                         "guarantee detection of crops, large edits, or shared prompts.")
    (OUT / "overlap.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result["eligible_per_source"], result["decoded_old_per_source"], "matches", len(matches),
          "positive controls", sum(row["detected"] for row in controls), "/", len(controls), flush=True)


if __name__ == "__main__":
    main()
