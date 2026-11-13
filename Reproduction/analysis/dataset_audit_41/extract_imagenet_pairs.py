import argparse
import csv
import hashlib
import io
import re
import tarfile
from pathlib import Path

from PIL import Image, ImageOps, ImageStat, ImageChops

from paired_imagenet22k import read_rows, write_rows


def thumbnail(image):
    return ImageOps.exif_transpose(image).convert("RGB").resize((48, 48), Image.Resampling.LANCZOS)


def collect(archive, candidates, data, output, source, ancestor=""):
    pairs = []
    misses = []
    for member in archive:
        if not member.isfile():
            continue
        name = Path(member.name)
        if name.suffix.lower()==".tar" and name.stem in {s.partition("_")[0] for s in candidates}:
            with archive.extractfile(member) as stream, tarfile.open(fileobj=stream, mode="r:") as nested:
                more, skipped = collect(nested, candidates, data, output, source, member.name+"::")
                pairs.extend(more)
                misses.extend(skipped)
        if name.suffix.lower() not in (".jpeg",".jpg",".png") or name.stem not in candidates:
            continue
        entry = candidates[name.stem]
        with archive.extractfile(member) as stream:
            raw = stream.read()
        timm_file = data/entry["timm_path"]
        if hashlib.sha256(timm_file.read_bytes()).hexdigest()!=entry["timm_sha256"]:
            raise ValueError(f"Cached timm file changed: {timm_file}")
        with Image.open(io.BytesIO(raw)) as ori, Image.open(timm_file) as processed:
            a,b = thumbnail(ori),thumbnail(processed)
            rms=(sum(x*x for x in ImageStat.Stat(ImageChops.difference(a,b)).rms)/3)**0.5

        if rms > 12:
            misses.append(dict(original_id=name.stem, archive_member=ancestor+member.name, thumbnail_rmse=rms))
            continue
        suffix = name.suffix.lower()
        relative = Path("originals")/(name.stem+suffix)
        dest=output/relative
        sha=hashlib.sha256(raw).hexdigest()
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=sha:
            raise ValueError(f"Two different original files for {name.stem}")
        dest.write_bytes(raw)
        pairs.append(dict(original_id=name.stem,original_path=relative.as_posix(),original_sha256=sha,
                          original_source=source,archive_member=ancestor+member.name,thumbnail_rmse=rms))
    return pairs, misses


if __name__=="__main__":
    parser=argparse.ArgumentParser(description='extract imagenet pairs')
    parser.add_argument("--tar",type=Path,nargs="+",required=True)
    parser.add_argument("--source",required=True,help="Acquisition source and release of these archives")
    parser.add_argument("--data",type=Path,required=True,help="7454data root containing timm JPEGs")
    parser.add_argument("--candidates",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    candidates={r["original_id"]:r for r in read_rows(args.candidates)}
    if not all(re.fullmatch(r"n\d{8}_\d+", key) for key in candidates):
        raise ValueError("Unexpected ImageNet ID")
    (args.output/"originals").mkdir(parents=True,exist_ok=True)
    pairs=[]
    review=[]
    for path in args.tar:
        with tarfile.open(path, mode="r:*") as tar:
            good, flag=collect(tar,candidates,args.data,args.output,args.source+" / "+path.name)
            pairs.extend(good)
            review.extend(flag)
    if review:
        write_rows(args.output/"review.csv",review)
    if not pairs:
        raise SystemExit("No confirmed image pairs in these archives.")
    if len({p["original_id"] for p in pairs})!=len(pairs):
        raise ValueError("Duplicate IDs across original archives")
    write_rows(args.output/"pairs.csv",sorted(pairs,key=lambda p:p["original_id"]))
    print(f"Verified {len(pairs)} ID / content pairs. {len(review)} require visual review.")
