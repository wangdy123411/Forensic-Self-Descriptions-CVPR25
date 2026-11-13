from project_paths import CODE_ROOT

import argparse
import bisect
import csv
import hashlib
import json
import math
import sys
from pathlib import Path


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, records):
    temporary = path.with_suffix(".partial.csv")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contribution(real_score, sorted_fake):

    left = bisect.bisect_left(sorted_fake, real_score)
    right = bisect.bisect_right(sorted_fake, real_score)
    return 1.0 - (left + right) / (2 * len(sorted_fake))


def export(args):
    run = args.project / "Results/reproduction/section41_final_20261110"
    predictions = run / "available_4.1_20261109T233647238685Z/4.1/predictions.csv"
    expected = json.loads((run / "delivery.json").read_text())["files"]
    assert digest(predictions) == expected[predictions.relative_to(run).as_posix()]
    rows = read_rows(predictions)
    timm_scores = {r["image_id"]: r for r in rows if r["source"] == "ImageNet-22k"}
    glide = [{"image_id": r["image_id"], "sha256": r["sha256"], "fake_score": r["fake_score"]}
             for r in rows if r["source"] == "GLIDE" and r["label"] == "1" and r["status"] == "ok"]
    targets = []
    for item in read_rows(args.data / "real/ImageNet-22k/provenance.csv"):
        base = timm_scores[item["image_id"]]
        assert (base["status"], base["label"], base["sha256"]) == ("ok", "0", item["sha256"])
        targets.append(dict(original_id=item["original_image_id"], image_id=item["image_id"],
                            timm_path=item["relative_path"], timm_sha256=item["sha256"],
                            timm_width=item["width"], timm_height=item["height"],
                            timm_fake_score=base["fake_score"]))
    assert len(glide) == len({r["sha256"] for r in glide}) == 2000
    assert len(targets) == len(timm_scores) == len({r["original_id"] for r in targets}) == 1000
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "timm_candidates.csv", targets)
    write_rows(args.output / "glide_fixed.csv", glide)
    weights = args.project / "Weights/fsd_v1.2.0/download_manifest.json"
    assets = json.loads(weights.read_text(encoding="utf-8-sig"))["assets"]
    locked = {f: digest(args.output / f) for f in ("timm_candidates.csv", "glide_fixed.csv")}
    (args.output / "inputs.json").write_text(json.dumps(dict(predictions_sha256=digest(predictions),
        files=locked, weights={s["name"]:s["sha256"] for s in assets if s["name"] in
                            ("config.json","fre.pt","gmm.pt","fsd_transforms.pt")}), indent=2))
    fixed = sorted(float(r["fake_score"]) for r in glide)
    baseline = sum(contribution(float(r["timm_fake_score"]), fixed) for r in targets)/len(targets)
    print(f"Locked {len(targets)} timm IDs and {len(glide)} GLIDE scores. Full timm AUC = {baseline:.7f}")


def score(args):
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, str(CODE_ROOT))
    from fsd import FSDDetector
    identity = json.loads((args.bundle / "inputs.json").read_text())
    for f, value in identity["files"].items():
        if digest(args.bundle / f) != value:
            raise ValueError(f"Input file changed: {f}")
    weights = args.project / "Weights/fsd_v1.2.0"
    for name, value in identity["weights"].items():
        if digest(weights / name) != value:
            raise ValueError(f"Unexpected detection weight: {name}")
    device = torch.cuda.get_device_name(0)
    if "Blackwell" not in device:
        raise ValueError(f"Expected Blackwell, found {device}")
    targets = {r["original_id"]: r for r in read_rows(args.bundle/"timm_candidates.csv")}
    fake = sorted(float(r["fake_score"]) for r in read_rows(args.bundle/"glide_fixed.csv"))
    pairs = read_rows(args.pairs)
    ids = [r["original_id"] for r in pairs]
    if not ids or len(ids) != len(set(ids)) or not set(ids).issubset(targets):
        raise ValueError("Paired manifest must have unique original IDs from the evaluation.")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    model = FSDDetector.load(weights_dir=weights, device="cuda:0", attribution=False)
    args.output.mkdir(exist_ok=True, parents=True)
    path = args.output/"scored_pairs.csv"
    previous = {r["original_id"]:r for r in read_rows(path)} if path.exists() else {}
    results = []
    for row in pairs:
        entry = targets[row["original_id"]]
        original = args.pairs.parent / row["original_path"]
        if not row["original_source"].strip() or digest(original) != row["original_sha256"]:
            raise ValueError(f"Missing provenance or unexpected bytes: {original}")
        cached = previous.get(row["original_id"])
        if cached and cached["original_sha256"] == row["original_sha256"] and cached["reference_hash"] == identity["predictions_sha256"]:
            results.append(cached)
            continue
        with Image.open(original) as image:
            ow, oh = image.size
        with torch.inference_mode():
            original_score = -float(model.score(original).z_score)
        if not math.isfinite(original_score):
            raise ValueError(f"Nonfinite score for {original}")
        ref = float(entry["timm_fake_score"])
        results.append(dict(original_id=row["original_id"], original_source=row["original_source"],
                            original_sha256=row["original_sha256"], timm_sha256=entry["timm_sha256"],
                            original_width=ow, original_height=oh, timm_width=entry["timm_width"],
                            timm_height=entry["timm_height"], original_score=original_score,
                            timm_score=ref, original_auc_contribution=contribution(original_score, fake),
                            timm_auc_contribution=contribution(ref, fake),reference_hash=identity["predictions_sha256"]))

        previous[row["original_id"]] = results[-1]
        write_rows(path, [previous[i] for i in ids if i in previous])
    write_rows(path, results)
    orig = np.array([float(r["original_auc_contribution"]) for r in results])
    timm = np.array([float(r["timm_auc_contribution"]) for r in results])
    change = orig - timm
    summary = dict(gpu=device, n_pairs=len(pairs), fixed_glide_count=len(fake), auc_original=float(orig.mean()),
                   auc_timm_same_ids=float(timm.mean()), delta_original_minus_timm=float(change.mean()),
                   identical_bytes=sum(r["original_sha256"] == r["timm_sha256"] for r in results),
                   resized_pairs=sum((int(r["original_width"]),int(r["original_height"])) !=
                                     (int(r["timm_width"]),int(r["timm_height"])) for r in results))
    if len(change) > 1:
        rng = np.random.default_rng(2026)
        boot = [float(rng.choice(change,size=len(change),replace=True).mean()) for _ in range(5000)]
        summary["paired_real_bootstrap_95ci_fixed_glide"] = np.quantile(boot, [.025,.975]).tolist()
    (args.output / "paired_auc.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='paired imagenet22k')
    modes = parser.add_subparsers(dest="mode", required=True)
    p = modes.add_parser("export")
    for key in ("project","data","output"):
        p.add_argument("--" + key, type=Path, required=True)
    p = modes.add_parser("score")
    for key in ("project","bundle","pairs","output"):
        p.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    (export if args.mode == "export" else score)(args)
