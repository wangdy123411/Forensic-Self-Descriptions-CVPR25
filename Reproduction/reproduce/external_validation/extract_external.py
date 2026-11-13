from project_paths import PROJECT_ROOT, CODE_ROOT, DATA_ROOT

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
PROJECT = PROJECT_ROOT
DATA = DATA_ROOT
INTEGRATED = PROJECT / "Results/reproduction/section41_final_20261110/integrated"
WEIGHTS = PROJECT / "Weights/fsd_v1.2.0"
sys.path.insert(0, str(CODE_ROOT))
from fsd.fre import FRE
from fsd.fsd_computation import _solve_constrained_least_squares


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def records(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def prepare(path, fre):
    with Image.open(path) as image:
        source = torch.from_numpy(np.array(image.convert("L"))).to(
            device=fre.device, dtype=torch.float64)
    kernel = fre.conv.constrain(fre.conv.w.double(), fre.conv.one_middle.double())
    size = kernel.shape[-1]
    half = size // 2
    h, w = source.shape
    residual = torch.empty((8, h-size+1, w-size+1), device=source.device, dtype=source.dtype)
    for y in range(0, residual.shape[1], 64):
        n = min(64, residual.shape[1]-y)
        block = source[y:y+n+size-1][None, None]
        residual[:, y:y+n] = source[y+half:y+half+n, half:-half] - F.conv2d(block, kernel)[0]
    h, w = residual.shape[-2:]
    factor = 1024 / min(h, w)
    resized = F.interpolate(residual[None], size=(round(h*factor), round(w*factor)), mode="bilinear",
                            antialias=False, align_corners=False)[0]
    y, x = ((size-1024)//2 for size in resized.shape[-2:])
    return resized[:, y:y+1024, x:x+1024].contiguous()


def patch_gram(r):

    channels, height, width = r.shape
    device = r.device
    offsets = torch.arange(11, device=device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    oy, ox = oy.flatten(), ox.flatten()
    dy = (oy[None, :]-oy[:, None]) % height
    dx = (ox[None, :]-ox[:, None]) % width
    freq = torch.fft.rfft2(r)
    blocks = r.new_empty((channels, 121, channels, 121))
    for i in range(channels):
        corr = torch.fft.irfft2(freq[i].conj()[None]*freq[i:], s=(height, width))
        gram = corr[:, dy, dx]
        blocks[i, :, i:, :] = gram.permute(1, 0, 2)
        if i+1 < channels:
            blocks[i+1:, :, i, :] = gram[1:].transpose(1, 2)
    result = blocks.reshape(968, 968)
    del blocks, freq, corr, gram

    interior = torch.ones((height, width), device=device, dtype=torch.bool)
    interior[5:-5, 5:-5] = False
    yy, xx = torch.where(interior)
    for mode, sign in (("circular", -1), ("reflect", 1)):
        padded = F.pad(r[None], (5, 5, 5, 5), mode=mode)[0]
        windows = padded.unfold(1, 11, 1).unfold(2, 11, 1)
        for start in range(0, len(yy), 2048):
            rows = windows[:, yy[start:start+2048], xx[start:start+2048]].permute(1, 0, 2, 3)
            rows = rows.reshape(-1, 968).contiguous()
            result.addmm_(rows.T, rows, alpha=sign)
    return result


@torch.inference_mode()
def extract(path, fre):
    r = prepare(path, fre)
    gram = patch_gram(r)
    for scale in (0.5, 0.25):
        smaller = F.interpolate(r[None], scale_factor=scale, mode="bilinear",
                                antialias=False, align_corners=False)[0]
        gram += patch_gram(smaller)
    center = torch.arange(8, device=r.device)*121+60
    keep = torch.arange(968, device=r.device) % 121 != 60
    normal = gram[keep][:, keep]
    normal.diagonal().add_(1e-5)
    rhs = gram[keep][:, center].sum(1)
    lhs = r.new_zeros((968, 968))
    lhs[:960, :960] = normal
    for i in range(8):
        lhs[960+i, 120*i:120*(i+1)] = 1
        lhs[120*i:120*(i+1), 960+i] = 1
    return torch.linalg.solve(lhs, torch.cat([rhs, r.new_ones(8)]))[:960].cpu().numpy()


@torch.inference_mode()
def direct_reference(path, fre):

    residual = prepare(path, fre)
    mask = torch.ones((11, 11), device=fre.device, dtype=torch.bool)
    mask[5, 5] = False

    def tiles():
        for scale in (1.0, 0.5, 0.25):
            resized = F.interpolate(
                residual[None], scale_factor=scale, mode="bilinear",
                antialias=False, align_corners=False,
            )[0]
            padded = F.pad(resized[None], (5, 5, 5, 5), mode="reflect")[0]
            view = padded.unfold(1, 11, 1).unfold(2, 11, 1)
            rows = 4096 // resized.shape[-1]
            for start in range(0, resized.shape[-2], rows):
                yield view[:, start:start + rows].reshape(8, -1, 11, 11).permute(1, 0, 2, 3)

    return _solve_constrained_least_squares(
        tiles(), mask, 5, 8, 120, fre.device
    ).cpu().numpy()


def initialize(device, threads):
    if digest(WEIGHTS/"fre.pt") != "d95b9c50837dbf7b660bbefa20cdaa5db5e59601a9d6544573c10e78e04906bb":
        raise ValueError("FRE weight digest mismatch")
    if digest(WEIGHTS/"config.json") != "7cc34433045adb998762e00de7de25c50f9c1e10dbac1c18899c6c63c4cfafe4":
        raise ValueError("FSD configuration digest mismatch")
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return FRE.from_pretrained(WEIGHTS/"fre.pt", device=device)


def calibrate(fre, direct=False):
    index = {r["image_id"]: r for r in records(INTEGRATED/"feature_index.csv")}
    all_features = np.load(INTEGRATED/"raw_fsd_960.npy", mmap_mode="r")
    rows = records(DATA/"manifests/eval_unique.csv")
    result = []
    for name in ("GigaGAN", "FireFly"):
        row = next(r for r in rows if r["source"] == name)
        example = DATA/row["relative_path"]
        if digest(example) != row["sha256"] or index[row["image_id"]]["image_sha256"] != row["sha256"]:
            raise ValueError("Calibration input digest mismatch")
        start = time.perf_counter()
        vector = direct_reference(example, fre) if direct else extract(example, fre)
        reference = all_features[int(index[row["image_id"]]["row"])]
        rel = np.linalg.norm(vector-reference)/np.linalg.norm(reference)
        maximum = abs(vector-reference).max()
        record = dict(source=name, image_id=row["image_id"], image_sha256=row["sha256"],
                      rel_l2=float(rel), max_abs=float(maximum), seconds=time.perf_counter()-start)
        print(json.dumps(record), flush=True)
        result.append(record)
        if not np.isfinite(vector).all() or rel >= 1e-6:
            raise ValueError("Archived feature calibration failed")
    return result


def calibrate_external_shapes(fre, cache):
    index = {r["image_id"]: r for r in records(INTEGRATED/"feature_index.csv")}
    provenance = records(DATA/"manifests/provenance.csv")
    historic = next(r for r in provenance if "GigaGAN_t2i_coco256_rep/" in r["original_relative_path"])
    external_manifest = HERE/"input/firefly/selection.csv"
    external = records(external_manifest)[0]
    gold = np.load(INTEGRATED/"raw_fsd_960.npy", mmap_mode="r")
    cases = (
        ("archived_GigaGAN_COCO", historic, DATA/historic["relative_path"], extract,
         gold[int(index[historic["image_id"]]["row"])]),
        ("MNW_Firefly_v2", external, external_manifest.parent/external["path"], direct_reference,
         np.load(cache/(external["sha256"]+".npy"))),
    )
    evidence = []
    for name, row, path, compute, expected in cases:
        if digest(path) != row["sha256"]:
            raise ValueError(f"Verification digest mismatch: {path}")
        with Image.open(path) as image:
            width, height = image.size
        start = time.perf_counter()
        vector = compute(path, fre)
        error = np.linalg.norm(vector-expected)/np.linalg.norm(expected)
        result = dict(source=name, image_id=row["image_id"], image_sha256=row["sha256"],
                      width=width, height=height, method=compute.__name__, rel_l2=float(error),
                      max_abs=float(abs(vector-expected).max()), seconds=time.perf_counter()-start)
        evidence.append(result)
        print(json.dumps(result), flush=True)
        if not np.isfinite(error) or error >= 1e-6:
            raise ValueError("Additional shape calibration failed")
    (HERE / f"calibration_extended_{fre.device.type}.json").write_text(json.dumps(evidence, indent=2))


def save_vector(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as stream:
        temporary = Path(stream.name)
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description='extract external')
    parser.add_argument("manifests", nargs="*", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", default=6, type=int)
    parser.add_argument("--cache", type=Path, default=HERE/"features")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--direct-calibration", action="store_true")
    parser.add_argument("--extra-calibration", action="store_true")
    args = parser.parse_args()
    fre = initialize(args.device, args.threads)
    if args.extra_calibration:
        calibrate_external_shapes(fre, args.cache)
    if args.calibrate or args.direct_calibration:
        calibration = calibrate(fre, direct=args.direct_calibration)
        method = "tiled_author_" if args.direct_calibration else ""
        (HERE/("calibration_"+method+args.device+".json")).write_text(json.dumps(calibration, indent=2))
    for manifest in args.manifests:
        args.cache.mkdir(parents=True, exist_ok=True)
        rows = records(manifest)
        for i, row in enumerate(rows):
            path = manifest.parent/row["path"]
            if digest(path) != row["sha256"]:
                raise ValueError(f"Input SHA256 mismatch: {path}")
            output = args.cache/(row["sha256"]+".npy")
            if output.is_file():
                x = np.load(output, allow_pickle=False)
                if x.shape != (960,) or x.dtype != np.float64 or not np.isfinite(x).all():
                    raise ValueError(f"Invalid existing cache: {output}")
                continue
            start = time.perf_counter()
            x = extract(path, fre)
            if x.shape != (960,) or x.dtype != np.float64 or not np.isfinite(x).all():
                raise ValueError(f"Invalid extracted vector: {path}")
            save_vector(output, x)
            print(f"{i+1}/{len(rows)} {row['image_id']} {time.perf_counter()-start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
