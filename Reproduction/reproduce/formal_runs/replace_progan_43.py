import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from PIL import Image
import torch

from formal_42 import ROOT, PARENT, load_merged, read_csv
from clustering import cluster
from inputs import CODE, WEIGHTS, Samples
from feature_cache import FeatureCache
from fsd import FSDDetector
from zeroshot_common import child_path, file_hash, load_manifest, write_csv, write_json

NAME = "section43_ossia_progan_20261110"
INPUT = ROOT / "Reproduction" / "ossia_progan_test_1000"
OUTPUT = PARENT / NAME
BASELINE = PARENT / "section43_final_20261110" / "4.3"


def status(phase, **details):
    record = dict(phase=phase, utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), **details)
    write_json(OUTPUT / "status.json", record)
    print(json.dumps(record), flush=True)


def check_extractor():

    parent = ROOT / "Results/parallel_blackwell/style3/run_metadata.json"
    identity = json.loads(parent.read_text())["identity"]["feature_cache"]["extraction_identity"]
    config = json.loads((WEIGHTS / "config.json").read_text())
    assert config["fre"] == identity["fre_config"] and config["fsd"] == identity["fsd_config"]
    assert file_hash(WEIGHTS / config["fre"]["weights_file"]) == identity["fre_weights_sha256"]
    assert all(file_hash(CODE / "fsd" / name) == digest for name, digest in identity["extractor_code_sha256"].items())
    assert identity["dimensions"] == 960 and identity["dtype"] == "<f8"
    assert str(torch.__version__) == identity["torch_version"] and torch.version.cuda == identity["cuda_version"]
    assert torch.cuda.get_device_name(0) == identity["gpu"] and "Blackwell" in identity["gpu"]
    assert all(importlib.metadata.version(n) == v for n, v in identity["packages"].items())
    return identity


def extract(rows, identity):
    start = time.monotonic()
    new = 0
    vectors = []
    with FeatureCache(OUTPUT / "checkpoint", identity) as cache:
        model = FSDDetector.load(weights_dir=WEIGHTS, device="cuda:0", attribution=False)
        for position, row in enumerate(rows, start=1):
            path = child_path(INPUT / "data", row["relative_path"])
            with Image.open(path) as image:
                if image.size != (256, 256):
                    raise ValueError(f"Unexpected native size for {row['image_id']}: {image.size}")
            v = cache.get(row)
            if v is None:
                with torch.inference_mode():
                    v = model.compute_fsd(path).detach().cpu().numpy()
                cache.put(row, v)
                new += 1
            vectors.append(v)
            if position == 1 or position % 25 == 0:
                status("extracting", completed=position, total=len(rows), extracted_this_session=new,
                       seconds_per_new_image=(time.monotonic()-start)/new if new else None)
        cache.integrity_check()
        if cache.coverage(rows, verify_payloads=True)["missing_count"]:
            raise ValueError("Replacement FSDs incomplete")
        return np.asarray(vectors, dtype="<f8"), cache.signature


def publish(rows, new_vectors, signature, identity):
    base, base_matrix_sha = load_merged()
    if file_hash(BASELINE / "assignments.csv") != json.loads((BASELINE / "checksums.json").read_text())["assignments.csv"]:
        raise ValueError("Baseline assignments changed")
    original = read_csv(BASELINE / "assignments.csv")
    indices = {row["image_id"]: i for i, row in enumerate(base.rows)}
    active = [indices[r["image_id"]] for r in original]
    old_rows = [base.rows[i] for i in active]
    old_features = base.features[active]
    old_pg = [r for r in old_rows if r["source"] == "ProGAN"]
    kept = [r for r in old_rows if r["source"] != "ProGAN"]
    if (len(rows), len(old_pg), len(kept)) != (1000, 1000, 6997):
        raise ValueError("Invalid replacement cohort size")
    if {r["sha256"] for r in kept} & {r["sha256"] for r in rows}:
        raise ValueError("A replacement image also occurs among retained sources")
    chosen = iter(zip(rows, new_vectors))
    mixed = [next(chosen) if r["source"] == "ProGAN" else (r,v) for r,v in zip(old_rows,old_features)]
    mrows = [r for r,_ in mixed]
    matrix = np.ascontiguousarray(np.stack([v for _,v in mixed]), dtype="<f8")
    if len({r["image_id"] for r in mrows}) != len(mrows):
        raise ValueError("Duplicate combined image ID")
    matrix_digest = hashlib.sha256(memoryview(matrix).cast("B")).hexdigest()
    with tempfile.TemporaryDirectory(dir=OUTPUT) as tmp:
        out = Path(tmp) / "4.3"
        out.mkdir()
        index = [dict(row=i,image_id=r["image_id"],source=r["source"],image_sha256=r["sha256"],
                      feature_sha256=hashlib.sha256(v.tobytes()).hexdigest()) for i,(r,v) in enumerate(zip(mrows,matrix))]
        write_csv(out / "input_index.csv", tuple(index[0]), index)
        shutil.copyfile(INPUT / "manifest.csv", out / "replacement_manifest.csv")
        np.save(out / "replacement_features.npy",new_vectors,allow_pickle=False)
        status("clustering", k=[6,12,24], n=len(matrix))
        samples=Samples(mrows,matrix,mrows,[])
        cluster(samples,out,"cuda:0",2026,n_init=10)
        output_rows=read_csv(out / "assignments.csv")
        if [r["image_id"] for r in output_rows]!=[r["image_id"] for r in mrows]:
            raise ValueError("Cluster assignments do not match the input index")
        metrics=json.loads((out/"metrics.json").read_text())
        assert [r["observed_k"] for r in metrics["metrics"]]==[6,12,24]
        write_json(out/"run.json",dict(gpu=identity["gpu"],original_integrated_npy_sha256=base_matrix_sha,
                   input_matrix_bytes_sha256=matrix_digest,baseline_assignment_sha256=file_hash(BASELINE/"assignments.csv"),
                   source_counts=dict(Counter(r["source"] for r in mrows)),unchanged_rows=len(kept),
                   old_new_progan_image_sha256_overlap=len({r["sha256"] for r in old_pg}&{r["sha256"] for r in rows}),
                   replacement_manifest_sha256=file_hash(INPUT/"manifest.csv"),replacement_fsd_signature=signature,
                   feature_extraction_identity=identity,seed=2026,n_init=10,representation="raw 960-D float64 FSD"))
        write_json(out/"checksums.json",{f.name:file_hash(f) for f in out.iterdir() if f.is_file()})
        out.rename(OUTPUT/"4.3")


def main():
    OUTPUT.mkdir(exist_ok=True, parents=True)
    with (OUTPUT / "run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUTPUT/"4.3/checksums.json").exists():
            status("completed")
            return
        try:
            torch.set_num_threads(4)
            torch.manual_seed(2026)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False
            status("verifying_ossia_images")
            rows=load_manifest(INPUT/"manifest.csv",INPUT/"data",verify_hashes=True)
            if len(rows)!=1000 or any(r["source"]!="ProGAN" or r["split"]!="test" or r["label"]!="1" for r in rows):
                raise ValueError("Expected exactly 1,000 unique ProGAN test images")
            identity=check_extractor()
            vectors,signature=extract(rows,identity)
            publish(rows,vectors,signature,identity)
            status("completed",n_sources=6,n_images=7997,n_new=1000)
        except BaseException as exc:
            status("failed",error=repr(exc))
            raise


if __name__=="__main__":
    main()
