from project_paths import PROJECT_ROOT, EXTENSION_ROOT

from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = PROJECT_ROOT
sys.path.insert(0, str(EXTENSION_ROOT / "reproduce"))
from inputs import INPUTS, PRODUCTION_HASH, default_shards, load_samples
from zeroshot_common import PREDICTION_FIELDS, file_hash, write_csv, write_json

OUT = ROOT / "Results/reproduction/section41_final_20261110"


def status(phase, **extra):
    write_json(OUT/"job.json", dict(phase=phase, pid=os.getpid(),
               utc=datetime.now(timezone.utc).isoformat(), **extra))
    print(phase, flush=True)


def integrate():
    manifest = INPUTS/"eval_unique.csv"
    sources = default_shards(False)
    samples = load_samples(manifest, sources, preview=False)
    if file_hash(manifest) != PRODUCTION_HASH or samples.features.shape != (28992, 960):
        raise ValueError("Expected the original manifest and 28,992 x 960 raw vectors")
    origins = {}
    for source, metadata in zip(sources, samples.provenance):
        with (source/"predictions.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                origins[row["image_id"]] = metadata["signature"]
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        folder=Path(tmp)/"integrated"
        folder.mkdir()
        np.save(folder/"raw_fsd_960.npy", samples.features.astype("<f8",copy=False), allow_pickle=False)
        index = [dict(row=i,image_id=r["image_id"],source=r["source"],label=r["label"],image_sha256=r["sha256"],
                      feature_sha256=hashlib.sha256(x.tobytes()).hexdigest(),extraction_signature=origins[r["image_id"]])
                 for i,(r,x) in enumerate(zip(samples.rows,samples.features))]
        write_csv(folder/"feature_index.csv",tuple(index[0]),index)
        write_csv(folder/"original_predictions.csv",PREDICTION_FIELDS,samples.rows)
        write_json(folder/"provenance.json",dict(manifest_sha256=PRODUCTION_HASH,shape=list(samples.features.shape),
                   dtype="<f8",row_order="eval_unique.csv",counts=dict(Counter(r["source"] for r in samples.rows)),
                   shards=samples.provenance))
        folder.rename(OUT/"integrated")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT/"delivery.json").is_file():
            return
        try:
            if not (OUT/"integrated").exists():
                status("validating_and_integrating")
                integrate()
            status("running_section_4.1", count=28992)
            previous=list(OUT.glob("available_4.1_*"))
            if not previous:
                subprocess.run([sys.executable, str(EXTENSION_ROOT/"run.py"),"reproduce","4.1","--output-parent",str(OUT)],
                               cwd=ROOT,check=True)
            run,=OUT.glob("available_4.1_*")
            section=run/"4.1"
            metrics=json.loads((section/"metrics.json").read_text())
            if metrics["scored_count"]!=28992 or metrics["failed_count"] or len(metrics["pairs"])!=63:
                raise ValueError("Formal results have incomplete image/pair coverage")
            with (section/"predictions.csv").open(newline="") as f:
                predicted=[row["image_id"] for row in csv.DictReader(f)]
            with (OUT/"integrated/feature_index.csv").open(newline="") as f:
                feature_ids=[row["image_id"] for row in csv.DictReader(f)]
            if predicted!=feature_ids:
                raise ValueError("Formal predictions do not match the integrated index")
            files = {str(p.relative_to(OUT)): file_hash(p) for directory in [OUT/"integrated",run]
                     for p in directory.rglob("*") if p.is_file()}
            write_json(OUT/"delivery.json",dict(run=run.name,files=files,
                       counts=dict(images=len(predicted),pairs=len(metrics["pairs"])),manifest_sha256=PRODUCTION_HASH))
            status("completed", result=run.name)
        except BaseException as error:
            status("failed",error=repr(error))
            raise


if __name__=="__main__":
    main()
