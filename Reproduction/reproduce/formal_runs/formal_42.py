from project_paths import PROJECT_ROOT, EXTENSION_ROOT

import csv
import fcntl
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT=PROJECT_ROOT
sys.path.insert(0,str(EXTENSION_ROOT / "reproduce"))
from inputs import INPUTS, KNOWN, UNKNOWN, PRODUCTION_HASH, Samples, manifest_rows
from experiments import attribution
from zeroshot_common import file_hash, write_json

PARENT=ROOT/"Results/reproduction"
SOURCE=PARENT/"section41_final_20261110"
OUT=PARENT/"section42_final_20261110"


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def load_merged():
    manifest=INPUTS/"eval_unique.csv"
    delivery=json.loads((SOURCE/"delivery.json").read_text())
    if file_hash(manifest)!=PRODUCTION_HASH or delivery["manifest_sha256"]!=PRODUCTION_HASH:
        raise ValueError("Full manifest differs from the 4.1 handoff")
    folder=SOURCE/"integrated"
    for name in ("feature_index.csv","raw_fsd_960.npy","original_predictions.csv","provenance.json"):
        if file_hash(folder/name)!=delivery["files"]["integrated/"+name]:
            raise ValueError("Handoff checksum mismatch: "+name)
    rows=read_csv(folder/"original_predictions.csv")
    index=read_csv(folder/"feature_index.csv")
    expected=manifest_rows(manifest)
    matrix=np.load(folder/"raw_fsd_960.npy",mmap_mode="r",allow_pickle=False)
    if matrix.shape!=(28992,960) or matrix.dtype!=np.dtype("<f8") or not len(expected)==len(rows)==len(index)==len(matrix):
        raise ValueError("Feature coverage mismatch")
    for i,(x,r,a,m) in enumerate(zip(matrix,rows,index,expected)):
        if (r["status"]!="ok" or any(r[k]!=m[k] for k in ("image_id","source","label","sha256")) or
            int(a["row"])!=i or any(a[k]!=m[k] for k in ("image_id","source","label")) or
            a["image_sha256"]!=m["sha256"] or hashlib.sha256(x.tobytes()).hexdigest()!=a["feature_sha256"]):
            raise ValueError(f"Bad feature mapping at row {i}")
    prov=json.loads((folder/"provenance.json").read_text())
    return Samples(rows,matrix,expected,prov["shards"]),delivery["files"]["integrated/raw_fsd_960.npy"]


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/"lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        final=OUT/"4.2"
        if (final/"checksums.json").exists():
            return
        def state(phase,**kwargs):
            write_json(OUT/"job.json",dict(phase=phase,pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**kwargs))
            print(phase,flush=True)
        try:
            state("validating_integrated_FSD")
            samples,digest=load_merged()
            torch.set_num_threads(4)
            torch.manual_seed(2026)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False
            gpu=torch.cuda.get_device_name(0)
            if "Blackwell" not in gpu:
                raise RuntimeError("Unexpected GPU: "+gpu)
            state("fitting_and_scoring",gpu=gpu)
            with tempfile.TemporaryDirectory(dir=OUT) as temp:
                stage=Path(temp)/"4.2"
                stage.mkdir()
                attribution(samples,stage,"cuda:0",2026)
                split=read_csv(stage/"split.csv")
                train=[r for r in split if r["role"]=="train"]
                test=[r for r in split if r["role"]=="test"]
                predicted=read_csv(stage/"attribution.csv")
                if {r["source"] for r in train}!=set(KNOWN) or not all(r["source"] in (*KNOWN,*UNKNOWN) for r in test):
                    raise ValueError("Source protocol mismatch")
                for key in ("sha256","image_id"):
                    if {r[key] for r in train}&{r[key] for r in test}:
                        raise ValueError("Train/test overlap: "+key)
                if [r["image_id"] for r in predicted]!=[r["image_id"] for r in test]:
                    raise ValueError("Attribution/test alignment mismatch")
                metrics=json.loads((stage/"metrics.json").read_text())
                if len(test)!=metrics["known_n"]+metrics["unknown_n"]:
                    raise ValueError("Wrong metric coverage")
                stats=dict(protocol=metrics["protocol"],gpu=gpu,device="cuda:0",seed=2026,
                           matrix=str(SOURCE/"integrated/raw_fsd_960.npy"),matrix_sha256=digest,
                           manifest_sha256=PRODUCTION_HASH,total_raw_vectors=len(samples.rows),
                           train_counts=dict(Counter(r["source"] for r in train)),
                           test_counts=dict(Counter(r["source"] for r in test)),train_test_shared_hashes=0,
                           missing_unknown=metrics["missing_unknown"])
                write_json(stage/"run.json",stats)
                write_json(stage/"checksums.json",{p.name:file_hash(p) for p in stage.iterdir() if p.is_file()})
                stage.rename(final)
            state("completed",gpu=gpu)
        except BaseException as error:
            state("failed",error=repr(error))
            raise


if __name__=="__main__":
    main()
