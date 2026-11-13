from project_paths import PROJECT_ROOT, CODE_ROOT, EXTENSION_ROOT, WEIGHTS_ROOT, INPUTS_ROOT

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

CODE = CODE_ROOT
ROOT = PROJECT_ROOT
INPUTS = INPUTS_ROOT
WEIGHTS = WEIGHTS_ROOT
sys.path[:0] = [str(CODE), str(EXTENSION_ROOT/"zeroshot_validation")]
from zeroshot_common import MANIFEST_FIELDS, file_hash, PREDICTION_FIELDS

PRODUCTION_HASH = "2672fea786c671586928f2cef62cda1777925e9ae5ab1cf1ddbd0066fac22a26"
KNOWN = ("ImageNet-1k","StyleGAN","StyleGAN3","SD1.4","ProGAN")
UNKNOWN = ("StyleGAN2","SD3.0","DALLE 3","FireFly")
CLUSTER = ("ImageNet-1k","ProGAN","StyleGAN3","GLIDE","SD1.5","DALLE 3","MJ v6","FireFly")


@dataclass
class Samples:
    rows: list
    features: np.ndarray
    manifest: list
    provenance: list


def manifest_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        if not set(MANIFEST_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Missing fields in evaluation manifest")
        rows=list(reader)
    if not rows or len({r["image_id"] for r in rows})!=len(rows):
        raise ValueError("Empty manifest or duplicate image IDs")
    return rows


def default_shards(preview):
    primary=INPUTS/"primary_final"
    if preview and not primary.exists():
        primary=INPUTS/"primary_preview"
    plan=json.loads((ROOT/"assignment.json").read_text())
    runs=[primary]
    for part in plan["parts"]:
        shard=ROOT/"Results/parallel_blackwell"/part["key"]
        if not preview or (shard/"predictions.csv").is_file():
            runs.append(shard)
    return runs


def load_samples(manifest, shards, preview, cap=None):
    rows=manifest_rows(manifest)
    if manifest.resolve()==(INPUTS/"eval_unique.csv").resolve() and file_hash(manifest)!=PRODUCTION_HASH:
        raise ValueError("The current run's full manifest digest does not match")
    expected={r["image_id"]:r for r in rows}
    seen=set()
    predictions={}
    vectors={}
    details=[]
    counts=Counter()
    for shard in shards:
        predfile=shard/"predictions.csv"
        if not predfile.exists():
            raise FileNotFoundError(f"Waiting for {predfile}. Use 'status' to check coverage.")
        meta=json.loads((shard/"run_metadata.json").read_text("utf-8-sig"))["identity"]
        if meta["weights_sha256"]!={n:file_hash(WEIGHTS/n) for n in meta["weights_sha256"]}:
            raise ValueError(f"Detection weights mismatch in {shard}")
        db=(shard/"raw_fsd.sqlite" if (shard/"raw_fsd.sqlite").is_file() else shard/"feature_cache/raw_fsd.sqlite")
        data=predfile.read_bytes()
        if data and not data.endswith(b'\n') and not preview:
            raise ValueError(f"Unterminated predictions in {shard}")
        pred=list(csv.DictReader(data[:data.rfind(b'\n')+1].decode("utf-8-sig").splitlines()))
        conn=sqlite3.connect(db.resolve().as_uri()+"?mode=ro",uri=True)
        try:
            header=dict(conn.execute("select key,value from metadata"))
            identity=json.loads(header["extraction_identity"])
            sig=hashlib.sha256(header["extraction_identity"].encode()).hexdigest()
            if not (sig==header["extraction_signature"] and identity==meta["feature_cache"]["extraction_identity"]
                    and identity["representation"]=="raw_fsd_before_projections"
                    and identity["dimensions"]==960 and identity["dtype"]=="<f8"
                    and identity["fre_weights_sha256"]==file_hash(WEIGHTS/"fre.pt")
                    and all(h==file_hash(CODE/"fsd"/name) for name,h in identity["extractor_code_sha256"].items())
                    and identity["fsd_config"]==meta["model_config"]["fsd"]):
                raise ValueError(f"Incompatible raw FSD metadata in {shard}")
            for r in pred:
                ident=r["image_id"]
                if ident in seen or ident not in expected or any(r[k]!=expected[ident][k] for k in MANIFEST_FIELDS):
                    raise ValueError(f"Duplicate, foreign, or mislabeled image: {ident}")
                if r["status"]!="ok" or not np.isfinite([float(r[k]) for k in ("raw_score","z_score","fake_score")]).all():
                    raise ValueError(f"Invalid prediction: {ident}")
                seen.add(ident)
                if cap and counts[r["source"]]>=cap:
                    continue
                item=conn.execute("select image_sha256,source,label,dimensions,dtype,feature,feature_sha256,extraction_signature from features where image_id=?",(ident,)).fetchone()
                if not item or item[:5]!=(r["sha256"],r["source"],int(r["label"]),960,"<f8") or item[7]!=sig:
                    raise ValueError(f"FSD identity mismatch: {ident}")
                blob=item[5]
                x=np.frombuffer(blob,dtype="<f8")
                if x.shape!=(960,) or not np.isfinite(x).all() or hashlib.sha256(blob).hexdigest()!=item[6]:
                    raise ValueError(f"Corrupt FSD vector: {ident}")
                counts[r["source"]]+=1
                vectors[ident]=x
                predictions[ident]=r
            details.append(dict(shard=str(shard),rows=len(pred),signature=sig,gpu=identity["gpu"],manifest_sha256=meta["manifest_sha256"]))
        finally:
            conn.close()
    if not preview and seen!=set(expected):
        raise ValueError(f"Final run requires all {len(expected)} prediction IDs; currently found {len(seen)}.")
    result=[predictions[r["image_id"]] for r in rows if r["image_id"] in predictions]
    if not result:
        raise ValueError("No successful prediction/FSD pairs")
    return Samples(result,np.stack([vectors[r["image_id"]] for r in result]),rows,details)
