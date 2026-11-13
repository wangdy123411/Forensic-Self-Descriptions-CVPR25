import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from inputs import ROOT, INPUTS, WEIGHTS, KNOWN, UNKNOWN, CLUSTER, default_shards, load_samples, manifest_rows
from experiments import detection, attribution, released
from clustering import cluster
from zeroshot_common import write_json, file_hash, child_path


def readiness():
    r=ROOT/"Results/parallel_blackwell/worker.json"
    worker=json.loads(r.read_text()) if r.is_file() else {"status":"not_started"}
    receipt=INPUTS/"primary_final/receipt.json"
    primary=json.loads(receipt.read_text()) if receipt.is_file() else None
    return dict(primary=primary,blackwell=worker,ready=bool(primary and primary["count"]==20992 and worker["status"]=="completed"))


def prepare_training(args):

    from fsd import FSDDetector
    rows=manifest_rows(args.train_manifest)
    test=manifest_rows(args.manifest)
    test_hash={r["sha256"] for r in test}
    test_ids={r["image_id"] for r in test}
    if set(r["source"] for r in rows)!=set(KNOWN) or any(
            r["split"]!="train" or r["label"]!=str(int(r["source"]!="ImageNet-1k")) for r in rows):
        raise ValueError("Training manifest must explicitly use split=train and contain the five known sources")
    hashes=[r["sha256"] for r in rows]
    if len(hashes)!=len(set(hashes)) or test_hash.intersection(hashes) or test_ids.intersection(r["image_id"] for r in rows):
        raise ValueError("Duplicate training content or train/evaluation overlap")
    detector=FSDDetector.load(weights_dir=WEIGHTS,device=args.device,attribution=False)
    vectors=[]
    for r in rows:
        path=child_path(args.train_root,r["relative_path"])
        if file_hash(path)!=r["sha256"]:
            raise ValueError(f"Bad training image digest: {path}")
        raw=detector.compute_fsd(path).detach().cpu().numpy()
        if raw.shape!=(960,) or raw.dtype!=np.float64 or not np.isfinite(raw).all():
            raise ValueError("Invalid training vector")
        vectors.append(raw)
    if args.save.suffix!=".npz":
        raise ValueError("--save must end in .npz")
    args.save.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.save.parent,suffix=".npz",delete=False) as f:
        tmp=Path(f.name)
    try:
        np.savez(tmp,representation="raw_fsd_before_projections",fre_sha256=file_hash(WEIGHTS/"fre.pt"),
                 features=np.stack(vectors),sources=np.array([r["source"] for r in rows]),
                 sha256=np.array(hashes),image_ids=np.array([r["image_id"] for r in rows]))
        if args.save.exists():
            raise FileExistsError(args.save)
        tmp.rename(args.save)
    finally:
        tmp.unlink(missing_ok=True)
    print(args.save)


def main():
    p=argparse.ArgumentParser(description='run')
    p.add_argument("section",choices=["status","4.1","4.2","4.3","all","released","4.2-refit","encode-train"])
    p.add_argument("--manifest",type=Path,default=INPUTS/"eval_unique.csv")
    p.add_argument("--shard",type=Path,action="append",help="Explicit extraction output directory, repeat per shard")
    p.add_argument("--device",choices=["cpu","cuda:0"],default="cuda:0")
    p.add_argument("--smoke",action="store_true",help="Partial preview using up to 24 actual vectors per available source")
    p.add_argument("--wait",action="store_true",help="Wait for both lab shards to complete")
    p.add_argument("--train-npz",type=Path,help="Independent 960-D features for the five known sources")
    p.add_argument("--train-manifest",type=Path)
    p.add_argument("--train-root",type=Path)
    p.add_argument("--save",type=Path)
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--reject-loglik",type=float,help="Optional fixed threshold from independent calibration data")
    p.add_argument("--require-paper-sources",action="store_true",help="Fail if any requested paper source is missing")
    p.add_argument("--output-parent",type=Path,default=ROOT/"Results/reproduction")
    args=p.parse_args()
    if args.section!="4.2-refit" and (args.train_npz is not None or args.reject_loglik is not None):
        p.error("Training archives and refit rejection thresholds require explicit 4.2-refit")
    if args.section=="status":
        print(json.dumps(readiness(),indent=2))
        return
    torch.set_num_threads(1 if args.smoke else 4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    if args.section=="encode-train":
        if not (args.train_manifest and args.train_root and args.save):
            p.error("encode-train requires --train-manifest, --train-root and --save")
        prepare_training(args)
        return
    if args.wait and (args.smoke or args.shard):
        p.error("--wait is for the final two-machine run")
    if not args.smoke and not args.shard:
        state=readiness()
        while args.wait and not state["ready"]:
            if state["blackwell"]["status"]=="failed":
                raise RuntimeError(state["blackwell"])
            time.sleep(30)
            state=readiness()
        if not state["ready"]:
            p.error("The extraction shards are unfinished. Use 'status', '--wait', or '--smoke --device cpu'.")
    if args.device!="cpu" and not torch.cuda.is_available():
        p.error("CUDA is unavailable for this run")
    samples=load_samples(args.manifest,args.shard or default_shards(args.smoke),args.smoke,24 if args.smoke else None)
    sections=("4.1","4.2","4.3") if args.section=="all" else (args.section,)
    if args.require_paper_sources:
        from zeroshot_common import REAL_SOURCES,FAKE_SOURCES
        needed=set()
        if "4.1" in sections: needed.update((*REAL_SOURCES,*FAKE_SOURCES))
        if "4.2" in sections or "released" in sections:
            p.error("Released checkpoint classes differ from the paper's five-known/four-unknown split.")
        if "4.2-refit" in sections: needed.update((*KNOWN,*UNKNOWN))
        if "4.3" in sections: needed.update(CLUSTER)
        missing=needed-{r["source"] for r in samples.rows}
        if missing: p.error(f"Missing requested paper sources: {sorted(missing)}")
    args.output_parent.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name=f"{'smoke' if args.smoke else 'available'}_{args.section}_{stamp}"
    with tempfile.TemporaryDirectory(dir=args.output_parent,prefix=".building-") as temp:
        out=Path(temp)/"result"
        out.mkdir()
        for section in sections:
            dest=out/section
            dest.mkdir()
            if section=="4.1":
                detection(samples,dest,args.device,args.manifest)
            elif section=="4.2-refit":
                attribution(samples,dest,args.device,args.seed,args.train_npz,args.smoke,args.reject_loglik)
            elif section in ("4.2","released"):
                released(samples,dest,args.device)
            else:
                cluster(samples,dest,args.device,args.seed,3 if args.smoke else 10)
        write_json(out/"run.json",dict(sections=sections,device=args.device,seed=args.seed,
            mode="smoke_partial_nonbenchmark" if args.smoke else "available_dataset_reproduction",
            exact_paper_reproduction=False,manifest_sha256=file_hash(args.manifest),
            selected_by_source=dict(Counter(r["source"] for r in samples.rows)),shards=samples.provenance,
            training_archive_sha256=file_hash(args.train_npz) if args.train_npz else None))
        final=args.output_parent/name
        out.rename(final)
    print(final,flush=True)


if __name__=="__main__":
    main()
