from project_paths import CODE_ROOT

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HEADS = ("Real", "StyleGAN3", "ProGAN", "GigaGAN", "Stable Diffusion XL")
SOURCES = ("ImageNet-1k", "StyleGAN3", "ProGAN", "GigaGAN", "SDXL")
METHODS = ("previous_knn", "two_centers")
LOCK = {
    "gigagan_two_centers/coordinates.npz": "ad28c8ebd31d3a835bfdd3be0dfb86ad63d2b6af72d145420d73640110bfeb95",
    "gigagan_two_centers/results.json": "6476f098bce7de0f4bbf9a09a8d13f3da5e5faaf29eadf4c6f5ae0cf13ff25a3",
    "reference_banks.csv": "0e073ea9f1ae2ca867f3dbf35ffcfd60ba8419043b518cfaf3999962696110f5",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def assignments(records):


    return np.array([int(hashlib.sha256(("fsd-reference-neighbors-v1/sha256/"+r["sha256"])
                    .encode()).hexdigest()[:16],16)%5 for r in records])


def load_queries(selections, cache, excluded):
    records, features = [], []
    seen = set()
    for manifest in selections:
        for r in read_csv(manifest):
            h = r["sha256"]
            if h in excluded:
                continue
            if h in seen:
                raise ValueError(f"Duplicate new image {h}")
            seen.add(h)
            path = manifest.parent / (r.get("path") or r["relative_path"])
            if sha(path)!=h:
                raise ValueError(f"Input digest mismatch: {path}")
            z = np.load(cache/(h+".npy"),allow_pickle=False)
            if z.shape!=(960,) or z.dtype!=np.float64 or not np.isfinite(z).all():
                raise ValueError(f"Invalid saved feature for {h}")
            source = {"gigagan":"GigaGAN","firefly":"Firefly"}[r["source"].lower()]
            variant = r.get("variant") or r.get("generator") or source
            records.append(dict(sha256=h,image_id=r.get("image_id") or h[:24],source=source,variant=variant,
                                filename=path.name,origin=r.get("source_url") or r.get("url","")))
            features.append(z)
    return records,np.stack(features)


def gmm_likelihoods(project, features):
    code = CODE_ROOT
    sys.path.insert(0,str(code))
    from fsd.attribution import load_source_gmms
    from fsd.projection import apply_projections,load_transforms
    weights=project/"Weights/fsd_v1.2.0"
    model=json.loads((project/"Results/reproduction/section42_author_v120_20261110/4.2/model.json").read_text())
    files={}
    for name,digest in model["weight_sha256"].items():
        files[name]=next(base/name for base in (weights,code) if (base/name).is_file())
        if sha(files[name])!=digest:
            raise ValueError(f"Unexpected checkpoint: {name}")
    projections = load_transforms(files["fsd_transforms.pt"],"cpu")+load_transforms(files["attribution_transforms.pt"],"cpu")
    models = load_source_gmms(files["source_gmms.pt"],"cpu")
    torch.set_num_threads(4)
    with torch.inference_mode():
        z = apply_projections(torch.from_numpy(features),projections)
        return torch.stack([models[h].score_samples(z) for h in HEADS],dim=1).numpy()


def mean5(z,bank):

    values=[]
    for block in np.array_split(z,max(1,(len(z)+15)//16)):
        distances=np.linalg.norm(block[:,None,:]-bank[None,:,:],axis=2)
        values.extend(np.partition(distances,4,axis=1)[:,:5].mean(axis=1))
    return np.asarray(values)


def apply_rules(z,winner,maximum,previous,lookup,raw,coordinates,protocols):
    gallery=read_csv(previous/"reference_banks.csv")
    folds=range(5)
    output={p:{mode:{m:np.zeros((len(z),5),dtype=bool) for m in METHODS}
               for mode in ("matched95_frozen","calibration_frozen")} for p in protocols}
    evidence=[]
    for f in folds:
        ds=np.empty((len(z),len(HEADS)))
        for j,source in enumerate(SOURCES):
            ids=[r["image_id"] for r in gallery if int(r["test_fold"])==f and r["source"]==source]
            assert len(ids)==len(set(ids))==200
            ds[:,j]=mean5(z,raw[[lookup[i] for i in ids]])
        relative=ds[np.arange(len(z)),winner]/coordinates["rel5_scale"][f,winner]
        two=np.linalg.norm(z[:,None,:]-coordinates["two_centers"][f],axis=2)/coordinates["two_radii"][f]
        giga={"previous_knn":ds[:,3],"two_centers":two.min(axis=1)}
        evidence.append((relative,ds[:,3],two))
        for name,setting in protocols.items():
            gates={r["rule"]:r for r in setting["fold_cutoffs"] if r["test_fold"]==f}
            for m in METHODS:
                cut=gates[m]
                eligible=(relative<=cut["common_ceiling"]) & ((winner!=3)|(giga[m]<=cut["gigagan_ceiling"]))
                for mode,gm in (("matched95_frozen",setting["matched95_gmm_cutoffs"][m]),("calibration_frozen",cut["gmm_cutoff"])):
                    output[name][mode][m][:,f]=eligible & (maximum>=gm)
    return output,evidence


def stats(mask,a,b):
    old,new=a[mask],b[mask]
    return dict(n=int(mask.sum()),previous=int(old.sum()),two_centers=int(new.sum()),
                yes_to_no=int((old & ~new).sum()),no_to_yes=int((~old & new).sum()),
                pct_point_change=round(100*float(new.mean()-old.mean()),5))


def run(args):
    previous=args.project/"Results/reproduction/section42_reference_neighbors_20261112"
    for file,digest in LOCK.items():
        if sha(previous/file)!=digest:
            raise ValueError(f"Changed frozen model file: {file}")
    with np.load(previous/"gigagan_two_centers/coordinates.npz",allow_pickle=False) as npz:
        coordinates={k:npz[k] for k in ("rel5_scale","two_centers","two_radii","heads")}
    assert tuple(coordinates["heads"])==HEADS
    protocols=json.loads((previous/"gigagan_two_centers/results.json").read_text())["protocols"]
    base=args.project/"Results/reproduction/section41_final_20261110/integrated"
    index=read_csv(base/"feature_index.csv")
    lookup={r["image_id"]:int(r["row"]) for r in index}
    raw=np.load(base/"raw_fsd_960.npy",mmap_mode="r",allow_pickle=False)
    excluded={line.strip() for line in args.exclude.read_text(encoding="utf-8-sig").splitlines() if line.strip()} if args.exclude else set()
    records,z=load_queries(args.selection,args.features,excluded)
    hashes={r["image_sha256"] for r in index}
    if any(r["sha256"] in hashes for r in records):
        raise ValueError("Old evaluation image in new cohort")
    archived=read_csv(previous/"decisions.csv")
    anchors=[]
    for s in ("GigaGAN","FireFly","DALLE 3","SDXL","ImageNet-1k","ProGAN","StyleGAN3"):
        for f in range(5):
            anchors.append(next(r for r in archived if r["source"]==s and int(r["fold"])==f))
    anchor_vectors=raw[[lookup[r["image_id"]] for r in anchors]]
    all_z=np.concatenate((z,anchor_vectors))
    likelihood=gmm_likelihoods(args.project,all_z)
    winner=likelihood.argmax(axis=1)
    maximum=likelihood.max(axis=1)
    np.testing.assert_array_equal(np.array(HEADS)[winner[len(z):]],[r["candidate"] for r in anchors])
    np.testing.assert_allclose(maximum[len(z):],[float(r["gmm_max"]) for r in anchors],rtol=0,atol=1e-6)
    rules,ev=apply_rules(all_z,winner,maximum,previous,lookup,raw,coordinates,protocols)
    fold=assignments(records)
    n=len(records)
    a_ix=np.arange(n,n+len(anchors))
    a_fold=[int(r["fold"]) for r in anchors]
    anchor_decisions={r["image_id"]:r for r in read_csv(previous/"gigagan_two_centers/decisions.csv")}
    for p,by_mode in rules.items():
        for mode,methods in by_mode.items():
            for m,dec in methods.items():
                prior_mode="matched95" if mode=="matched95_frozen" else "heldout"
                np.testing.assert_array_equal(dec[a_ix,a_fold],
                    [anchor_decisions[r["image_id"]][f"{p}_{prior_mode}_{m}"]=="1" for r in anchors])
    sources=np.array([r["source"] for r in records])
    variants=np.array([r["variant"] for r in records])
    giga=sources=="GigaGAN"
    fire=sources=="Firefly"
    main=np.arange(n)
    for i,r in enumerate(records):
        repeat=ev[fold[i]]
        r.update(fold=int(fold[i]),candidate=HEADS[winner[i]],max_log_likelihood=float(maximum[i]),
                 winner_relative5=float(repeat[0][i]),gigagan_mean5=float(repeat[1][i]),
                 gigagan_imagenet_distance=float(repeat[2][i,0]),gigagan_coco_distance=float(repeat[2][i,1]))
    result={"frozen_artifact_sha256":LOCK,"archived_decision_replay_n":len(anchors),
            "selection_sha256":{f"{f.parent.name}/{f.name}":sha(f) for f in args.selection},
            "overlap_exclusions":sorted(excluded),
            "assignment":"Hash fsd-reference-neighbors-v1/sha256/<image byte SHA256> to choose one of the five previously frozen galleries. Also evaluate all queries separately against each gallery.",
            "display_protocol":"original_three_known matched95_frozen; threshold kept from original 95% operating point",
            "gmm_nomination_by_source":{s:dict(Counter(np.array(HEADS)[winner[:n][sources==s]])) for s in sorted(set(sources))},
            "cohorts":dict(Counter(sources)), "protocols":{}}
    metrics=[]
    groups={"GigaGAN":giga,"Firefly":fire}
    for v in sorted(set(variants[giga])):
        groups["GigaGAN/"+v]=giga & (variants==v)
    for p,by_mode in rules.items():
        result["protocols"][p]={}
        for mode,methods in by_mode.items():
            summary={}
            a,b=methods["previous_knn"][:n],methods["two_centers"][:n]
            for i,r in enumerate(records):
                for m,ds in methods.items():
                    r[f"{p}_{mode}_{m}"]=int(ds[i,fold[i]])
            for group,mask in groups.items():
                if not mask.any():
                    continue
                selections=[("accepted",a,b)]
                if group!="Firefly":
                    nominated=(winner[:n]==3)[:,None]
                    selections.append(("correctly_accepted_as_gigagan",a & nominated,b & nominated))
                for measure,x,y in selections:
                    s=stats(mask,x[main,fold],y[main,fold])
                    metrics.append(dict(protocol=p,cutoffs=mode,cohort=group,measure=measure,**s))
                    s["all_images_evaluated_with_each_frozen_gallery"]=[stats(mask,x[:,f],y[:,f]) for f in range(5)]
                    summary[group+"/"+measure]=s
            if fire.any():
                summary["firefly_by_gmm_nomination"] = {
                    h:stats(fire & (winner[:n]==j),a[main,fold],b[main,fold])
                    for j,h in enumerate(HEADS) if np.any(fire & (winner[:n]==j))}
            result["protocols"][p][mode]=summary
    args.out.mkdir(parents=True,exist_ok=True)
    write_csv(args.out/"summary.csv",metrics)
    write_csv(args.out/"decisions.csv",records)
    (args.out/"results.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:{a:b for a,b in s.items() if a!="all_images_evaluated_with_each_frozen_gallery"}
                      for k,s in result["protocols"]["original_three_known"]["matched95_frozen"].items()},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description='evaluate external')
    for name in ("project","features","out"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--exclude",type=Path,help="One excluded byte SHA-256 per line from the overlap audit.")
    p.add_argument("--selection",action="append",required=True,type=Path)
    run(p.parse_args())
