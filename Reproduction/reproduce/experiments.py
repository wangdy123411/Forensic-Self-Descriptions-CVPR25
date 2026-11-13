from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from inputs import KNOWN, UNKNOWN, CODE, WEIGHTS, INPUTS
from zeroshot_common import summarize, write_json, write_csv, PREDICTION_FIELDS, file_hash
from fsd.projection import load_transforms, apply_projections
from fsd.gmm import TorchGMM, load_gmm
from fsd.attribution import load_source_gmms
from export_tables import export_run


def batches(x, size=128):
    for i in range(0,len(x),size):
        yield x[i:i+size]


def detection(samples,out,device,manifest):
    config=json.loads((WEIGHTS/"config.json").read_text())
    scoring=config["scoring"]
    transforms=load_transforms(WEIGHTS/config["transforms"]["weights_file"],device)
    model=load_gmm(WEIGHTS/config["gmm"]["weights_file"],device=device)
    with torch.inference_mode():
        raw=np.concatenate([model.score_samples(apply_projections(torch.from_numpy(x).to(device),transforms)).cpu().numpy()
                            for x in batches(samples.features)])
    z=(raw-float(scoring["train_mean"]))/float(scoring["train_std"])
    threshold=float(scoring["default_threshold"])
    if not np.isfinite(z).all():
        raise ValueError("Non-finite detection score")
    archived=np.array([float(r["z_score"]) for r in samples.rows])
    decisions=z<threshold
    predictions=[dict(r,raw_score=repr(float(ll)),z_score=repr(float(v)),fake_score=repr(float(-v)),
                      threshold=repr(threshold),predicted_fake=str(int(fake)))
                 for r,ll,v,fake in zip(samples.rows,raw,z,decisions)]
    write_csv(out/"predictions.csv",PREDICTION_FIELDS,predictions)
    report=summarize(predictions,samples.manifest)
    report["max_abs_z_difference_from_image_run"]=float(np.max(np.abs(z-archived)))
    report["changed_decisions"]=int(np.sum(decisions!=np.array([bool(int(r["predicted_fake"])) for r in samples.rows])))
    write_json(out/"metrics.json",report)
    export_run(out,manifest,dataset_status_path=INPUTS/"dataset_status.json")


def openset_metrics(truth,candidates,score,names,*,require_all_classes=True):

    labels=np.asarray([names.index(s) if s in names else -1 for s in truth])
    known=labels>=0
    nk=int(known.sum())
    nu=len(labels)-nk
    if not nk or not nu or not np.isfinite(score).all():
        raise ValueError("Open-set evaluation needs known and unknown samples and finite scores")
    totals=np.bincount(labels[known],minlength=len(names))
    observed=totals>0
    if require_all_classes and not observed.all():
        raise ValueError("Every evaluated known source needs a held-out sample")
    tp=np.zeros(len(names),int)
    predicted=np.zeros(len(names),int)
    def point(t,ka,ua):
        return dict(threshold=t,known_accept=ka/nk,unknown_accept=ua/nu,
                    correct_known_accept=float(tp.sum()/nk),unknown_reject=1-ua/nu,
                    known_macro_f1=float(np.mean(2*tp[observed]/(totals+predicted)[observed])))
    curve=[point("above_max",0,0)]
    ka=ua=0
    order=np.argsort(-score,kind="stable")
    for index,i in enumerate(order):
        k=int(candidates[i])
        predicted[k]+=1
        if known[i]:
            ka+=1
            tp[k]+=int(labels[i]==k)
        else:
            ua+=1
        if index+1==len(order) or score[i]!=score[order[index+1]]:
            curve.append(point(float(score[i]),ka,ua))
    def area(y,x):
        return float(sum((b[x]-a[x])*(a[y]+b[y])/2 for a,b in zip(curve,curve[1:])))
    recalls=tp[observed]/totals[observed]
    metrics=dict(known_macro_accuracy=float(np.mean(recalls)),known_n=nk,unknown_n=nu,
                 known_source_accuracy=dict(zip(np.asarray(names)[observed].tolist(),recalls.tolist())),
                 au_oscr=area("correct_known_accept","unknown_accept"),
                 au_crr_fang_macro_f1_vs_crr=-area("known_macro_f1","unknown_reject"),
                 known_unknown_roc_auc=area("known_accept","unknown_accept"))
    return metrics,curve


def heldout(rows,known_names,seed):

    unknown_hashes={r["sha256"] for r in rows if r["source"] in UNKNOWN}
    train=[]
    for name in known_names:
        idx=[i for i,r in enumerate(rows) if r["source"]==name and r["sha256"] not in unknown_hashes]
        idx.sort(key=lambda i:hashlib.sha256(f"{seed}:{rows[i]['sha256']}".encode()).digest())
        train.extend(idx[:len(idx)//2])
    hashes={rows[i]["sha256"] for i in train}
    test=[i for i,r in enumerate(rows) if r["source"] in (*known_names,*UNKNOWN) and r["sha256"] not in hashes]
    return np.array(train,int),np.array(test,int)


def pack_gmm(name,model):
    d=dict(name=name,n_components=model.n_components,covariance_type=model.covariance_type)
    for key in ("weights_","means_","covariances_","precisions_cholesky_"):
        v=getattr(model,key).detach().cpu()
        if not torch.isfinite(v).all():
            raise ValueError(f"Non-finite trained GMM: {name}")
        d[key]=v
    return d


def attribution(samples,out,device,seed,train_npz=None,smoke=False,tau=None):
    rows=samples.rows
    present=set(r["source"] for r in rows)
    names=[n for n in KNOWN if n in present]
    if (not smoke and set(names)!=set(KNOWN)) or not names:
        raise ValueError(f"Missing known test sources: {sorted(set(KNOWN)-present)}")
    if train_npz:
        with np.load(train_npz,allow_pickle=False) as data:
            if str(data["representation"].item())!="raw_fsd_before_projections" or str(data["fre_sha256"].item())!=file_hash(WEIGHTS/"fre.pt"):
                raise ValueError("Training archive has a different feature representation or extractor")
            x=np.array(data["features"],dtype=np.float64)
            labels=data["sources"].astype(str)
            hashes=data["sha256"].astype(str)
            ids=data["image_ids"].astype(str)
        if x.shape!=(len(labels),960) or len(hashes)!=len(x) or len(ids)!=len(x) or not np.isfinite(x).all():
            raise ValueError("Malformed training feature archive")
        if set(labels)!=set(names) or len(set(ids))!=len(ids) or len(set(hashes))!=len(hashes):
            raise ValueError("Training set must contain unique records from exactly the known sources")
        if set(hashes)&{r["sha256"] for r in samples.manifest} or set(ids)&{r["image_id"] for r in samples.manifest}:
            raise ValueError("Training and evaluation content or IDs overlap")
        test=[i for i,r in enumerate(rows) if r["source"] in (*names,*UNKNOWN)]
        mode="independent_training_features"
    else:
        train,test=heldout(rows,names,seed)
        x=samples.features[train]
        labels=np.array([rows[i]["source"] for i in train])
        hashes=[rows[i]["sha256"] for i in train]
        ids=[rows[i]["image_id"] for i in train]
        mode="seeded_holdout_from_available_known_test_pool"
    split=[dict(image_id=i,sha256=h,source=s,role="train") for i,h,s in zip(ids,hashes,labels)]
    split.extend(dict(image_id=rows[i]["image_id"],sha256=rows[i]["sha256"],source=rows[i]["source"],role="test") for i in test)
    write_csv(out/"split.csv",("image_id","sha256","source","role"),split)
    models=[]
    scores=[]
    fit_info={}
    for name in names:
        xx=x[labels==name]
        m=TorchGMM(n_components=5,covariance_type="tied",random_state=seed,device=device)
        if len(xx)<5:
            raise ValueError(f"{name} needs at least 5 independent training vectors")
        m.fit(xx)
        m._update_inference_cache()
        models.append(pack_gmm(name,m))
        fit_info[name]=dict(n=len(xx),iterations=m.n_iter_,converged=bool(m.converged_))
        scores.append(np.concatenate([m.score_samples(t).cpu().numpy() for t in batches(samples.features[test])]))
    torch.save(dict(sources=models,representation="raw_fsd_before_projections",seed=seed),out/"fitted_gmms.pt")
    ll=np.stack(scores,axis=1)
    if not np.isfinite(ll).all():
        raise ValueError("Non-finite GMM likelihood")
    best=ll.argmax(axis=1)
    max_ll=ll.max(axis=1)
    actual=[rows[i]["source"] for i in test]
    metrics,curve=openset_metrics(actual,best,max_ll,names)
    metrics.update(protocol=mode,representation="raw FSD",known=names,unknown=[s for s in UNKNOWN if s in present],
                   missing_unknown=sorted(set(UNKNOWN)-present),fit=fit_info,seed=seed,reject_log_likelihood=tau,
                   strict_paper_reproduction=False)
    output=[]
    for j,i in enumerate(test):
        candidate=names[best[j]]
        output.append(dict(image_id=rows[i]["image_id"],source=actual[j],candidate=candidate,
                           max_log_likelihood=float(max_ll[j]),decision="" if tau is None else
                           (candidate if max_ll[j]>=tau else "unknown"),**dict(zip(names,ll[j].tolist()))))
    write_csv(out/"attribution.csv",tuple(output[0]),output)
    write_csv(out/"threshold_curve.csv",tuple(curve[0]),curve)
    write_json(out/"metrics.json",metrics)


def released(samples,out,device):
    def checkpoint(name,digest):
        paths=[WEIGHTS/name,CODE/name]
        path=next((p for p in paths if p.is_file()),None)
        if path is None or file_hash(path)!=digest:
            raise ValueError(f"Missing or invalid official checkpoint: {name}")
        return path
    attr=checkpoint("attribution_transforms.pt","90fab44761c138a70ec7de585d4ec998ce1fe1db8a36fd8631b5988207067605")
    gmms=checkpoint("source_gmms.pt","42b5e4110a4ff867503983a806b2c7c8674c545994d59ff5c4f01369c54e82d0")
    det=checkpoint("fsd_transforms.pt","1e87d792b413101e58d9de71551182a1fab8b879ca6f6ba9780b6adcb9a5a699")
    transforms=load_transforms(det,device)+load_transforms(attr,device)
    models=load_source_gmms(gmms,device)
    names=list(models)
    aliases={"DALLE 3":"DALL-E 3","FireFly":"Adobe Firefly","SDXL":"Stable Diffusion XL",
             "SD1.5":"Stable Diffusion 1.5","SD3.0":"Stable Diffusion 3","MJ v6":"Midjourney v6"}
    result=[]
    with torch.inference_mode():
        for offset in range(0,len(samples.rows),128):
            x=torch.from_numpy(samples.features[offset:offset+128]).to(device)
            x=apply_projections(x,transforms)
            ll=torch.stack([m.score_samples(x) for m in models.values()],1)
            if not torch.isfinite(ll).all():
                raise ValueError("Nonfinite released GMM score")
            best=ll.argmax(1)
            probs=torch.softmax((ll-ll.max(1,keepdim=True).values)/750.,dim=1)
            for j,k in enumerate(best.tolist()):
                r=samples.rows[offset+j]
                target="Real" if int(r["label"])==0 else aliases.get(r["source"],r["source"])
                supported=target in models
                result.append(dict(image_id=r["image_id"],source=r["source"],candidate=names[k],
                                   target=target if supported else "",supported=int(supported),
                                   correct=int(names[k]==target) if supported else "",
                                   max_log_likelihood=ll[j,k].item(),confidence=probs[j,k].item()))
    write_csv(out/"attribution.csv",tuple(result[0]),result)
    known=[r for r in result if r["supported"]]
    counts=Counter(r["target"] for r in known)
    hits=Counter(r["target"] for r in known if r["correct"])
    by_class=[dict(source=n,n=counts[n],correct=hits[n],recall=hits[n]/counts[n]) for n in names if counts[n]]
    write_csv(out/"per_class.csv",("source","n","correct","recall"),by_class)
    metrics=dict(protocol="official_release_14_way_unmasked_argmax",training_images_used=0,
                 scored=len(result),supported_images=len(known),unsupported_images=len(result)-len(known),
                 macro_accuracy=float(np.mean([r["recall"] for r in by_class])) if known else None,
                 micro_accuracy=sum(hits.values())/len(known) if known else None,
                 represented_model_classes=len(counts),total_model_classes=len(names),
                 unsupported_sources=dict(Counter(r["source"] for r in result if not r["supported"])),
                 closed_set_per_class=by_class,unknown_rejection_threshold=None,paper_table_3=False)
    if 0<len(known)<len(result):
        curve_metrics,curve=openset_metrics([r["target"] for r in result],
                np.array([names.index(r["candidate"]) for r in result]),
                np.array([r["max_log_likelihood"] for r in result]),names,require_all_classes=False)
        metrics["open_set_on_release_coverage"]=curve_metrics
        write_csv(out/"threshold_curve.csv",tuple(curve[0]),curve)
    write_json(out/"metrics.json",metrics)
    write_json(out/"model.json",dict(mode="released_14_way_argmax",classes=names,n=len(result),
               paper_table_3=False,temperature=750.0,training_images_used=0,
               weight_sha256={p.name:file_hash(p) for p in (det,attr,gmms)}))
