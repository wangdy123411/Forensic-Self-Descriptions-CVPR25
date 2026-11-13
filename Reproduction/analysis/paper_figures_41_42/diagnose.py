from project_paths import PROJECT_ROOT, EXTENSION_ROOT

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

PROJECT = PROJECT_ROOT
RESULT = PROJECT / "Results/reproduction"
OUT = Path(__file__).resolve().parent / "build"


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def loglik(x, means, weights, cov):
    lower = np.linalg.cholesky(cov)
    whitener = solve_triangular(lower, np.eye(len(cov)), lower=True).T
    white = x @ whitener
    mu = means @ whitener
    scores = np.column_stack([-.5*np.einsum("ij,ij->i", white-m, white-m) for m in mu])
    norm = .5*x.shape[1]*np.log(2*np.pi) + np.log(np.diag(lower)).sum()
    return logsumexp(scores+np.log(weights), axis=1)-norm


def main():
    OUT.mkdir(exist_ok=True)
    a = RESULT / "section42_final_20261110/4.2"
    b = RESULT / "section41_final_20261110/integrated"
    for filename, sha in json.loads((a/"checksums.json").read_text()).items():
        with (a/filename).open("rb") as f:
            assert hashlib.file_digest(f,"sha256").hexdigest()==sha
    baseline = json.loads((a/"metrics.json").read_text())
    names=baseline["known"]
    pred=records(a/"attribution.csv")
    split=records(a/"split.csv")
    train=[r for r in split if r["role"]=="train"]
    test=[r for r in split if r["role"]=="test"]
    assert [r["image_id"] for r in test]==[r["image_id"] for r in pred]
    assert not {r["sha256"] for r in train}&{r["sha256"] for r in test}
    index={r["image_id"]:r for r in records(b/"feature_index.csv")}
    x_all=np.load(b/"raw_fsd_960.npy",mmap_mode="r")
    for r in test+train:
        assert index[r["image_id"]]["image_sha256"]==r["sha256"] and index[r["image_id"]]["source"]==r["source"]
    x=x_all[[int(index[r["image_id"]]["row"]) for r in test]]
    g=torch.load(a/"fitted_gmms.pt", weights_only=True,map_location="cpu")["sources"]
    assert [s["name"] for s in g]==names
    ntrain=Counter(r["source"] for r in train)
    covs=[s["covariances_"].numpy() for s in g]
    pool=np.average(covs,axis=0,weights=[ntrain[n] for n in names])
    truth=[r["source"] for r in test]
    sys.path.insert(0,str(EXTENSION_ROOT / "reproduce"))
    from experiments import openset_metrics
    diagnostics={}
    floorcounts={}
    for n,c in zip(names,covs):
        eigen=np.linalg.eigvalsh(c)
        floorcounts[n]=int(np.sum(np.isclose(eigen,1e-6,rtol=1e-6,atol=1e-12)))
    for mode,matrices in [("as_fitted",covs),("shared_across_sources",[pool]*len(g))]:
        ll=np.column_stack([loglik(x,s["means_"].numpy(), s["weights_"].numpy(),c) for s,c in zip(g,matrices)])
        winner=ll.argmax(1)
        correct,_=openset_metrics(truth,winner,ll.max(1),names)
        confusion=[[int(np.sum((np.array(truth)==n)&(winner==j))) for j in range(len(names))] for n in names]
        if mode=="as_fitted":
            assert all(names[j]==r["candidate"] for j,r in zip(winner,pred))
            assert abs(correct["known_macro_accuracy"]-baseline["known_macro_accuracy"])<1e-12
            assert abs(correct["au_oscr"]-baseline["au_oscr"])<1e-12
        diagnostics[mode]={"metrics":correct, "confusion_known":confusion}
    result=dict(names=names,training_counts=ntrain,at_regularization_floor=floorcounts,
                covariance_floor=1e-6,feature_dimensions=x.shape[1],known_n=baseline["known_n"],
                unknown_n=baseline["unknown_n"],missing_unknown=baseline["missing_unknown"],
                method="Replace the five fitted within-source covariance matrices by their training-count-weighted mean. Means, component weights, IDs and split are unchanged. Post-hoc diagnostic on the existing test set.",
                runs=diagnostics)
    (OUT/"4_2_diagnostic.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__":
    with threadpool_limits(limits=4):
        main()
