import argparse
import numpy as np
from experiments import openset_metrics,heldout
from clustering import gpu_kmeans


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cuda",action="store_true")
    args=parser.parse_args()
    truth=["a","b","u","u"]
    best=np.array([0,1,0,1])
    perfect,_=openset_metrics(truth,best,np.array([4.,3.,2.,1.]),["a","b"])
    assert perfect["known_macro_accuracy"]==perfect["au_oscr"]==perfect["known_unknown_roc_auc"]==1
    tied,curve=openset_metrics(truth,best,np.ones(4),["a","b"])
    assert len(curve)==2 and tied["au_oscr"]==tied["known_unknown_roc_auc"]==.5
    wrong,_=openset_metrics(truth,1-best,np.array([4.,3.,2.,1.]),["a","b"])
    assert wrong["au_oscr"]==0 and wrong["known_unknown_roc_auc"]==1
    rows=[dict(source=source,sha256=f"{source}-{i}") for source in ("StyleGAN","ProGAN","FireFly") for i in range(12)]
    rows[-1]["sha256"]=rows[0]["sha256"]
    train,test=heldout(rows,["StyleGAN","ProGAN"],42)
    assert not {rows[i]["sha256"] for i in train}&{rows[i]["sha256"] for i in test}
    assert all(rows[i]["source"]!="FireFly" for i in train)
    if args.cuda:
        x=np.repeat(np.array([[-10.,0.],[10.,0.]]),6,axis=0)
        y,loss=gpu_kmeans(x,2,2026,2)
        assert loss<1e-8 and np.unique(y[:6]).size==np.unique(y[6:]).size==1 and y[0]!=y[-1]
    print("Metric and split checks passed"+(" (including CUDA Lloyd updates)" if args.cuda else ""))


if __name__=="__main__":
    main()
