import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import normalized_mutual_info_score
import torch

from inputs import CLUSTER
from zeroshot_common import write_csv, write_json


def gpu_kmeans(x,k,seed,n_init):

    points=torch.from_numpy(x).to("cuda:0")
    rng=np.random.RandomState(seed)
    best_loss=float("inf")
    best_labels=None
    for _ in range(n_init):
        centers,_=kmeans_plusplus(x,k,random_state=rng)
        centers=torch.as_tensor(centers,device=points.device)
        old=None
        for step in range(300):
            d=torch.cdist(points,centers).square()
            labels=d.argmin(1)
            if old is not None and torch.equal(old,labels):
                break
            old=labels
            membership=torch.nn.functional.one_hot(labels,k).to(points.dtype)
            counts=membership.sum(0)
            centers=(membership.T@points)/counts.clamp_min(1)[:,None]
            empty=torch.nonzero(counts==0).flatten()
            if len(empty):
                far=d.min(1).values.argsort(descending=True)[:len(empty)]
                centers[empty]=points[far]
        distances=torch.cdist(points,centers).square()
        loss,labels=distances.min(1)
        loss=loss.sum().item()
        if loss<best_loss:
            best_loss=loss
            best_labels=labels.cpu().numpy().copy()
    if best_labels is None:
        raise ValueError("K-means failed to return finite inertia")
    return best_labels,best_loss


def cluster(samples,out,device,seed,n_init=10):
    available={r["source"] for r in samples.rows}
    names=[s for s in CLUSTER if s in available]
    idx=[i for i,r in enumerate(samples.rows) if r["source"] in names]
    if len(names)<2 or len(idx)<4*len(names):
        raise ValueError("Need at least two sources and enough observations for 4N clusters")
    x=samples.features[idx]
    y=np.array([names.index(samples.rows[i]["source"]) for i in idx])
    assignments=[dict(image_id=samples.rows[i]["image_id"],source=samples.rows[i]["source"]) for i in idx]
    reports=[]
    for multiplier in (1,2,4):
        k=multiplier*len(names)
        if device=="cpu":
            km=KMeans(n_clusters=k,n_init=n_init,max_iter=300,random_state=seed,algorithm="lloyd").fit(x)
            lab,loss=km.labels_,float(km.inertia_)
        else:
            lab,loss=gpu_kmeans(x,k,seed,n_init)
        counts=np.zeros((k,len(names)),int)
        np.add.at(counts,(lab,y),1)
        mapped=counts.argmax(1)[lab]
        perclass=[float(np.mean(mapped[y==i]==i)) for i in range(len(names))]
        a,b=linear_sum_assignment(-counts)
        reports.append(dict(k=k,observed_k=int(np.count_nonzero(counts.sum(1))),n_sources=len(names),
                            avg_accuracy=float(np.mean(perclass)),purity=float(counts.max(1).sum()/len(x)),
                            nmi=float(normalized_mutual_info_score(y,lab,average_method="arithmetic")),
                            hungarian_accuracy=float(counts[a,b].sum()/len(x)),inertia=loss))
        for row,label in zip(assignments,lab):
            row[f"cluster_k{k}"]=int(label)
        write_csv(out/f"contingency_k{k}.csv",("cluster",*names),
                  [dict(cluster=j,**dict(zip(names,values.tolist()))) for j,values in enumerate(counts)])
    write_csv(out/"assignments.csv",tuple(assignments[0]),assignments)
    write_csv(out/"table4.csv",tuple(reports[0]),reports)
    write_json(out/"metrics.json",dict(representation="raw FSD",distance="Euclidean",sources=names,
               missing_sources=sorted(set(CLUSTER)-available),n_init=n_init,seed=seed,device=device,
               avg_accuracy_definition="macro per-source recall after cluster majority labeling",
               hungarian_definition="one-to-one rectangular assignment, unmatched clusters count as errors",
               metrics=reports))
