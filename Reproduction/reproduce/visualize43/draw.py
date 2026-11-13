import argparse
import collections
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


SOURCE_COLORS = {
    "ImageNet-1k": "#1f77b4", "ProGAN": "#ff7f0e", "StyleGAN3": "#2ca02c",
    "GLIDE": "#d62728", "SD1.5": "#9467bd", "DALLE 3": "#8c564b",
    "MJ v6": "#e377c2", "FireFly": "#7f7f7f"
}
SOURCE_LABELS = {"ImageNet-1k":"ImageNet-1k (real)", "DALLE 3":"DALL·E 3", "FireFly":"Firefly"}
CLUSTER_COLORS = ["#0072B2","#D55E00","#009E73","#8D67B4","#B18708","#C95D91"]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, dest, name):
    fig.savefig(dest/f"{name}.png", dpi=300, facecolor="white")
    plt.close(fig)


def scatter(dest, name, xy, category, values, colors, labels):
    fig=plt.figure(figsize=(8.8,6.9))
    ax=fig.add_axes((.035,.117,.93,.855))

    order=np.random.default_rng(2026).permutation(len(category))
    ax.scatter(xy[order,0],xy[order,1],s=7.5,color=[colors[v] for v in category[order]],
               alpha=.76,linewidths=0,edgecolors="none",rasterized=True)
    ax.set_aspect("equal",adjustable="box")
    ax.margins(.045)
    ax.set_axis_off()
    handles=[Line2D([],[],linestyle="",marker="o",markersize=8,markerfacecolor=colors[v],
                    markeredgewidth=0,label=labels.get(v,v)) for v in values]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.012),ncol=3,frameon=False,
               handlelength=.9,handletextpad=.45,columnspacing=2.25,labelspacing=.5,fontsize=13)
    save(fig,dest,name)


def heatmap(dest, counts, sources):
    share=counts/counts.sum(axis=1,keepdims=True)
    fig,ax=plt.subplots(figsize=(8.7,6.0))
    fig.subplots_adjust(left=.252,right=.963,bottom=.12,top=.84)
    palette=LinearSegmentedColormap.from_list("fraction",["#f5f8fa","#d6e4ee","#9ac1d8","#5596b7","#1b698f","#08496b"])
    ax.imshow(share,interpolation="nearest",cmap=palette,vmin=0,vmax=1,aspect="auto")
    for i in range(len(sources)):
        for j in range(counts.shape[1]):
            value=share[i,j]
            label="0%" if not counts[i,j] else (f"{100*value:.1f}%".replace(".0%","%"))
            ax.text(j,i,label,ha="center",va="center",color="white" if value>.63 else "#183548",
                    fontsize=14,weight="bold" if value>=.5 else "normal")
    ax.set_xticks(range(counts.shape[1]),[f"C{i}" for i in range(counts.shape[1])],fontsize=14)
    ax.set_yticks(range(len(sources)),[SOURCE_LABELS.get(x,x) for x in sources],fontsize=14)
    ax.tick_params(axis="both",length=0,pad=8)
    ax.set_xlabel("K-means cluster", fontsize=14,labelpad=10)
    ax.set_title("Share of each source assigned to each cluster",fontsize=18,loc="left",pad=15)
    for v in np.arange(.5,counts.shape[1]-1): ax.axvline(v,color="white",lw=1.8)
    for v in np.arange(.5,counts.shape[0]-1): ax.axhline(v,color="white",lw=1.8)
    for spine in ax.spines.values(): spine.set_visible(False)
    save(fig,dest,"Source_by_cluster_K6")


def table(dest, ref, metrics):
    matrix=ref["rows"]
    paper_n=ref["n_sources"]
    n=len(metrics["sources"])
    body=[]
    for r in matrix:
        values=r["values"]
        method="FSD (paper)" if r["method"]=="Ours" else r["method"]
        body.append((method,str(paper_n),*[f"{v:.2f}" for v in values]))
    vals=[float(v[key]) for v in metrics["metrics"] for key in ("avg_accuracy","purity","nmi")]
    body.append(("FSD (this run)",str(n),*[f"{v:.3f}" for v in vals]))
    top=.867;bottom=.055
    rowh=(top-bottom)/(len(body)+2)
    x=np.array([.025,.181,.234]+[.234+.741/9*i for i in range(1,10)])
    fig,ax=plt.subplots(figsize=(13.4,5.6))
    fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    font={"fontfamily":"DejaVu Serif","fontsize":12.2}
    ax.text(x[0],.935,"Table 4.  Unsupervised source clustering",fontsize=19,va="center",
            fontfamily="DejaVu Serif",weight="bold")
    def line(a,b,y,lw=.68):ax.plot([x[a],x[b]],[y,y],color="#242a2e",lw=lw,solid_capstyle="butt")
    def label(a,b,y,text_,weight="normal",align="center",fontsize=None):
        opts=dict(font);opts["fontsize"]=fontsize or font["fontsize"]
        ax.text(x[a]+.006 if align=="left" else (x[a]+x[b])/2,y,text_,ha=align,va="center",weight=weight,**opts)
    ax.add_patch(Rectangle((x[0],bottom),x[-1]-x[0],rowh,facecolor="#eaf2f4",edgecolor="none"))
    line(0,-1,top,1.0);line(2,-1,top-rowh,.6);line(0,-1,top-2*rowh,.87);line(0,-1,bottom,1.0)
    for i in (0,1,2,5,8,11):
        ax.plot([x[i],x[i]],[bottom,top],color="#242a2e",lw=.55)
    label(0,1,top-rowh,"Method","bold",align="left");label(1,2,top-rowh,"N","bold")
    for j,mult in enumerate(("N","2N","4N")):
        start=2+j*3
        label(start,start+3,top-.5*rowh,f"K = {mult}","bold")
        for i,title in enumerate(["Avg. Acc.","Purity","NMI"]):
            label(start+i,start+i+1,top-1.5*rowh,title,"bold",fontsize=12.0)
    max_paper=np.max(np.array([r["values"] for r in matrix]),axis=0)
    for j,row in enumerate(body):
        y=top-(j+2.5)*rowh
        if j>=len(body)-2:line(0,-1,y+.5*rowh,.8)
        for i,value in enumerate(row):
            bold=j==len(body)-1 or (i>=2 and np.isclose(float(value),max_paper[i-2]))
            label(i,i+1,y,value,"bold" if bold else "normal",align="left" if i==0 else "center")
    save(fig,dest,"Table_04_clustering")


def main():
    p=argparse.ArgumentParser(description='draw')
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--embedding",type=Path,required=True,help="Directory produced by embed.py")
    p.add_argument("--paper",type=Path,required=True,help="Table 4 reference transcription")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    metrics=json.loads((args.results/"metrics.json").read_text())
    sources=metrics["sources"]
    assignments=read_csv(args.results/"assignments.csv")
    coords=read_csv(args.embedding/"tsne_coordinates.csv")
    by_id={r["image_id"]:r for r in coords}
    if len(coords)!=len(by_id) or len(assignments)!=len(by_id) or {r["image_id"] for r in assignments}!=set(by_id):
        raise ValueError("t-SNE image IDs and formal assignments differ")
    n=len(sources)
    counts=np.zeros((n,n),int)
    xy=[];clusters=[];colors=[]
    for row in assignments:
        point=by_id[row["image_id"]]
        if row["source"]!=point["source"]: raise ValueError("Mismatched source")
        j=int(row[f"cluster_k{n}"])
        xy.append((float(point["tsne_1"]),float(point["tsne_2"])))
        clusters.append("C"+str(j));colors.append(row["source"])
        counts[sources.index(row["source"]),j]+=1
    check=read_csv(args.results/f"contingency_k{n}.csv")
    assert np.array_equal(counts, np.array([[int(row[s]) for s in sources] for row in check]).T)
    xy=np.asarray(xy)
    plt.rcParams.update({"font.family":"DejaVu Sans","text.color":"#202629","axes.labelcolor":"#202629",
                         "xtick.color":"#202629","ytick.color":"#202629","axes.titlesize":18})
    scatter(args.output,"Figure_05_tSNE_sources",xy,np.asarray(colors),sources,SOURCE_COLORS,SOURCE_LABELS)
    keys=["C"+str(i) for i in range(n)]
    scatter(args.output,"tSNE_Kmeans_K6",xy,np.asarray(clusters),keys,dict(zip(keys,CLUSTER_COLORS)),{})
    heatmap(args.output,counts,sources)
    table(args.output,json.loads(args.paper.read_text(encoding="utf-8-sig")),metrics)
    validated=dict(n_images=len(assignments),source_counts=dict(collections.Counter(colors)),
                   cluster_counts=dict(collections.Counter(clusters)),all_ids_matched=True,
                   assignment_column=f"cluster_k{n}",cell_units="percentage within each source",
                   cluster_ids="zero-based original IDs",embedding=json.loads((args.embedding/"embedding.json").read_text()))
    (args.output/"visualization_metadata.json").write_text(json.dumps(validated,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
