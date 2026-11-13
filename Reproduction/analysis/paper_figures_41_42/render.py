from project_paths import PROJECT_ROOT, EXTENSION_ROOT

import argparse
import csv
import json
import re
from pathlib import Path

import pdfplumber
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

PROJECT=PROJECT_ROOT
ROOT=PROJECT/"Results/reproduction"
PROTOCOL=PROJECT/"Results/zeroshot_20261106_protocol"
RUN41=ROOT/"section41_final_20261110"
T=RUN41/json.loads((RUN41/"delivery.json").read_text())["run"]/"4.1/tables"
RELEASE42=ROOT/"section42_author_v120_20261110/4.2"
M42=json.loads((RELEASE42/"metrics.json").read_text())
SOURCE_DATA=Path(__file__).resolve().parent/"build"
OVERLAP42=EXTENSION_ROOT/"reproduce/section42_paper_overlap/metrics.json"
SUBSTITUTED42=EXTENSION_ROOT/"reproduce/section42_paper_substituted/metrics.json"
REF=json.loads((PROTOCOL/"paper_reference.json").read_text())
REAL=REF["real_source_order"]
GEN=REF["generator_order"]
LOCAL="FSD local (21 gen.)"
PAPER="FSD (paper)"
MATCH="FSD paper (21 gen.)"
ASSETS=[]
for font,filename in [("T","times.ttf"),("B","timesbd.ttf"),("I","timesi.ttf")]:
    pdfmetrics.registerFont(TTFont(font,str(Path("C:/Windows/Fonts")/filename)))


def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def fmt(x):
    return "—" if x is None or x=="" else f"{float(x):.3f}"


def method(s):
    s=s.replace("Abadyetal.","Abady et al.").replace("Fangetal.","Fang et al.")
    return PAPER if s=="Ours" else re.sub(r"\s*\[\d+\]", "", s)


def unpack(pdf_table):
    hdr,middle,last=pdf_table.extract()
    data=[list(vals) for vals in zip(*[cell.splitlines() for cell in middle],strict=True)]+[last]
    for row in data:
        row[0]=method(row[0])
        row[1:]=[re.sub(r"(?<=\d)\("," (",x) for x in row[1:]]
    return hdr,data


def text(c,x,y,s,size,font="T",align="left",color="#17191b"):
    c.setFont(font,size);c.setFillColor(HexColor(color))
    getattr(c,{"left":"drawString","center":"drawCentredString","right":"drawRightString"}[align])(x,y,str(s))


def grid(c,x,top,widths,headers,body,*,rh=20,fs=11,left=(0,),highlight=(),lines=(),leaders=(),merges=()):
    bounds=[x]
    for w in widths: bounds.append(bounds[-1]+w)
    table=[headers]+body
    end=top-rh*len(table)
    for j in highlight:
        c.setFillColor(HexColor("#e9f2f4"))
        c.rect(x,top-(j+2)*rh,bounds[-1]-x,rh,fill=1,stroke=0)
    c.setStrokeColor(HexColor("#27292b"));c.setLineWidth(.57)
    for b in bounds: c.line(b,top,b,end)
    for y in (top,top-rh,end): c.line(x,y,bounds[-1],y)
    for j,startcol in lines:
        c.line(bounds[startcol],top-(1+j)*rh,bounds[-1],top-(1+j)*rh)
    skip={(j+1,col) for col,start,n,s in merges for j in range(start,start+n)}
    for ri,row in enumerate(table):
        assert len(row)==len(widths),(ri,row)
        for ci,s in enumerate(row):
            if (ri,ci) in skip:continue
            font="B" if ri==0 or (ri-1) in highlight or (ri-1,ci) in leaders else "T"
            w=widths[ci]; size=fs
            while pdfmetrics.stringWidth(str(s),font,size)>w-7:
                size-=.2
                if size<fs*.83: raise ValueError(f"cell too narrow: {s}")
            align="left" if ci in left else "center"
            tx=bounds[ci]+4.5 if align=="left" else bounds[ci]+w/2
            text(c,tx,top-(ri+.5)*rh-size*.32,s,size,font,align)
    for col,start,n,s in merges:
        parts=s.split("\n")
        cy=top-(1+start+n/2)*rh
        for j,part in enumerate(parts):
            text(c,bounds[col]+5,cy+(len(parts)-1-2*j)*fs*.54-fs*.30,part,fs)
    return end


def maximums(data,firstcol,end=None):
    pairs=set()
    for j in range(firstcol,len(data[0]) if end is None else end):
        parsed=[float(str(r[j]).split()[0]) for r in data]
        pairs.update((i,j) for i,x in enumerate(parsed) if x==max(parsed))
    return pairs


def page(c,key,title,widths,hdr,body,*,caption_num,fs=12,rh=21,**args):
    pad=20; width=sum(widths)+2*pad
    height=61+(1+len(body))*rh
    c.setPageSize((width,height))
    text(c,pad,height-23,f"{caption_num}.",15,"B")
    capwidth=pdfmetrics.stringWidth(f"{caption_num}.","B",15)
    text(c,pad+capwidth+6,height-23,title,15)
    grid(c,pad,height-43,widths,hdr,body,fs=fs,rh=rh,**args)
    c.showPage();ASSETS.append(key)


def open_set_table(c, paper, overlap_metrics, substituted_metrics):
    overlap=json.loads(overlap_metrics.read_text(encoding="utf-8"))
    substituted=json.loads(substituted_metrics.read_text(encoding="utf-8"))
    supplement=substituted.get("supplement") or {}
    supplemental_count=len(supplement.get("records",[]))
    raw=paper.pages[5].find_tables()[0].extract()
    hdr=raw[0]; b=[]; merges=[]
    for category,*cols in raw[1:-1]:
        items=list(zip(*[s.splitlines() for s in cols],strict=True))
        merges.append((0,len(b),len(items)+(1 if category=="Open-set" else 0),category.replace("Metric-\nlearning","Metric-learning")))
        b.extend([["",method(m),*vals] for m,*vals in items])
    b.append(["",PAPER,*raw[-1][2:]])

    widths=[110,138,94,102,89]
    metric_widths=[108,111,109,113,92]
    metric_headers=["Known images","Unknown images","Macro accuracy","F1–CRR area","AU-OSCR"]
    extra_height=16 if supplemental_count else 0
    c.setPageSize((sum(widths)+40,575+extra_height))
    c.translate(0,extra_height)
    text(c,20,550,"Table 3. Open-set source attribution",16,"B")
    text(c,20,524,"Paper protocol: 5 known and 4 unknown sources",13,"B")
    grid(c,20,513,widths,hdr,b,fs=12,rh=20,left=(0,1),merges=merges,
         lines=[(2,0),(4,0),(6,0),(9,1)],leaders=maximums(b,2))

    def metric_values(metrics):
        return [f"{metrics['known_n']:,}",f"{metrics['unknown_n']:,}",
                fmt(metrics["known_macro_accuracy"]),fmt(metrics["au_crr_fang_macro_f1_vs_crr"]),
                fmt(metrics["au_oscr"])]

    text(c,20,268,"Frozen FSD v1.2.0: 14 candidates, 8 with known test images",13,"B")
    grid(c,20,258,metric_widths,metric_headers,[metric_values(M42["open_set_on_release_coverage"])],
         fs=12,rh=21,left=(),highlight=(0,))

    text(c,20,191,"Frozen FSD v1.2.0 protocols (known + unknown sources)",13,"B")
    comparisons=[]
    for name,metrics in [("Paper subset",overlap),("Substitutions",substituted)]:
        counts=f"{len(metrics['known_sources'])} + {len(metrics['unknown_sources'])}"
        comparisons.append([f"{name} ({counts})",*metric_values(metrics)])
    grid(c,20,181,[110,78,83,96,87,79],["Protocol",*metric_headers],comparisons,
         fs=11.5,rh=21,left=(0,),highlight=(0,1),lines=[(1,0)])
    names={"ImageNet-1k":"ImageNet-1k (Real GMM)","DALLE 3":"DALL-E 3",
           "FireFly":"Adobe Firefly","StyleGAN":"StyleGAN (v1)"}
    def source_list(sources):
        return ", ".join(names.get(s,s) for s in sources)

    def replacements(unavailable):
        return ", ".join(f"{substituted['substitutions'][s]} for {names.get(s,s)}" for s in unavailable)

    missing_known=overlap["unavailable_paper_known_gmms"]
    missing_unknown=overlap["unavailable_paper_unknown_samples"]
    notes=[
        ("Known sources in both runs",source_list(overlap["known_sources"])),
        ("Unknown sources in both runs",source_list(overlap["unknown_sources"])),
        ("Missing known GMMs",source_list(missing_known)),
        ("Missing unknown test images",source_list(missing_unknown)),
        ("Known substitutions",replacements(missing_known)),
        ("Unknown substitution",replacements(missing_unknown)),
    ]
    if supplemental_count:
        regular_count=substituted["source_counts"]["FireFly"]-supplemental_count
        notes.append(("Firefly images in 5 + 4",f"{regular_count:,} Synthbuster + {supplemental_count:,} MNW Firefly v2"))
    for i,(label,detail) in enumerate(notes):
        y=101-i*16
        text(c,20,y,label,11.4,"B")
        text(c,179,y,detail,11.4)
    c.showPage()
    ASSETS.append("Table_03_open_set")


def main(build,overlap_metrics,substituted_metrics,table3_only=False):
    build.mkdir(parents=True,exist_ok=True)
    paper=pdfplumber.open(PROJECT/"2503.21003v1.pdf")
    if table3_only:
        c=Canvas(str(build/"Table_03_open_set.pdf"),pageCompression=1)
        c.setTitle("Table 3. Open-set source attribution")
        open_set_table(c,paper,overlap_metrics,substituted_metrics)
        c.save()
        paper.close()
        return
    d=json.loads((SOURCE_DATA/"4_2_diagnostic.json").read_text())
    c=Canvas(str(build/"FSD_tables_4_1_4_2.pdf"),pageCompression=1)
    c.setTitle("FSD · Detection and open-set attribution | published and reproduced tables")
    t1,t2=paper.pages[4].find_tables()
    hdr,b=unpack(t1)
    local=rows(T/"main_table_1_macro_auc.csv")
    matched=next(r for r in local if r["value_kind"]=="paper_reference_subset")
    meas=next(r for r in local if r["value_kind"]=="local_measured")
    matchedrow=[MATCH]+[fmt(matched[k]) for k in REAL+["Average"]]
    b_local=[LOCAL]+[fmt(meas[k]) for k in REAL+["Average"]]
    b[-1][0]="FSD paper (24 gen.)"
    page(c,"Table_01_mean_AUC","Zero-shot detection: mean AUC",
         [137,80,80,80,77,76],hdr,b+[matchedrow,b_local],
         caption_num="Table 1",highlight=[10],lines=[(8,0),(9,0),(10,0)],leaders=maximums(b,1))

    hdr,b=unpack(t2)
    vals=rows(T/"main_table_2_worst_auc.csv")
    assert [v["real_source"] for v in vals]==REAL
    b_local=[LOCAL]+[fmt(v["measured_worst_auc"])+(f" ({v['measured_generators']})" if v["measured_worst_auc"] else "") for v in vals]
    b[-1][0]="FSD paper (24 gen.)"
    page(c,"Table_02_worst_AUC","Worst-case detection AUC and generator",
         [122,115,115,115,115],hdr,b+[b_local],
         caption_num="Table 2",highlight=[9],lines=[(8,0),(9,0)],leaders=maximums(b,1))

    open_set_table(c,paper,overlap_metrics,substituted_metrics)

    heads=["Method","Avg.","ProG","Prj.G","SG","SG2","SG3","BigG","GigaG","EG3D","Tm.Xf",
           "GLIDE","G.Dif.","L.Dif.","SD1.3","SD1.4","SD1.5","SD2.1","SDXL","SD3.0","DLEM","DLE2","DLE3","MJv5","MJv6","Firefly"]
    appendix_text=(PROTOCOL/"page_15_plumber.txt").read_text("utf-8-sig").splitlines()
    for number in (10,11,12,13):
        ref=REF["appendix_tables"][str(number)]
        real=ref["real_source"]
        lines=appendix_text[next(i for i,s in enumerate(appendix_text) if s.startswith(f"Table{number}."))+2:][:9]
        b=[]
        for original,parsed in zip(ref["rows"],lines,strict=True):
            row=[method(original["method"]),fmt(original["printed_average"]),*[fmt(original["auc"][s]) for s in GEN]]
            assert parsed.split()[1:]==row[1:],(number,original["method"])
            b.append(row)
        values=rows(next(T.glob(f"appendix_table_{number}_*.csv")))
        assert [v["generator"] for v in values]==GEN
        loc=[LOCAL,fmt(meas[real]),*[fmt(v["local_measured_auc"]) for v in values]]
        b[-1][0]="FSD paper (24 gen.)"
        if real=="MIDB":loc[0]="FSD local (unavailable)"
        page(c,f"Table_{number:02d}_{real.replace('-','_')}",f"{real}: zero-shot AUC by generator",
             [114,43]+[38]*24,heads,b+[loc],
             caption_num=f"Table {number}",fs=11.4,rh=20,highlight=[9],lines=[(8,0),(9,0)],leaders=maximums(b,1))


    names=d["names"]; official=d["runs"]["as_fitted"]; pooled=d["runs"]["shared_across_sources"]
    c.setPageSize((623,356));text(c,20,331,"Local five-source refit: covariance diagnostic",16,"B")
    text(c,20,305,"Predicted source on 3,000 known test images",13,"B")
    b=[[n,*cnt,f"{100*official['metrics']['known_source_accuracy'][n]:.1f}%"] for n,cnt in zip(names,official["confusion_known"])]
    grid(c,20,294,[110,83,77,80,75,77,81],["True source",*names,"Recall"],b,fs=11.8,rh=18,leaders={(j,j+1) for j in range(5)})
    text(c,20,166,"Covariance of the 960-dimensional FSD",13,"B")
    counts=d["training_counts"];floor=d["at_regularization_floor"]
    grid(c,20,156,[183]+[80]*5,["Quantity",*names],[["Training images",*[counts[n] for n in names]],
        ["Eigenvalues at the 10⁻⁶ floor",*[floor[n] for n in names]]],fs=11.8,rh=18)
    text(c,20,81,"Covariance comparison on the same test split",13,"B")
    methods=[("Local refit: separate covariance per source",official),("Diagnostic: one pooled covariance",pooled)]
    b=[[n,fmt(a["metrics"]["known_macro_accuracy"]),fmt(a["metrics"]["au_oscr"]),fmt(a["metrics"]["known_unknown_roc_auc"])] for n,a in methods]
    grid(c,20,71,[232,111,94,146],["Model","Known accuracy","AU-OSCR","Known/unknown AUC"],b,
         fs=11.8,rh=18,highlight=[1],lines=[(1,0)])
    c.showPage();ASSETS.append("Diagnostic_4_2")
    audit=json.loads((RELEASE42/"audit.json").read_text())
    c.setPageSize((590,394))
    text(c,20,368,"Frozen FSD v1.2.0: 14-way source attribution",16,"B")
    text(c,20,342,"Top-1 accuracy on eight represented classes",13,"B")
    counts=[[a["source"],a["n"],a["correct"],f"{100*a['recall']:.2f}%"] for a in M42["closed_set_per_class"]]
    counts.extend([["All images (micro)",f"{M42['supported_images']:,}",f"{sum(a['correct'] for a in M42['closed_set_per_class']):,}",
                    f"{100*M42['micro_accuracy']:.2f}%"],
                   ["Class mean (macro)","—","—",f"{100*M42['macro_accuracy']:.2f}%"]])
    grid(c,20,332,[177,111,118,141],["Target","Images","Correct","Top-1"],counts,fs=12,rh=18,
         lines=[(8,0)],highlight=[9])
    text(c,20,110,"Same 1,500 held-out images",13,"B")
    subset=audit["paired_previous_known_tests"]
    paired=[[k,x["n"],f"{100*x['local_refit_correct']/x['n']:.1f}%",f"{100*x['author_14way_correct']/x['n']:.1f}%"] for k,x in subset.items()]
    grid(c,20,99,[171,96,132,148],["Dataset","Images","Local five-way refit","Frozen 14-way model"],paired,fs=12,rh=20)
    c.showPage();ASSETS.append("Author_weights_4_2")
    c.save()
    (build/"pages.json").write_text(json.dumps(ASSETS,indent=2),encoding="utf-8")
    print(ASSETS)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description='render')
    parser.add_argument("--output",type=Path,default=Path("tmp/clean_tables_41_42"))
    parser.add_argument("--overlap-metrics",type=Path,default=OVERLAP42)
    parser.add_argument("--substituted-metrics",type=Path,default=SUBSTITUTED42)
    parser.add_argument("--table-3-only",action="store_true")
    args=parser.parse_args()
    main(args.output,args.overlap_metrics,args.substituted_metrics,args.table_3_only)
