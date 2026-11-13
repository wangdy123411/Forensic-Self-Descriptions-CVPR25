from project_paths import DATA_ROOT, ARCHIVE_ROOT

import bisect
import csv
import hashlib
import json
import zlib
from collections import defaultdict
from pathlib import Path

out = Path(__file__).resolve().parent
data = DATA_ROOT
up = ARCHIVE_ROOT
run = data.parent / "NTU/PHD_2026/CE7454/Project2/Results/reproduction/section41_final_20261110/available_4.1_20261109T233647238685Z/4.1"

def rows(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def save(p, entries):
    with (out / p).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(entries[0]))
        w.writeheader()
        w.writerows(entries)

def lookup(rs, key):
    index = {v[key]: v for v in rs}
    assert len(index) == len(rs)
    return index


names = {"D": "DMID", "S": "Synthbuster", "O": "OSSIA", "A": "Authors' own", "G": "GRIP_CLIP_extended"}
paper = {"BigGAN":"D", "DALLE 2":"DS", "DALLE 3":"AS", "DALLE M":"D", "EG3D":"D",
         "FireFly":"S", "GigaGAN":"D", "GLIDE":"DS", "Guided Dif":"D", "Latent Dif":"D",
         "MJ v5":"S", "MJ v6":"A", "ProGAN":"O", "Proj.GAN":"O", "SD1.3":"S", "SD1.4":"OS",
         "SD1.5":"A", "SD2.1":"S", "SD3.0":"A", "SDXL":"AS", "StyleGAN":"O", "StyleGAN2":"O",
         "StyleGAN3":"O", "Tam.Xformer":"O"}
real = {"COCO2017":"COCO2017 val2017", "ImageNet-1k":"ILSVRC validation via GRIP",
        "ImageNet-22k":"timm/imagenet-22k-wds train-0000", "MIDB":"missing"}
pred = lookup(rows(run / "predictions.csv"), "image_id")
ev = lookup(rows(data / "manifests/eval_unique.csv"), "image_id")
prov = lookup([r for r in rows(data / "manifests/provenance.csv") if r["included_in_unique_eval"] == "1"], "image_id")
assert pred.keys() == ev.keys() == prov.keys() and len(pred) == 28992
assert len({p["sha256"] for p in pred.values()}) == len(pred)
groups = defaultdict(lambda: defaultdict(list))
for k,p in pred.items():
    v=prov[k]
    assert p["status"] == "ok"
    assert all(p[c] == v[c] == ev[k][c] for c in ("source","label","sha256","relative_path"))
    groups[p["source"]][v["original_relative_path"].split("/")[0]].append(float(p["fake_score"]))
assert not (groups.keys() - (paper.keys() | real.keys()))
refs = sorted(groups["ImageNet-22k"]["timm"])
assert len(refs)==1000
def auc(scores):
    return sum(bisect.bisect_left(refs,x)+bisect.bisect_right(refs,x) for x in scores)/(2*len(refs)*len(scores))

summary = []
for g in list(paper)+list(real):
    local=groups.get(g,{})
    sets={names[c] for c in paper[g]} if g in paper else {g}
    relation=("missing" if not local else "see real distribution" if g in real else
              "same named packages" if sets==set(local) else "partial substitution" if sets.intersection(local) else "substitution")
    ls=[x for vals in local.values() for x in vals]
    summary.append(dict(source=g,kind="synthetic" if g in paper else "real",paper_packages=" + ".join(names[c] for c in paper[g]) if g in paper else g,
                        paper_test_total=1000*(len(paper[g]) if g in paper else 1),
                        evaluated_packages="; ".join(f"{p}: {len(xs)}" for p,xs in sorted(local.items())),
                        evaluated_unique=len(ls),relation=relation,real_release=real.get(g,""),
                        local_IN22k_pooled_auc=f"{auc(ls):.7f}" if g in paper and ls else "",
                        local_IN22k_per_package_auc="; ".join(f"{p}: {auc(xs):.7f}" for p,xs in sorted(local.items())) if g in paper else ""))
save("cohorts.csv",summary)
detail=[]
for pack, prefix in [("DMID","TestSet/glide_text2img_valid/"),("Synthbuster","synthbuster/glide/")]:
    z={r["path"]:r for r in json.loads((up/pack/"metadata/original_zip_index.json").read_text(encoding="utf-8-sig")) if r["path"].startswith(prefix) and r["path"].lower().endswith(".png")}
    acquired = lookup(rows(up/pack/"manifest.csv"),"relative_path")
    members=set()
    for k, v in prov.items():
        if v["source"]!="GLIDE" or not v["original_relative_path"].startswith(pack+"/images/"):
            continue
        path=v["original_relative_path"][len(pack)+1:]
        member=path[len("images/"):]
        original=acquired[path]
        raw=(data/v["relative_path"]).read_bytes()
        sha=hashlib.sha256(raw).hexdigest()
        crc=f"{zlib.crc32(raw):08x}"
        assert member not in members and original["source"]=="GLIDE" and sha==original["sha256"]==v["sha256"]
        assert len(raw)==int(z[member]["size"]) and crc==z[member]["crc32"].lower()
        members.add(member)
        detail.append(dict(package=pack,archive_member=member,image_id=k,local_path=v["relative_path"],
                           sha256=sha,zip_crc32=crc,bytes=len(raw),fake_score=pred[k]["fake_score"]))
    assert members==set(z) and len(z)==1000
prompt_path = out.parent / "zeroshot_work/synthbuster_prompts.csv"
prompt_index = lookup(rows(prompt_path), "image name (matching Raise-1k)")
prompt_bytes = prompt_path.read_bytes()
archive_index = json.loads((up/"Synthbuster/metadata/original_zip_index.json").read_text())
ref_prompt = next(x for x in archive_index if x["path"] == "synthbuster/prompts.csv")
assert len(prompt_bytes) == ref_prompt["size"] and f"{zlib.crc32(prompt_bytes):08x}" == ref_prompt["crc32"]
assert set(prompt_index) == {Path(r["archive_member"]).stem for r in detail if r["package"]=="Synthbuster"}
for r in detail:
    s = Path(r["archive_member"]).stem
    v = prompt_index[s] if r["package"]=="Synthbuster" else {}
    if v:
        assert all(v[k].strip() for k in ("Prompt","Glide/SD 1.x CFG scale","Glide upsample temperature","Glide choice"))
    r.update(publisher_prompt_key=s if v else "", cfg_scale=v.get("Glide/SD 1.x CFG scale",""),
             upsample_temperature=v.get("Glide upsample temperature",""), image_choice=v.get("Glide choice",""))
save("glide_2000.csv",detail)
print("Verified",len(pred),"prediction rows;",len(detail),"GLIDE images cover both complete published ZIP groups and pass local SHA-256 + ZIP member size/CRC32 checks.")
