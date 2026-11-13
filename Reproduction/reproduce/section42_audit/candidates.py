from project_paths import PROJECT_ROOT, CODE_ROOT

from pathlib import Path
import os, sys, csv, json, hashlib
from collections import Counter
for k in ('MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[k]='4'
import numpy as np
import torch
torch.set_num_threads(4)
ROOT=PROJECT_ROOT
PROJECT=PROJECT_ROOT
RESULTS=PROJECT/'Results/reproduction'
AUTHOR=RESULTS/'section42_author_v120_20261110/4.2'
BASE=RESULTS/'section41_final_20261110/integrated'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE_ROOT))
from fsd.attribution import load_source_gmms
from fsd.projection import load_transforms, apply_projections

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def weights(name):
    dirs=(PROJECT/'Weights/fsd_v1.2.0', CODE_ROOT)
    p=next(d/name for d in dirs if (d/name).is_file())
    with p.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==model['weight_sha256'][name]
    return p

def describe(mask,allowed,true,known,reference_threshold=None):
    sub=ll[mask][:,allowed]
    which=np.array(names)[allowed][sub.argmax(1)]
    score=sub.max(1); y=true[mask]; k=known[mask]; u=~k
    assert sum(k) and sum(u)
    hit=which==y
    unsorted=score[u]; uns=np.sort(unsorted)
    def wins(a):
        return (np.searchsorted(uns,a,side='left')+np.searchsorted(uns,a,side='right'))/(2*len(uns))
    recall={t:float(hit[k&(y==t)].mean()) for t in sorted(set(y[k]))}
    tau=float(np.quantile(score[k],.05,method='lower'))
    def at(t):
        return dict(tau=t,known_accept=float((score[k]>=t).mean()),unknown_accept=float((score[u]>=t).mean()),
                    correct_known_accept=float(((score[k]>=t)&hit[k]).mean()),
                    unknown_accept_by_source={s:float((score[u&(src[mask]==s)]>=t).mean()) for s in sorted(set(src[mask][u]))})
    return dict(candidates=np.array(names)[allowed].tolist(),nknown=int(sum(k)),nunknown=int(sum(u)),
                macro_accuracy=float(np.mean(list(recall.values()))),per_known_source=recall,
                known_unknown_auc=float(wins(score[k]).mean()),au_oscr=float((wins(score[k])*hit[k]).mean()),
                at_95_percent_known_accept_test_descriptive=at(tau),
                at_same_baseline_threshold=at(reference_threshold if reference_threshold is not None else tau)), which, score

model=json.loads((AUTHOR/'model.json').read_text()); names=model['classes']
arch=read(AUTHOR/'attribution.csv'); idx=read(BASE/'feature_index.csv')
f=np.load(BASE/'raw_fsd_960.npy',mmap_mode='r')
assert len(arch)==len(idx)==len(f)==28992
assert all(a['image_id']==b['image_id'] and a['source']==b['source'] and int(b['row'])==i for i,(a,b) in enumerate(zip(arch,idx)))
src=np.array([r['source'] for r in arch]); targets=np.array([r['target'] for r in arch])
supported=targets!=''; represented=sorted(set(targets[supported])); assert len(represented)==8
t=load_transforms(weights('fsd_transforms.pt'),'cpu')+load_transforms(weights('attribution_transforms.pt'),'cpu')
gm=load_source_gmms(weights('source_gmms.pt'),'cpu'); assert list(gm)==names
ll=np.empty((len(f),len(names)),np.float64)
with torch.inference_mode():
    for start in range(0,len(f),256):
        x=apply_projections(torch.from_numpy(f[start:start+256].copy()),t)
        ll[start:start+len(x)]=torch.stack([m.score_samples(x) for m in gm.values()],1).numpy()
assert np.isfinite(ll).all()
arg=np.array(names)[ll.argmax(1)]; assert np.array_equal(arg,[r['candidate'] for r in arch])
max_delta=float(max(abs(ll.max(1)-np.array([float(r['max_log_likelihood']) for r in arch]))))
def select(name): return [i for i,n in enumerate(names) if n in name]
all_rows=np.ones(len(src),bool)
unknown_paper={'StyleGAN2','SD3.0','DALLE 3','FireFly'}
unknown_models={'StyleGAN2','Stable Diffusion 3','DALL-E 3','Adobe Firefly'}
known3={'ImageNet-1k','ProGAN','StyleGAN3'}
sub_paper=np.isin(src,list(known3|unknown_paper))
pknown=np.isin(src,list(known3))
report={'score_parity':dict(top1_agrees=len(arch),max_abs_loglik_delta=max_delta),'comparisons':{}}
for tag,mask,known,base,ext in [('represented_8_vs_all_14',all_rows,supported,represented,names),
    ('paper_overlap_3_vs_10',sub_paper,pknown,['Real','ProGAN','StyleGAN3'],[n for n in names if n not in unknown_models])]:
    before,w0,s0=describe(mask,select(base),targets,known)
    after,w1,s1=describe(mask,select(ext),targets,known,before['at_same_baseline_threshold']['tau'])
    k=known[mask]; truth=targets[mask]; changed=(w0==truth)&(w1!=truth)&k
    assert (s1>=s0).all()
    report['comparisons'][tag]=dict(before=before,after=after,
        previously_correct_lost=int(changed.sum()),loss_to_added_candidates=dict(Counter(w1[changed])),
        mean_max_score_increase_known=float((s1[k]-s0[k]).mean()),mean_max_score_increase_unknown=float((s1[~k]-s0[~k]).mean()))
replay=report['comparisons']['represented_8_vs_all_14']['after']
ref=json.loads((AUTHOR/'metrics.json').read_text())
assert abs(replay['macro_accuracy']-ref['macro_accuracy'])<1e-12
assert abs(replay['au_oscr']-ref['open_set_on_release_coverage']['au_oscr'])<1e-11
prov={r['image_id']:r['original_relative_path'].split('/')[0] for r in read(ROOT/'7454data/manifests/provenance.csv')}
groups={}
for s in ('DALLE 3','SDXL','StyleGAN','SD1.4','StyleGAN3'):
    for a in arch:
        if a['source']==s: groups.setdefault((s,prov[a['image_id']]),Counter())[a['candidate']]+=1
report['package_predictions']={s+' | '+p:dict(c.most_common()) for (s,p),c in groups.items()}
with (OUT/'candidate_audit.json').open('w',encoding='utf-8') as o:
    json.dump(report,o,ensure_ascii=False,indent=2)
print('Top-1 parity',len(arch),'max score delta',max_delta)
for tag,run in report['comparisons'].items():
    for side in ('before','after'):
        a=run[side]
        print(tag,side, 'accuracy',a['macro_accuracy'],'known/unknown AUC',a['known_unknown_auc'],'AU-OSCR',a['au_oscr'],'unknown accept @95%',a['at_95_percent_known_accept_test_descriptive']['unknown_accept'],'unknown accept at baseline threshold',a['at_same_baseline_threshold']['unknown_accept'])
    print('lost',run['previously_correct_lost'],run['loss_to_added_candidates'])
print('source breakdown',report['package_predictions'])
