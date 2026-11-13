from project_paths import PROJECT_ROOT

import csv,json
from pathlib import Path
from collections import Counter
import numpy as np
out=Path(__file__).resolve().parent
p=PROJECT_ROOT / 'Results/reproduction/section42_author_v120_20261110/4.2/attribution.csv'
with p.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
known=np.array([r['supported']=='1' for r in rows]); hit=np.array([r['correct']=='1' for r in rows])
src=np.array([r['source'] for r in rows]); pred=np.array([r['candidate'] for r in rows])
report={}
for score in ['max_log_likelihood','confidence']:
    s=np.array([float(r[score]) for r in rows])
    a=np.sort(s[~known])
    prob=(np.searchsorted(a,s[known],'right')+np.searchsorted(a,s[known],'left'))/(2*len(a))
    tau=float(np.quantile(s[known],.05,method='lower'))
    accept=s>=tau
    report[score]=dict(known_unknown_auc=float(prob.mean()),au_oscr=float((prob*hit[known]).mean()),
        tau_95known=tau,known_accept=float(accept[known].mean()),unknown_accept=float(accept[~known].mean()),
        per_unsupported_source={t:dict(n=int(sum(src==t)),accepted=int(sum((src==t)&accept)),
              accepted_as=dict(Counter(pred[(src==t)&accept]).most_common(3))) for t in sorted(set(src[~known]))})
(out/'rejection_scores.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
for name,r in report.items():
    print(name,{k:v for k,v in r.items() if k!='per_unsupported_source'})
    for t in ('MJ v5','Proj.GAN','SD1.4','StyleGAN'): print(t,r['per_unsupported_source'][t])
