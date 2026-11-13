import csv
import fcntl
import json
from pathlib import Path
import tempfile
from datetime import datetime,timezone
from collections import Counter

import torch
from formal_42 import ROOT, PARENT, load_merged, read_csv
from experiments import released
from inputs import WEIGHTS, KNOWN, PRODUCTION_HASH
from zeroshot_common import file_hash, write_json
from fsd import FSDDetector

OUT=PARENT/"section42_author_v120_20261110"


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        final=OUT/"4.2"
        if (final/"checksums.json").is_file():
            return
        samples,digest=load_merged()
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        gpu=torch.cuda.get_device_name(0)
        if "Blackwell" not in gpu:
            raise ValueError(gpu)
        with tempfile.TemporaryDirectory(dir=OUT) as temp:
            stage=Path(temp)/"4.2"
            stage.mkdir()
            released(samples,stage,"cuda:0")
            pred=read_csv(stage/"attribution.csv")
            if [r["image_id"] for r in pred]!=[r["image_id"] for r in samples.rows]:
                raise ValueError("Image list does not match full integrated run")
            by_id={r["image_id"]:r for r in pred}
            model=FSDDetector.load(weights_dir=WEIGHTS,device="cuda:0",attribution=True)
            checks=[]
            for key in ("calibration","in22k","sd21","tam","style1","style2","style3","sdxl"):
                row=read_csv(ROOT/f"data/manifests/{key}.csv")[0]
                img=ROOT/"data"/row["relative_path"]
                assert file_hash(img)==row["sha256"]
                cached=by_id[row["image_id"]]
                reference=model.attribute(img)
                assert reference.source==cached["candidate"]
                delta=abs(reference.confidence-float(cached["confidence"]))
                if delta>1e-7: raise ValueError((key,delta))
                checks.append(dict(source=row["source"],image_id=row["image_id"],candidate=reference.source,
                                   confidence_abs_difference=delta))
            old=read_csv(PARENT/"section42_final_20261110/4.2/attribution.csv")
            paired=[(r,by_id[r["image_id"]]) for r in old if r["source"] in KNOWN and by_id[r["image_id"]]["supported"]=="1"]
            paired_metrics={}
            for source in KNOWN:
                subset=[(o,n) for o,n in paired if o["source"]==source]
                if subset:
                    paired_metrics[source]=dict(n=len(subset),local_refit_correct=sum(o["candidate"]==source for o,n in subset),
                                           author_14way_correct=sum(n["correct"]=="1" for o,n in subset))
            write_json(stage/"audit.json",dict(gpu=gpu,utc=datetime.now(timezone.utc).isoformat(),
                       input_matrix_sha256=digest,manifest_sha256=PRODUCTION_HASH,training_images_used=0,
                       native_image_api_checks=checks,paired_previous_known_tests=paired_metrics,
                       class_mapping={r["source"]:r["target"] or None for r in pred}))
            write_json(stage/"checksums.json",{p.name:file_hash(p) for p in stage.iterdir()})
            stage.rename(final)
        print((final/"metrics.json").read_text(),flush=True)
        print((final/"audit.json").read_text(),flush=True)

if __name__=="__main__":
    main()
