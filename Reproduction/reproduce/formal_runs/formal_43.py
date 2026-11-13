from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import tempfile
import json
import fcntl
import torch
from formal_42 import PARENT, load_merged, read_csv
from inputs import CLUSTER, PRODUCTION_HASH
from clustering import cluster
from zeroshot_common import file_hash, write_json


def main():
    output=PARENT/"section43_final_20261110"
    output.mkdir(exist_ok=True)
    with (output/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        final=output/"4.3"
        if (final/"checksums.json").is_file():
            print(final)
            return
        samples,matrix_digest=load_merged()
        selected={r["image_id"]:r["source"] for r in samples.rows if r["source"] in CLUSTER}
        counts=Counter(selected.values())
        torch.set_num_threads(4)
        torch.manual_seed(2026)
        torch.backends.cuda.matmul.allow_tf32=False
        gpu=torch.cuda.get_device_name(0)
        if "Blackwell" not in gpu or len(selected)!=7997 or len(counts)!=6:
            raise ValueError((gpu,counts))
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            work=Path(tmp)/"4.3"
            work.mkdir()
            cluster(samples,work,"cuda:0",2026,n_init=10)
            rows=read_csv(work/"assignments.csv")
            if {r["image_id"]:r["source"] for r in rows}!=selected or len(rows)!=len(selected):
                raise ValueError("Cluster assignments do not match the evaluation population.")
            metrics=json.loads((work/"metrics.json").read_text())
            if [r["observed_k"] for r in metrics["metrics"]]!=[6,12,24]:
                raise ValueError("Unexpected number of observed clusters")
            write_json(work/"run.json",dict(gpu=gpu,utc=datetime.now(timezone.utc).isoformat(),source_counts=dict(counts),
                      feature_matrix_sha256=matrix_digest,manifest_sha256=PRODUCTION_HASH,n=len(rows),dimensions=960))
            write_json(work/"checksums.json",{p.name:file_hash(p) for p in sorted(work.iterdir())})
            work.rename(final)
        print(final)
        print((final/"metrics.json").read_text())


if __name__=="__main__":
    main()
