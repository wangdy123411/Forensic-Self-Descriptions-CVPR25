"""Run the unchanged upstream public API in an isolated process."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from fsd.detector import FSDDetector
from an.robustness.transforms import CONDITION_MAP, apply_condition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--condition", default="clean", choices=CONDITION_MAP)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = dict(
        started_utc=datetime.now(timezone.utc).isoformat(),
        image=str(args.image),
        image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
        condition=args.condition,
        device=args.device,
        torch=torch.__version__,
        api="Unmodified fsd.detector.FSDDetector.attribute",
        source_modified=False,
        default_max_size=1024,
        requested_max_size=args.max_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        detector = FSDDetector.load(
            ROOT / "an/weights/v1.2.0", device=args.device, attribution=True
        )
        detector.config["fsd"]["max_size"] = args.max_size
        report["fsd_config"] = dict(detector.config["fsd"])
        with Image.open(args.image) as opened:
            image = apply_condition(opened.convert("RGB"), CONDITION_MAP[args.condition])
        report["input_size"] = list(image.size)
        report["patch_storage_bytes"] = sum(
            (args.max_size // (2**level)) ** 2 * 8 * 121 * 8 for level in range(3)
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            free, total = torch.cuda.mem_get_info()
            report["cuda_before"] = dict(
                free_bytes=free, total_bytes=total, gpu=torch.cuda.get_device_name()
            )
        print(json.dumps(dict(status="starting_literal_upstream", **report)), flush=True)
        inference_start = time.perf_counter()
        result = detector.attribute(image)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        report["inference_seconds"] = time.perf_counter() - inference_start
        report["result"] = asdict(result)
        report["status"] = "success"
    except Exception as exc:
        report["status"] = (
            "cuda_out_of_memory" if isinstance(exc, torch.OutOfMemoryError) else "error"
        )
        report["exception_type"] = type(exc).__name__
        report["exception"] = str(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        report["process_seconds"] = time.perf_counter() - started
        if args.device.startswith("cuda") and torch.cuda.is_initialized():
            report["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] in {"success", "cuda_out_of_memory"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
