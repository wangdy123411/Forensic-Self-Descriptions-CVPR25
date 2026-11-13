"""Record GPU utilization/temperature/memory while a long experiment runs."""

from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=6)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["utc", "gpu_utilization_percent", "memory_used_mib", "temperature_c", "power_w"]
    exists = args.output.exists()
    deadline = time.monotonic() + args.hours * 3600
    with args.output.open("a", newline="", buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        while time.monotonic() < deadline and not args.output.with_suffix(".stop").exists():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if result.returncode == 0:
                parts = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
                if len(parts) == 4:
                    writer.writerow(
                        dict(zip(fields, [datetime.now(timezone.utc).isoformat()] + parts))
                    )
                    stream.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
