import argparse
import os
import runpy
import subprocess
import sys

from project_paths import EXTENSION_ROOT


COMMANDS = {
    "manifest": "prepare_zeroshot_manifest.py",
    "evaluate": "evaluate_zeroshot.py",
    "download": "download_zeroshot_datasets.py",
    "sample-zip": "sample_remote_zip.py",
    "sample-zip-fast": "sample_remote_zip_fast.py",
    "sample-ossia": "sample_ossia_tar.py",
    "validate": "zeroshot_validation/run_validation.py",
    "verify-cache": "zeroshot_validation/verify_cached_scoring.py",
    "tables": "zeroshot_validation/export_tables.py",
    "reproduce": "reproduce/run.py",
    "metrics": "reproduce/check_metrics.py",
    "attribution": "reproduce/section42_paper_sources.py",
    "cluster-embed": "reproduce/visualize43/embed.py",
    "cluster-plot": "reproduce/visualize43/draw.py",
    "pca": "reproduce/visualize43/pca3d/compute.py",
    "pca-plot": "reproduce/visualize43/pca3d/static_plot.py",
    "pca-viewer": "reproduce/visualize43/pca3d/viewer.py",
}


def checks():
    successful = True
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([
        str(EXTENSION_ROOT), str(EXTENSION_ROOT.parent),
        environment.get("PYTHONPATH", ""),
    ])
    for folder in ("tests", "zeroshot_validation/tests", "zeroshot_monitor/tests"):
        result = subprocess.run([
            sys.executable, "-m", "unittest", "discover", "-s",
            str(EXTENSION_ROOT / folder), "-v",
        ], env=environment)
        successful = result.returncode == 0 and successful
    return 0 if successful else 1


def main():
    parser = argparse.ArgumentParser(description="Reproduction experiment tools")
    parser.add_argument("command", choices=[*COMMANDS, "check", "script"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "check":
        if args.arguments:
            parser.error("check takes no extra arguments")
        return checks()
    if args.command == "script":
        if not args.arguments:
            parser.error("script requires a Python path relative to Reproduction")
        relative, *arguments = args.arguments
    else:
        relative, arguments = COMMANDS[args.command], args.arguments
    target = (EXTENSION_ROOT / relative).resolve()
    if not target.is_relative_to(EXTENSION_ROOT) or target.suffix != ".py" or not target.is_file():
        parser.error("Choose an existing Python script inside Reproduction")
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), *arguments]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
