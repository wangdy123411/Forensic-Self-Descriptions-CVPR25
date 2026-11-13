"""Regenerate tables, paired bootstrap intervals, plots, and report for an AN run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robustness.analysis import analyze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing records.jsonl")
    parser.add_argument(
        "--bootstrap", type=int, default=1000, help="Paired, source-stratified bootstrap draws"
    )
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument(
        "--source-mapping", type=Path, help="JSON mapping dataset source to exact checkpoint source"
    )
    parser.add_argument(
        "--skip-content-sensitivity",
        action="store_true",
        help="Skip the separate archive-stem proxy cluster-bootstrap sensitivity",
    )
    args = parser.parse_args()
    result = analyze(
        args.run_dir, bootstrap=args.bootstrap, seed=args.seed, source_mapping=args.source_mapping
    )
    if (
        not args.skip_content_sensitivity
        and args.bootstrap > 0
        and any(
            row.get("n_real", 0) > 0 and row.get("n_fake", 0) > 0
            for row in result.get("macro_results", [])
        )
    ):
        from robustness.content_sensitivity import analyze_content_sensitivity

        sensitivity = analyze_content_sensitivity(
            args.run_dir, bootstrap=args.bootstrap, seed=args.seed
        )
        report = args.run_dir / "report.md"
        with report.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n## Supplementary dependence check\n\n"
                f"The {sensitivity['n_fake_originals']} synthetic test originals have "
                f"{sensitivity['n_fake_proxy_clusters']} distinct archive-stem proxy groups; "
                f"{sensitivity['n_cross_generator_shared_proxy_clusters']} stems repeat across generators. "
                "These are unconfirmed reference-content proxies. A separate sensitivity analysis "
                "resamples shared stems together while retaining the primary point estimates and "
                "the original image/source-stratified intervals.\n\n"
                "[Proxy-cluster sensitivity and interpretation](content_sensitivity.md) · "
                "[Supplementary interval table](content_sensitivity_metrics.csv).\n"
            )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "n_clean_test",
                    "n_common_test",
                    "n_calibration_real_clean",
                    "warnings",
                )
            },
            indent=2,
        )
    )
    print(f"Report: {args.run_dir.resolve() / 'report.md'}")


if __name__ == "__main__":
    main()
