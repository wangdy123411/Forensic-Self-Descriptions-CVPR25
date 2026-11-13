"""Select native images from official Synthbuster/RAISE ZIPs via HTTP ranges."""

from pathlib import Path
import argparse

from robustness.data import acquire_subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["synthbuster", "raise", "both"], default="synthbuster"
    )
    parser.add_argument("--per-source", type=int, default=100)
    parser.add_argument("--real-test", type=int, default=100)
    parser.add_argument("--real-calibration", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument(
        "--raise-url",
        default=None,
        help="Override the public RAISE archive URL; TLS verification remains enabled",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    datasets = ["synthbuster", "raise"] if args.dataset == "both" else [args.dataset]
    for dataset in datasets:
        acquire_subset(
            root,
            dataset,
            args.per_source,
            args.real_test,
            args.real_calibration,
            args.seed,
            args.list_only,
            args.raise_url if dataset == "raise" else None,
        )


if __name__ == "__main__":
    main()
