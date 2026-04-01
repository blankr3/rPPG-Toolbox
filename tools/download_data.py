#!/usr/bin/env python
"""
CLI for downloading and verifying rPPG-Toolbox datasets.

Usage:
    # List all datasets and their status
    python tools/download_data.py --list

    # Download UBFC-rPPG (auto-download)
    python tools/download_data.py --dataset UBFC-rPPG

    # Check MMPD setup (prints instructions if missing)
    python tools/download_data.py --dataset MMPD

    # Override data directory
    RPPG_DATA_DIR=/my/data python tools/download_data.py --dataset UBFC-rPPG
    python tools/download_data.py --dataset UBFC-rPPG --data_dir /my/data
"""

import argparse
import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset.data_manager import ensure_dataset, list_datasets, DATASETS


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify rPPG-Toolbox datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        help="Dataset to download or verify",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all datasets and their status",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Override data directory (alternative to RPPG_DATA_DIR env var)",
    )
    args = parser.parse_args()

    if not args.list and not args.dataset:
        parser.print_help()
        sys.exit(1)

    if args.list:
        list_datasets(data_dir=args.data_dir)

    if args.dataset:
        try:
            path = ensure_dataset(args.dataset, data_dir=args.data_dir)
            print(f"\nDataset ready: {path}")
        except FileNotFoundError as e:
            print(f"\n{e}", file=sys.stderr)
            sys.exit(1)
        except ImportError as e:
            print(f"\n{e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
