"""CLI helper to download the CSIC 2010 dataset files (HTTPS only).

Usage:
    python scripts/download_data.py --dest data/raw
"""

from __future__ import annotations

import argparse
import pathlib

from http2vec.data.download import download_csic2010


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the CSIC2010 dataset files over HTTPS into --dest."
    )
    parser.add_argument(
        "--dest",
        default="data/raw",
        help="Destination directory for the downloaded files (default: data/raw).",
    )
    args = parser.parse_args()

    paths = download_csic2010(pathlib.Path(args.dest))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
