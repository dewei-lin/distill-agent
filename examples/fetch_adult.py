"""Provenance / refresh script for the example dataset.

`examples/adult_sample.csv` is a 6,000-row sample of the UCI **Adult**
(Census Income) dataset, committed to the repo so the demo runs out of the
box. This script documents where that data comes from and lets you
regenerate it.

Dataset: Becker, B. & Kohavi, R. (1996). Adult. UCI Machine Learning
Repository. https://doi.org/10.24432/C5XW20  — distributed under the
Creative Commons Attribution 4.0 license, which permits redistribution.

The Adult dataset is a natural fit for a data-cleaning demo: missing
values are encoded as the non-standard token "?", every categorical value
carries leading whitespace, column names use hyphens, `capital-gain` is
heavily zero-inflated with a coded 99999 top value, and `education` /
`education-num` are redundant encodings of each other.

Usage
-----
    # Option A — official UCI package (recommended)
    pip install ucimlrepo
    python examples/fetch_adult.py

    # Option B — already have adult.data locally
    python examples/fetch_adult.py --from-file path/to/adult.data

Either way it writes examples/adult_sample.csv (6,000 rows, retaining the
dataset's naturally-occurring exact-duplicate rows so the duplicate stage
has something real to find).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]
SAMPLE_ROWS = 6000
RANDOM_STATE = 42


def load_via_ucimlrepo():
    """Fetch the Adult dataset using the official UCI package."""
    from ucimlrepo import fetch_ucirepo  # type: ignore[import-untyped]

    adult = fetch_ucirepo(id=2)
    df = adult.data.features.copy()
    df["income"] = adult.data.targets.iloc[:, 0].values
    # Normalise column names to the canonical hyphenated form.
    df.columns = ADULT_COLUMNS[: len(df.columns)]
    return df


def load_from_file(path: Path):
    """Read a local raw adult.data file (headerless CSV)."""
    import pandas as pd

    return pd.read_csv(path, header=None, names=ADULT_COLUMNS, skipinitialspace=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate examples/adult_sample.csv")
    parser.add_argument(
        "--from-file",
        type=Path,
        default=None,
        help="Path to a local raw adult.data file (skips the UCI download).",
    )
    args = parser.parse_args()

    if args.from_file is not None:
        df = load_from_file(args.from_file)
        source = str(args.from_file)
    else:
        try:
            df = load_via_ucimlrepo()
            source = "UCI ML Repository (ucimlrepo, id=2)"
        except ImportError:
            print(
                "ucimlrepo not installed. Either:\n"
                "  pip install ucimlrepo\n"
                "or pass a local file:\n"
                "  python examples/fetch_adult.py --from-file adult.data",
                file=sys.stderr,
            )
            return 1

    # Sample, retaining naturally-occurring exact duplicates so the
    # duplicate-detection stage has real material.
    dup_mask = df.duplicated(keep=False)
    dup_rows = df[dup_mask]
    n_rest = max(SAMPLE_ROWS - len(dup_rows), 0)
    rest = df[~dup_mask].sample(n=min(n_rest, (~dup_mask).sum()), random_state=RANDOM_STATE)
    sample = (
        __import__("pandas")
        .concat([dup_rows, rest])
        .sample(frac=1, random_state=7)
        .reset_index(drop=True)
    )

    out_path = Path(__file__).resolve().parent / "adult_sample.csv"
    sample.to_csv(out_path, index=False)
    print(f"Source: {source}")
    print(f"Wrote {out_path} — {len(sample)} rows, "
          f"{int(sample.duplicated().sum())} exact-duplicate rows retained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
