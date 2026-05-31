"""File IO for DataClean Agent.

Load datasets by extension; save cleaned outputs in the original file
format AND as portable CSV. Handles encoding detection for CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Extensions we know how to read.
SUPPORTED_EXTENSIONS = frozenset(
    [".csv", ".tsv", ".xlsx", ".xls", ".sav", ".dta", ".sas7bdat", ".parquet", ".json"]
)


def load(path: Path | str) -> "pd.DataFrame":
    """Load a dataset by file extension.

    Returns a pandas DataFrame. Raises ValueError for unknown extensions
    and surfaces the underlying library error for malformed files.
    """
    import pandas as pd

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    ext = p.suffix.lower()
    if ext == ".csv":
        return _load_csv(p)
    if ext == ".tsv":
        return _load_csv(p, sep="\t")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if ext == ".sav":
        import pyreadstat  # type: ignore[import-untyped]

        df, _meta = pyreadstat.read_sav(str(p))
        return df
    if ext == ".dta":
        return pd.read_stata(p, convert_categoricals=False)
    if ext == ".sas7bdat":
        import pyreadstat  # type: ignore[import-untyped]

        df, _meta = pyreadstat.read_sas7bdat(str(p))
        return df
    if ext == ".parquet":
        return pd.read_parquet(p)
    if ext == ".json":
        return pd.read_json(p)
    raise ValueError(
        f"Unsupported file extension: {ext!r}. "
        f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def _load_csv(path: Path, sep: str = ",") -> "pd.DataFrame":
    """CSV/TSV loader with encoding detection.

    Tries utf-8 first; on UnicodeDecodeError sniffs the encoding with
    chardet and retries. Keeps `dtype=object` initially so we don't lose
    string padding before the format-resolution stage runs.
    """
    import pandas as pd

    try:
        return pd.read_csv(path, sep=sep, dtype=object, keep_default_na=True)
    except UnicodeDecodeError:
        import chardet

        with open(path, "rb") as fh:
            sample = fh.read(64 * 1024)
        guess = chardet.detect(sample)
        encoding = guess.get("encoding") or "latin-1"
        return pd.read_csv(
            path, sep=sep, dtype=object, keep_default_na=True, encoding=encoding
        )


def save_clean(
    df: "pd.DataFrame",
    *,
    original_path: Path | str,
    out_dir: Path | str,
    base_name: str = "clean",
) -> dict[str, Path]:
    """Save cleaned data in the original file format and as a portable CSV.

    Returns a dict mapping format key -> output path.
    """
    original_path = Path(original_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = original_path.suffix.lower()
    written: dict[str, Path] = {}

    # Always write a CSV copy for portability.
    csv_path = out_dir / f"{base_name}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")
    written["csv"] = csv_path

    # Also write in the original format if we know how.
    if ext == ".csv":
        return written  # already handled
    if ext == ".tsv":
        tsv_path = out_dir / f"{base_name}.tsv"
        df.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8", lineterminator="\n")
        written["tsv"] = tsv_path
    elif ext in (".xlsx", ".xls"):
        xlsx_path = out_dir / f"{base_name}.xlsx"
        df.to_excel(xlsx_path, index=False)
        written["xlsx"] = xlsx_path
    elif ext == ".parquet":
        pq_path = out_dir / f"{base_name}.parquet"
        df.to_parquet(pq_path, index=False)
        written["parquet"] = pq_path
    elif ext == ".json":
        json_path = out_dir / f"{base_name}.json"
        df.to_json(json_path, orient="records", indent=2)
        written["json"] = json_path
    # .sav, .dta, .sas7bdat: skip round-trip; CSV is fine for downstream use.

    # Always write a Parquet copy for typed, compressed downstream use.
    # Silently skipped if pyarrow / fastparquet is not installed.
    if "parquet" not in written:
        try:
            pq_path = out_dir / f"{base_name}.parquet"
            df.to_parquet(pq_path, index=False)
            written["parquet"] = pq_path
        except Exception:
            pass

    return written
