"""Stage 1 — Profiling.

Builds a `VariableProfile` for every column in a DataFrame without mutating
the data. This stage is purely diagnostic: it reports what the data *is*,
flags suspected type mismatches, and produces the input that subsequent
stages reason over.

Key design decisions:
  - `detected_type` is the agent's *semantic* guess (numeric vs integer vs
    boolean vs categorical vs string vs datetime), independent of how
    pandas happens to store the column.
  - Type detection is done by attempting safe coercion on a sample,
    NEVER on the live DataFrame. The format-resolution stage (stage 2)
    is the one that actually rewrites types.
  - All flags are advisory strings. They guide downstream stages and the
    audit report but never trigger silent changes.

The output is dict[column_name -> VariableProfile] plus a single
DecisionCard summarizing the pass (status = AGENT_AUTO).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .types import (
    CategoricalStats,
    DecisionCard,
    DecisionStatus,
    NumericStats,
    Stage,
    VariableProfile,
    VariableType,
)

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Tunables (constants kept module-level so adopters can monkey-patch easily)
# ---------------------------------------------------------------------------

# Above this fraction of unique values, an object column is treated as free
# text rather than categorical, regardless of absolute cardinality.
CATEGORICAL_MAX_UNIQUE_FRAC = 0.5

# Hard cap on unique values to still call something categorical, even if
# the fraction is low (e.g. 10k unique strings out of 100k rows shouldn't
# get one-hot-encoded by accident).
CATEGORICAL_MAX_UNIQUE_ABS = 50

# How many sample values to keep on the profile, and how many top-k for cats.
SAMPLE_VALUES_K = 8
TOP_CATEGORIES_K = 10

# Fraction of non-null sample values that must parse cleanly for us to flag
# a stored-as-string column as actually numeric / datetime / boolean.
COERCION_FLAG_THRESHOLD = 0.85

# Sample size used for coercion detection. Capped to avoid scanning large
# columns when we just need a heuristic.
COERCION_SAMPLE_N = 200

BOOLEAN_LIKE_VALUES = frozenset(
    {"true", "false", "t", "f", "yes", "no", "y", "n", "1", "0"}
)

# Date regex: anything with a 4-digit year and at least one separator.
# Intentionally permissive — the format stage does the real parsing.
_DATE_LIKE_RE = re.compile(
    r"^\s*("
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"        # 2024-03-15
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"     # 15/03/2024
    r"|\d{4}\d{2}\d{2}"                      # 20240315
    r")(\s.*)?$"
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def profile(df: "pd.DataFrame") -> dict[str, VariableProfile]:
    """Return a `VariableProfile` for every column of `df`.

    Does not mutate `df`. Safe to call repeatedly.
    """
    return {col: _profile_column(df, col) for col in df.columns}


def profile_summary_card(
    df: "pd.DataFrame",
    profiles: dict[str, VariableProfile],
) -> DecisionCard:
    """Build a single agent-auto card summarizing the profiling pass.

    Used so the decision log records that stage 1 ran and what it found,
    even though no human judgment was solicited.
    """
    n_total = df.shape[0]
    n_cols = df.shape[1]
    type_counts: dict[str, int] = {}
    flagged: list[str] = []
    for p in profiles.values():
        type_counts[p.detected_type.value] = type_counts.get(p.detected_type.value, 0) + 1
        for f in p.flags:
            flagged.append(f"{p.name}: {f}")

    n_with_missing = sum(1 for p in profiles.values() if p.n_missing > 0)

    card = DecisionCard(
        card_id="profile_summary",
        stage=Stage.PROFILE,
        issue=(
            f"Profiled {n_cols} variables across {n_total} rows. "
            f"{n_with_missing} have missing values; {len(flagged)} type-mismatch flags."
        ),
        recommendation="continue to format resolution",
        rationale="Profiling is diagnostic; no data changes were made.",
        default_action="continue",
        alternatives=["abort"],
        metadata={
            "n_rows": n_total,
            "n_cols": n_cols,
            "type_counts": type_counts,
            "n_with_missing": n_with_missing,
            "flags": flagged,
        },
        status=DecisionStatus.AGENT_AUTO,
        action_taken="continue",
        confirmed_by="agent_auto",
    )
    return card


# ---------------------------------------------------------------------------
# Per-column profiling
# ---------------------------------------------------------------------------


def _profile_column(df: "pd.DataFrame", col: str) -> VariableProfile:
    import pandas as pd

    s = df[col]
    n_total = len(s)
    n_missing = int(s.isna().sum())
    n_nonnull = n_total - n_missing
    n_unique = int(s.nunique(dropna=True))

    nonnull = s.dropna()
    sample_values = [str(v) for v in nonnull.head(SAMPLE_VALUES_K).tolist()]

    detected_type, flags = _detect_type(s, nonnull)

    numeric_stats: NumericStats | None = None
    cat_stats: CategoricalStats | None = None

    if detected_type in (VariableType.NUMERIC, VariableType.INTEGER):
        # Use the stored values if already numeric; otherwise coerce a copy
        # just for stats (does not mutate df).
        if pd.api.types.is_numeric_dtype(s):
            num = s.dropna()
        else:
            num = pd.to_numeric(nonnull, errors="coerce").dropna()
        if len(num) > 0:
            q25, q75 = num.quantile([0.25, 0.75])
            numeric_stats = NumericStats(
                mean=float(num.mean()),
                median=float(num.median()),
                std=float(num.std(ddof=1)) if len(num) > 1 else 0.0,
                min=float(num.min()),
                max=float(num.max()),
                q25=float(q25),
                q75=float(q75),
            )

    elif detected_type == VariableType.CATEGORICAL:
        vc = s.dropna().astype(str).value_counts().head(TOP_CATEGORIES_K)
        cat_stats = CategoricalStats(
            top_values=[(str(idx), int(count)) for idx, count in vc.items()]
        )

    # Cross-cutting flags
    if n_total > 0 and n_nonnull > 0 and n_unique == 1:
        flags.append("near-constant (1 unique value)")
    if n_total > 0 and n_unique == n_nonnull and n_nonnull > 1:
        # "all-unique" is purely factual. Only suggest it's a row
        # identifier when there's corroborating evidence (an id-like
        # name, or a sequential integer range) — a continuous numeric
        # measurement that happens to be all-distinct is NOT an ID.
        if _looks_like_identifier(str(col), s, detected_type):
            flags.append("all-unique (possible identifier)")
        else:
            flags.append("all-unique values")
    if n_total > 0 and (n_missing / n_total) >= 0.95:
        flags.append("nearly-empty (≥95% missing)")

    return VariableProfile(
        name=str(col),
        detected_type=detected_type,
        stored_dtype=str(s.dtype),
        n_total=n_total,
        n_missing=n_missing,
        n_unique=n_unique,
        sample_values=sample_values,
        numeric=numeric_stats,
        categorical=cat_stats,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------


def _detect_type(s: "pd.Series", nonnull: "pd.Series") -> tuple[VariableType, list[str]]:
    """Return (semantic type, list of advisory flags)."""
    import pandas as pd

    flags: list[str] = []

    # Empty: unknown
    if len(nonnull) == 0:
        return VariableType.UNKNOWN, ["entirely missing"]

    # Stored as datetime
    if pd.api.types.is_datetime64_any_dtype(s):
        return VariableType.DATETIME, flags

    # Stored as boolean
    if pd.api.types.is_bool_dtype(s):
        return VariableType.BOOLEAN, flags

    # Stored as numeric
    if pd.api.types.is_numeric_dtype(s):
        # Integer vs continuous numeric
        if _looks_integer(nonnull):
            # 2-value integer columns are usually encoded booleans
            uniq = set(nonnull.unique().tolist())
            if uniq.issubset({0, 1}):
                flags.append("0/1 encoded — may be boolean")
            return VariableType.INTEGER, flags
        return VariableType.NUMERIC, flags

    # Stored as object/string: try to detect a hidden semantic type
    sample = nonnull
    if len(sample) > COERCION_SAMPLE_N:
        sample = sample.sample(n=COERCION_SAMPLE_N, random_state=42)
    sample_str = sample.astype(str).str.strip()

    # Hidden numeric
    coerced_num = pd.to_numeric(sample_str, errors="coerce")
    num_ok = float(coerced_num.notna().mean())
    if num_ok >= COERCION_FLAG_THRESHOLD:
        flags.append(f"numeric stored as string ({num_ok:.0%} of sample parses)")
        if _looks_integer(coerced_num.dropna()):
            return VariableType.INTEGER, flags
        return VariableType.NUMERIC, flags

    # Hidden datetime
    date_like = sample_str.str.match(_DATE_LIKE_RE, na=False).mean()
    if date_like >= COERCION_FLAG_THRESHOLD:
        flags.append(f"datetime stored as string ({date_like:.0%} of sample matches)")
        return VariableType.DATETIME, flags

    # Hidden boolean (text yes/no/etc.)
    lower = sample_str.str.lower()
    bool_ok = float(lower.isin(BOOLEAN_LIKE_VALUES).mean())
    if bool_ok >= COERCION_FLAG_THRESHOLD:
        flags.append(f"boolean stored as string ({bool_ok:.0%} of sample is yes/no-like)")
        return VariableType.BOOLEAN, flags

    # Fall back to categorical vs free-text by cardinality
    n_nonnull = len(nonnull)
    n_unique = int(nonnull.nunique())
    unique_frac = n_unique / n_nonnull if n_nonnull else 0.0
    if n_unique <= CATEGORICAL_MAX_UNIQUE_ABS and unique_frac <= CATEGORICAL_MAX_UNIQUE_FRAC:
        return VariableType.CATEGORICAL, flags
    return VariableType.STRING, flags


# Name tokens that suggest a column is a row identifier.
_ID_NAME_TOKENS = frozenset(
    {"id", "key", "code", "uuid", "guid", "index", "idx", "identifier", "no", "num"}
)


def _looks_like_identifier(
    name: str,
    s: "pd.Series",
    detected_type: VariableType,
) -> bool:
    """Heuristic: is an all-unique column plausibly a ROW IDENTIFIER
    (rather than just a continuous measurement that happens to be
    all-distinct)?

    Evidence we accept:
      * the column name contains an id-like token, or ends in "id"/"key";
      * the column is a sequential integer run (0,1,2,... or 1,2,3,...);
      * the column is string-typed (free-text ids are common; a
        continuous *numeric* measurement being called an ID is the
        false-positive we are guarding against).
    """
    import numpy as np
    import pandas as pd

    lc = name.lower()
    parts = re.split(r"[^a-z0-9]+", lc)
    if any(t in _ID_NAME_TOKENS for t in parts):
        return True
    if lc.endswith("id") or lc.endswith("key"):
        return True

    # String / categorical all-unique columns: treat as possible IDs.
    if detected_type in (VariableType.STRING, VariableType.CATEGORICAL):
        return True

    # Integer columns: only if the values form a sequential run.
    if detected_type == VariableType.INTEGER and pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s.dropna(), errors="coerce").dropna().sort_values()
        if len(vals) > 1:
            diffs = np.diff(vals.to_numpy())
            if np.all(diffs == 1):
                return True

    return False


def _looks_integer(s: "pd.Series") -> bool:
    """True if every value in the series is mathematically an integer."""
    import numpy as np
    import pandas as pd

    if not pd.api.types.is_numeric_dtype(s):
        return False
    arr = s.to_numpy()
    if len(arr) == 0:
        return False
    # Treat NaN as not-integer-but-allowed; require finite values to be whole.
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return False
    return bool(np.all(np.equal(np.mod(finite, 1), 0)))
