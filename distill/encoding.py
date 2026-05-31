"""Feature engineering — encoding transforms.

Covers two operations:

  * **Frequency encoding** — replaces each categorical level with its
    relative frequency in the dataset. Suggested by LLM when a categorical
    column has too many levels for one-hot encoding to be practical
    (n_unique > 20, < 50 % of rows unique so it's not an ID column).

  * **Datetime decomposition** — extracts named components (year, month,
    day, dayofweek, hour, is_weekend) from a datetime column.
    Never auto-suggested — always user-initiated from the selector.

Both operations preserve the original column.
"""

from __future__ import annotations

import uuid
from typing import Any

import pandas as pd

from .types import DecisionCard, Stage, VariableProfile, VariableType

# Minimum number of unique values before frequency encoding is suggested.
_MIN_UNIQUE_FOR_FE = 20
# Skip columns where more than half the rows are unique (likely IDs).
_MAX_UNIQUE_FRAC = 0.5

_DATETIME_COMPONENTS = ("year", "month", "day", "dayofweek", "hour", "is_weekend")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_encoding(
    df: pd.DataFrame,
    profiles: dict[str, VariableProfile],
) -> list[DecisionCard]:
    """Return one DecisionCard per high-cardinality categorical column.

    Only flags columns where:
    - detected type is CATEGORICAL or STRING
    - n_unique > 20  (too many levels for one-hot encoding)
    - n_unique / n_total < 0.5  (not an ID-like column)

    Returns an empty list when nothing is flagged.
    """
    cards: list[DecisionCard] = []
    for col, profile in profiles.items():
        if profile.detected_type not in (VariableType.CATEGORICAL, VariableType.STRING):
            continue
        if col not in df.columns:
            continue
        if profile.n_unique <= _MIN_UNIQUE_FOR_FE:
            continue
        if profile.n_total > 0 and profile.n_unique / profile.n_total >= _MAX_UNIQUE_FRAC:
            continue  # near-unique → likely an ID column

        new_col = f"{col}_freq"
        card = DecisionCard(
            card_id=f"encoding_{col}_{uuid.uuid4().hex[:6]}",
            stage=Stage.ENCODING,
            variable=col,
            issue=(
                f"{profile.n_unique:,} unique values — too many for one-hot encoding"
            ),
            recommendation=f"frequency encode '{col}' → '{new_col}'",
            rationale=(
                f"With {profile.n_unique:,} categories, one-hot encoding would add "
                f"{profile.n_unique:,} columns. Frequency encoding replaces each "
                "value with its relative frequency, capturing cardinality information "
                "in a single numeric column."
            ),
            default_action="frequency_encode",
            alternatives=["leave_as_is"],
            metadata={"n_unique": profile.n_unique, "new_col": new_col},
        )
        cards.append(card)
    return cards


def apply_decision(
    df: pd.DataFrame,
    card: DecisionCard,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply a resolved encoding decision. Returns (new_df, info_dict)."""
    col = card.variable
    action = card.action_taken or card.default_action

    if action == "leave_as_is" or col not in df.columns:
        return df, {"action": action, "col": col}

    if action == "frequency_encode":
        new_col = str(card.metadata.get("new_col", f"{col}_freq"))
        out = frequency_encode(df, col, new_col)
        return out, {"action": action, "col": col, "new_col": new_col}

    return df, {"action": action, "col": col}


def frequency_encode(
    df: pd.DataFrame,
    col: str,
    new_col_name: str | None = None,
) -> pd.DataFrame:
    """Replace each category with its relative frequency in the dataset.

    The original column is preserved. Unseen values (after train/test split)
    map to 0.0.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")
    new_col = new_col_name or f"{col}_freq"
    freq_map = df[col].value_counts(normalize=True)
    out = df.copy()
    out[new_col] = df[col].map(freq_map).fillna(0.0)
    return out


def decompose_datetime(
    df: pd.DataFrame,
    col: str,
    components: list[str] | None = None,
) -> pd.DataFrame:
    """Extract datetime components from a column into new numeric columns.

    Always user-initiated — never auto-suggested.

    Parameters
    ----------
    components : subset of ("year","month","day","dayofweek","hour","is_weekend").
                 Defaults to all six when None.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")

    out = df.copy()
    series = pd.to_datetime(out[col], errors="coerce")

    wanted = list(components) if components else list(_DATETIME_COMPONENTS)
    accessor_map: dict[str, Any] = {
        "year":      series.dt.year,
        "month":     series.dt.month,
        "day":       series.dt.day,
        "dayofweek": series.dt.dayofweek,
        "hour":      series.dt.hour,
        "is_weekend": (series.dt.dayofweek >= 5).astype("Int8"),
    }

    for comp in wanted:
        if comp in accessor_map:
            out[f"{col}_{comp}"] = accessor_map[comp]

    return out
