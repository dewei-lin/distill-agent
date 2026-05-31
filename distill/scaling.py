"""Stage 8 — numeric scaling suggestions.

Evaluates all numeric columns together and identifies those where scaling
would genuinely improve downstream analysis or modelling. Only generates
decision cards when at least one column meets the criteria:

  * range is dramatically larger than other numeric columns (ratio > 10×)
  * heavily skewed distribution (|skewness| > 2) where log or robust scaling
    would help

Columns that are already on a small absolute scale (max |value| < 10) are
assumed to be pre-scaled and are silently skipped.

If no column meets any criterion the stage exits with an empty list and
nothing is surfaced to the analyst.

User-requested scaling is available through `scale_column()` for on-demand
application from the Claude Code conversation.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pandas as pd

from .types import DecisionCard, Stage, VariableProfile, VariableType

# A column whose max |value| is below this is assumed to be already scaled.
_SCALED_MAX_ABS = 10.0
# Flag columns whose range is this many times larger than the smallest range.
_RANGE_RATIO_THRESHOLD = 10.0
# |skewness| above this → suggest robust or log scaling.
_SKEW_ROBUST_THRESHOLD = 2.0
# Don't flag near-constant columns.
_MIN_RANGE = 1e-9


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_scaling(
    df: pd.DataFrame,
    profiles: dict[str, VariableProfile],
) -> list[DecisionCard]:
    """Return one DecisionCard per numeric column where scaling is warranted.

    Examines all numeric columns together: if they are already on comparable
    scales, no cards are produced. Returns an empty list when no action is
    needed.
    """
    # First pass: gather stats for every unscaled numeric column.
    candidates: dict[str, tuple[float, float, float]] = {}  # col -> (range, max_abs, skew)
    for col, profile in profiles.items():
        if profile.detected_type not in (VariableType.NUMERIC, VariableType.INTEGER):
            continue
        if col not in df.columns:
            continue
        stats = profile.numeric
        if stats is None:
            continue
        col_range = (stats.max or 0.0) - (stats.min or 0.0)
        if col_range < _MIN_RANGE:
            continue
        max_abs = max(abs(stats.min or 0.0), abs(stats.max or 0.0))
        if max_abs < _SCALED_MAX_ABS:
            continue  # already in a tight range — assume pre-scaled
        series = df[col].dropna()
        skewness = float(series.skew()) if len(series) > 1 else 0.0
        candidates[col] = (col_range, max_abs, skewness)

    if not candidates:
        return []

    # Cross-column range ratio.
    ranges = {col: v[0] for col, v in candidates.items()}
    max_range = max(ranges.values())
    min_range = min(r for r in ranges.values() if r > 0)
    large_range_cols: set[str] = set()
    if max_range / min_range >= _RANGE_RATIO_THRESHOLD:
        large_range_cols = {
            col
            for col, r in ranges.items()
            if r > min_range * _RANGE_RATIO_THRESHOLD
        }

    cards: list[DecisionCard] = []
    for col, (col_range, max_abs, skewness) in candidates.items():
        in_large_range = col in large_range_cols
        heavily_skewed = abs(skewness) > _SKEW_ROBUST_THRESHOLD

        if not in_large_range and not heavily_skewed:
            continue

        method = _choose_method(df, col, skewness, in_large_range, heavily_skewed)
        reason = _build_reason(col_range, skewness, in_large_range, heavily_skewed)
        new_col = f"{col}_scaled"

        card = DecisionCard(
            card_id=f"scaling_{col}_{uuid.uuid4().hex[:6]}",
            stage=Stage.SCALING,
            variable=col,
            issue=reason,
            recommendation=_method_label(method),
            rationale=_rationale(method, reason),
            default_action=method,
            alternatives=[
                a
                for a in [
                    "scale_standard",
                    "scale_minmax",
                    "scale_robust",
                    "scale_log",
                    "leave_as_is",
                ]
                if a != method
            ],
            metadata={
                "method": method,
                "new_col": new_col,
                "range": col_range,
                "max_abs": max_abs,
                "skewness": skewness,
                "in_large_range_group": in_large_range,
            },
        )
        cards.append(card)

    return cards


def apply_decision(
    df: pd.DataFrame,
    card: DecisionCard,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply a resolved scaling decision. Returns (new_df, info_dict)."""
    col = card.variable
    action = card.action_taken or card.default_action

    if action == "leave_as_is" or col not in df.columns:
        return df, {"action": action, "col": col}

    new_col = str(card.metadata.get("new_col", f"{col}_scaled"))
    out = _apply_scale(df, col, action, new_col)
    return out, {"action": action, "col": col, "new_col": new_col}


def scale_column(
    df: pd.DataFrame,
    col: str,
    method: str,
    inplace: bool = False,
) -> pd.DataFrame:
    """Scale a column on demand, bypassing the automated assessment.

    Called when the user explicitly requests scaling in the chat
    (e.g. "standardise income" or "robust-scale age").

    Parameters
    ----------
    method  : 'standard' | 'minmax' | 'robust' | 'log'
    inplace : if True, overwrite the original column; otherwise create
              a new ``<col>_scaled`` column and leave the original intact.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")

    # Accept both "standard" and "scale_standard" spellings.
    full_method = f"scale_{method}" if not method.startswith("scale_") else method
    new_col = col if inplace else f"{col}_scaled"
    return _apply_scale(df, col, full_method, new_col)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _apply_scale(
    df: pd.DataFrame, col: str, method: str, new_col: str
) -> pd.DataFrame:
    out = df.copy()
    series = out[col].astype(float)

    if method == "scale_standard":
        mu = float(series.mean())
        sigma = float(series.std(ddof=1))
        out[new_col] = (series - mu) / sigma if sigma > 0 else series - mu

    elif method == "scale_minmax":
        lo, hi = float(series.min()), float(series.max())
        out[new_col] = (series - lo) / (hi - lo) if hi > lo else series * 0.0

    elif method == "scale_robust":
        med = float(series.median())
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        out[new_col] = (series - med) / iqr if iqr > 0 else series - med

    elif method == "scale_log":
        out[new_col] = np.log1p(series.clip(lower=0))

    else:
        raise ValueError(f"Unknown scaling method: {method!r}")

    return out


def _choose_method(
    df: pd.DataFrame,
    col: str,
    skewness: float,
    in_large_range: bool,
    heavily_skewed: bool,
) -> str:
    if heavily_skewed and skewness > 0:
        min_val = float(df[col].min())
        return "scale_log" if min_val >= 0 else "scale_robust"
    if heavily_skewed:
        return "scale_robust"
    if in_large_range:
        return "scale_robust" if abs(skewness) > 1.0 else "scale_standard"
    return "scale_standard"


def _build_reason(
    col_range: float,
    skewness: float,
    in_large_range: bool,
    heavily_skewed: bool,
) -> str:
    parts: list[str] = []
    if in_large_range:
        parts.append(
            f"range ({col_range:.3g}) is large relative to other numeric columns"
        )
    if heavily_skewed:
        direction = "right" if skewness > 0 else "left"
        parts.append(f"{direction}-skewed (skew={skewness:.2f})")
    return "; ".join(parts) if parts else "unusual distribution"


def _method_label(method: str) -> str:
    return {
        "scale_standard": "standardise (z-score)",
        "scale_minmax": "scale to [0, 1]",
        "scale_robust": "robust-scale (median / IQR)",
        "scale_log": "log-transform (log1p)",
    }.get(method, method)


def _rationale(method: str, reason: str) -> str:
    base = f"Column has {reason}."
    if method == "scale_log":
        return f"{base} Log-transform compresses the right tail while preserving order."
    if method == "scale_robust":
        return (
            f"{base} Robust scaling uses the median and IQR, which are "
            "unaffected by extreme values."
        )
    if method == "scale_standard":
        return f"{base} Standardising centres the column at 0 with unit variance."
    if method == "scale_minmax":
        return f"{base} Min-max scaling squeezes all values into [0, 1]."
    return base
