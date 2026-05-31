"""Stage 7 — data binning suggestions.

Assesses numeric columns for potential discretisation. Only generates
decision cards when the distribution genuinely benefits from binning:

  * heavy skew (|skewness| > 1.5) — quantile bins create equal-sized groups
  * wide range relative to IQR (range > 20 × IQR) — natural-breaks binning
    clusters the actual data density

If no column meets the threshold the stage exits with an empty list and
nothing is surfaced to the analyst.

User-requested binning (bypassing the assessment) is available through
`bin_column()`, which can be called at any point from the Claude Code
conversation.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pandas as pd

from .types import DecisionCard, DecisionStatus, Stage, VariableProfile, VariableType

# Thresholds that decide whether a column warrants binning.
_SKEW_THRESHOLD = 1.5      # |skewness| above this → non-normal distribution
_RANGE_IQR_RATIO = 20.0    # (max − min) / IQR above this → heavy-tailed spread
_MIN_UNIQUE = 10            # skip columns that are already coarse-grained
_DEFAULT_N_BINS = 4


# ---------------------------------------------------------------------------
# Bin labels
# ---------------------------------------------------------------------------


def _fmt_thresholds(values: list[float]) -> list[str]:
    """Format threshold values with the fewest decimals that keep them distinct.

    Integer-valued thresholds render as "25"; fractional ones get just enough
    decimal places to stay unique (e.g. "3.1", "3.8"). Trailing ".0" is dropped.
    """
    if not values:
        return []
    for decimals in range(0, 7):
        formatted = [f"{v:.{decimals}f}" for v in values]
        if len(set(formatted)) == len(formatted):
            return formatted
    return [f"{v:.6g}" for v in values]


def make_bin_labels(cut_points: list[float]) -> list[str]:
    """Build human-readable, threshold-based labels from bin cut points.

    Given edges [c0, c1, ..., cn] (outer edges are slightly widened sentinels),
    the interior thresholds c1..c(n-1) drive readable labels:

        ["≤3.1", "3.1–3.8", "3.8–4.7", ">4.7"]

    These replace opaque "bin_1, bin_2, ..." labels so a reader understands
    each group at a glance. Labels are guaranteed unique (required by pd.cut).
    """
    n_bins = len(cut_points) - 1
    if n_bins <= 1:
        return ["all"]
    interior = _fmt_thresholds([float(x) for x in cut_points[1:-1]])
    if len(interior) != n_bins - 1:  # degenerate; fall back to positional
        return [f"bin_{i + 1}" for i in range(n_bins)]
    labels = [f"≤{interior[0]}"]
    for i in range(len(interior) - 1):
        labels.append(f"{interior[i]}–{interior[i + 1]}")
    labels.append(f">{interior[-1]}")
    # Guard against any accidental collision after formatting.
    if len(set(labels)) != len(labels):
        return [f"bin_{i + 1}" for i in range(n_bins)]
    return labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_binning(
    df: pd.DataFrame,
    profiles: dict[str, VariableProfile],
) -> list[DecisionCard]:
    """Return one DecisionCard per numeric column where binning is warranted.

    Returns an empty list when no column meets the criteria — the stage
    is silently skipped rather than surfacing an empty-handed prompt.
    """
    cards: list[DecisionCard] = []
    for col, profile in profiles.items():
        if profile.detected_type not in (VariableType.NUMERIC, VariableType.INTEGER):
            continue
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty or profile.n_unique < _MIN_UNIQUE:
            continue

        warranted, reason = _is_warranted(series, profile)
        if not warranted:
            continue

        method, cut_points = _suggest_method(series, profile)
        new_col = f"{col}_bin"

        card = DecisionCard(
            card_id=f"binning_{col}_{uuid.uuid4().hex[:6]}",
            stage=Stage.BINNING,
            variable=col,
            issue=f"{reason} — may benefit from discretisation",
            recommendation=(
                f"{method.replace('_', ' ')} into {len(cut_points) - 1} bins: "
                f"{_describe_bins(cut_points)}"
            ),
            rationale=_rationale(method, reason),
            default_action=method,
            alternatives=[
                a
                for a in [
                    "bin_equal_width",
                    "bin_quantile",
                    "bin_natural_breaks",
                    "leave_as_is",
                ]
                if a != method
            ],
            metadata={
                "method": method,
                "n_bins": len(cut_points) - 1,
                "cut_points": [float(x) for x in cut_points],
                "labels": make_bin_labels(cut_points),
                "new_col": new_col,
                "skewness": float(series.skew()),
                "range": [float(series.min()), float(series.max())],
                "reason": reason,
            },
        )
        cards.append(card)
    return cards


def apply_decision(
    df: pd.DataFrame,
    card: DecisionCard,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply a resolved binning decision. Returns (new_df, info_dict)."""
    col = card.variable
    action = card.action_taken or card.default_action

    if action == "leave_as_is" or col not in df.columns:
        return df, {"action": action, "col": col}

    meta = card.metadata
    # The action after an override IS the new method name.
    method = action
    n_bins = int(meta.get("n_bins", _DEFAULT_N_BINS))
    new_col = str(meta.get("new_col", f"{col}_bin"))

    series = df[col].dropna()
    cut_points = _compute_cuts(series, method, n_bins)
    labels = make_bin_labels(cut_points)
    out = df.copy()
    out[new_col] = pd.cut(
        out[col],
        bins=cut_points,
        labels=labels,
        include_lowest=True,
    ).astype("category")

    return out, {
        "action": action,
        "col": col,
        "new_col": new_col,
        "cut_points": [float(x) for x in cut_points],
        "labels": labels,
        "n_bins": len(cut_points) - 1,
    }


def bin_column(
    df: pd.DataFrame,
    col: str,
    method: str,
    bins: int | list[float] = _DEFAULT_N_BINS,
    new_col_name: str | None = None,
) -> pd.DataFrame:
    """Bin a column on demand, bypassing the automated assessment.

    Called when the user explicitly requests binning in the chat
    (e.g. "bin income into 5 quantile bins").

    Parameters
    ----------
    method : 'equal_width' | 'quantile' | 'natural_breaks'
    bins   : int (number of bins) or explicit list of cut-point edges
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")

    new_col = new_col_name or f"{col}_bin"
    series = df[col].dropna()

    if isinstance(bins, list):
        cut_points = [float(x) for x in bins]
    else:
        # Normalise method name so callers can omit the "bin_" prefix.
        action = f"bin_{method}" if not method.startswith("bin_") else method
        cut_points = _compute_cuts(series, action, int(bins))

    out = df.copy()
    out[new_col] = pd.cut(
        out[col],
        bins=cut_points,
        labels=make_bin_labels(cut_points),
        include_lowest=True,
    ).astype("category")
    return out


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_warranted(
    series: pd.Series,
    profile: VariableProfile,
) -> tuple[bool, str]:
    stats = profile.numeric
    if stats is None:
        return False, ""

    q25 = stats.q25 or 0.0
    q75 = stats.q75 or 0.0
    iqr = q75 - q25
    value_range = (stats.max or 0.0) - (stats.min or 0.0)

    if iqr > 0 and value_range > _RANGE_IQR_RATIO * iqr:
        return True, f"wide range ({value_range:.3g}) vs IQR ({iqr:.3g})"

    skewness = float(series.skew())
    if abs(skewness) > _SKEW_THRESHOLD:
        direction = "right" if skewness > 0 else "left"
        return True, f"{direction}-skewed (skew={skewness:.2f})"

    return False, ""


def _suggest_method(
    series: pd.Series,
    profile: VariableProfile,
) -> tuple[str, list[float]]:
    stats = profile.numeric
    q25 = (stats.q25 or 0.0) if stats else 0.0
    q75 = (stats.q75 or 0.0) if stats else 0.0
    iqr = q75 - q25
    value_range = ((stats.max or 0.0) - (stats.min or 0.0)) if stats else 0.0

    skewness = float(series.skew())

    if abs(skewness) > _SKEW_THRESHOLD:
        method = "bin_quantile"
    elif iqr > 0 and value_range > _RANGE_IQR_RATIO * iqr:
        method = "bin_natural_breaks"
    else:
        method = "bin_equal_width"

    cut_points = _compute_cuts(series, method, _DEFAULT_N_BINS)
    return method, cut_points


def _compute_cuts(series: pd.Series, method: str, n_bins: int) -> list[float]:
    clean = series.dropna().sort_values()
    if len(clean) == 0:
        return []

    if method == "bin_quantile":
        quantiles = np.linspace(0, 100, n_bins + 1)
        cuts = np.percentile(clean, quantiles).tolist()
    elif method == "bin_natural_breaks":
        cuts = _natural_breaks(clean.values, n_bins)
    else:  # equal_width or unknown
        lo, hi = float(clean.min()), float(clean.max())
        cuts = np.linspace(lo, hi, n_bins + 1).tolist()

    # De-duplicate and enforce strict monotonicity.
    unique = sorted({float(x) for x in cuts})
    if len(unique) < 2:
        lo, hi = float(clean.min()), float(clean.max())
        unique = np.linspace(lo, hi, n_bins + 1).tolist()

    # Widen outer edges slightly so pd.cut captures exact min/max.
    unique[0] -= 1e-9
    unique[-1] += 1e-9
    return [float(x) for x in unique]


def _natural_breaks(values: np.ndarray, n_classes: int) -> list[float]:
    """Approximate natural-breaks classification via k-means clustering."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        # Fallback to equal-width if sklearn is somehow unavailable.
        lo, hi = float(values.min()), float(values.max())
        return np.linspace(lo, hi, n_classes + 1).tolist()

    arr = np.sort(values.ravel())
    n = len(arr)
    n_classes = min(n_classes, n)

    # Cap sample to keep KMeans fast.
    if n > 5000:
        idx = np.round(np.linspace(0, n - 1, 5000)).astype(int)
        sample = arr[idx]
    else:
        sample = arr

    km = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    km.fit(sample.reshape(-1, 1))
    centers = np.sort(km.cluster_centers_.ravel())

    # Midpoints between consecutive cluster centres become bin edges.
    breaks: list[float] = [float(arr[0])]
    for i in range(len(centers) - 1):
        breaks.append(float((centers[i] + centers[i + 1]) / 2))
    breaks.append(float(arr[-1]))
    return breaks


def _describe_bins(cut_points: list[float]) -> str:
    """Short human-readable preview of the first few bin ranges."""
    if len(cut_points) < 2:
        return ""
    parts = [
        f"[{cut_points[i]:.3g}, {cut_points[i + 1]:.3g})"
        for i in range(len(cut_points) - 1)
    ]
    preview = ", ".join(parts[:3])
    return preview + ("…" if len(parts) > 3 else "")


def _rationale(method: str, reason: str) -> str:
    if method == "bin_quantile":
        return (
            f"Column is {reason}. Quantile bins create equal-sized groups, "
            "which are robust to skew and extreme values."
        )
    if method == "bin_natural_breaks":
        return (
            f"Column has {reason}. Natural-breaks binning finds boundaries "
            "that minimise within-group variance."
        )
    return (
        f"Column has {reason}. Equal-width bins give the most interpretable "
        "range-based groups."
    )
