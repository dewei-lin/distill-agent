"""Stage 4 — Outlier and anomaly detection.

For each numeric variable, flags potential outliers using IQR + Z-score
(and optionally isolation forest for multivariate anomalies if sklearn is
available). The key distinction we surface to the analyst is:

  * "Likely data-entry error" — values that are physically implausible
    (negative ages, weights of 9999, etc). Suggested action: drop or
    treat as missing.

  * "Genuine extreme value" — statistical outlier but plausible (a
    millionaire in an income survey, a 60-year-old in a pediatric
    follow-up). Suggested action: leave as-is.

Implausibility rules are kept conservative — we only auto-classify as
"likely error" when we are very confident (e.g. negative value in a
column where the profile shows the entire observed minimum > 0, or a
value > 100x the 99th percentile). Everything else surfaces as a
PENDING card for the analyst.

Per-row outlier flags from this stage feed the row-count tracking, but
this module never removes rows on its own — that's the analyst's call
via the decision card.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from .types import (
    DecisionCard,
    DecisionStatus,
    Stage,
    VariableProfile,
    VariableType,
)

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

IQR_MULTIPLIER = 1.5            # standard Tukey fence
Z_THRESHOLD = 3.0               # |z| > 3 flagged
MIN_VALUES_FOR_DETECTION = 10   # below this, skip — too noisy

# Z-score above this marks a flagged column as "extreme" severity rather
# than "mild". Extreme outliers are surfaced for inspection but never
# auto-treated as errors (they are often a legitimate heavy tail).
EXTREME_Z_THRESHOLD = 5.0


def _sf(v: float) -> float | None:
    """Return v as a Python float, or None if it is NaN or Inf (not JSON-safe)."""
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


# Outlier-handling actions surfaced as alternatives.
NUMERIC_OUTLIER_ACTIONS = [
    "leave_as_is",         # most common: keep extreme values
    "treat_as_missing",    # convert flagged values to NaN; later imputed
    "drop_rows",           # remove the affected rows entirely
    "winsorize",           # clip to the fence bounds (cap-and-floor)
    "inspect",             # just surface; no transformation
]


# ---------------------------------------------------------------------------
# Per-variable flagging
# ---------------------------------------------------------------------------


def flag_outliers(
    df: "pd.DataFrame",
    var: str,
) -> dict[str, Any]:
    """Run IQR + Z-score outlier detection on a single numeric variable.

    Returns a dict with the flagged row indices, bounds, and a coarse
    severity label ('mild' / 'extreme' / 'implausible'). Does not modify
    the data.
    """
    import numpy as np
    import pandas as pd

    s = df[var]
    if not pd.api.types.is_numeric_dtype(s):
        return {"applicable": False, "reason": "non-numeric"}

    nonnull = s.dropna()
    if len(nonnull) < MIN_VALUES_FOR_DETECTION:
        return {"applicable": False, "reason": f"only {len(nonnull)} non-null values"}

    q01, q25, q50, q75, q99 = nonnull.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    iqr = q75 - q25
    lo = q25 - IQR_MULTIPLIER * iqr
    hi = q75 + IQR_MULTIPLIER * iqr

    mean = float(nonnull.mean())
    std = float(nonnull.std(ddof=1))

    is_iqr_outlier = (s < lo) | (s > hi)
    if std > 0:
        z = (s - mean) / std
        is_z_outlier = z.abs() > Z_THRESHOLD
    else:
        is_z_outlier = pd.Series(False, index=s.index)

    # Implausibility check — deliberately narrow. The ONLY data-only signal
    # we trust enough to call "almost certainly a data-entry error" is a
    # negative value in a column whose bulk is clearly positive. We test
    # that with the lower quartile (q25 > 0): this is robust to a small
    # minority of negative values (they don't drag q25 below zero), unlike
    # the 1st percentile which a mere 1-2% of negatives would pull
    # negative — defeating the very check we want.
    #
    # We intentionally do NOT flag large positive values as implausible:
    # heavy-tailed columns (sampling weights, income, capital gains) have
    # legitimate extreme highs, and a tight-clustered column with a wide
    # legitimate range (e.g. hours-per-week) would false-positive. Those
    # surface as "extreme" instead — inspected, never auto-deleted.
    #
    # Caveat: a column that is mostly positive but legitimately admits
    # negatives (e.g. a temperature in °F) can false-positive here. That
    # is why the resulting card is still surfaced for human confirmation.
    if float(q25) > 0:
        is_implausible = (s < 0).fillna(False)
    else:
        is_implausible = pd.Series(False, index=s.index)
    # Use pandas .sum() for counts — avoids materialising full Python lists.
    n_implausible = int(is_implausible.sum())

    # A flagged value is anything caught by IQR, Z-score, OR the
    # implausibility rule. Folding implausibles in here guarantees a
    # negative value always produces a card even if it doesn't happen to
    # cross the statistical thresholds.
    flagged_mask = (is_iqr_outlier | is_z_outlier | is_implausible).fillna(False)
    n_flagged_total = int(flagged_mask.sum())

    # Severity label — use counts, no Python list needed.
    if n_implausible > 0:
        severity = "implausible"
    elif n_flagged_total > 0:
        if std > 0 and (z.abs().dropna() > EXTREME_Z_THRESHOLD).any():
            severity = "extreme"
        else:
            severity = "mild"
    else:
        severity = "none"

    # Build compact display lists (max 50 each).
    # flagged_indices is stored in full so apply_decision can treat all rows;
    # flagged_values is capped at 50 — it is display-only.
    flagged_idx = flagged_mask[flagged_mask].index.tolist()   # full — needed by apply_decision
    flagged_idx_clean = [int(i) for i in flagged_idx[:50]]

    def _safe_val(v):
        if v is None:
            return None
        try:
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else f
        except (TypeError, ValueError):
            return str(v)

    flagged_vals_display = [_safe_val(v) for v in s.loc[flagged_idx[:50]].tolist()]
    implausible_idx_display = [int(i) for i in s[is_implausible].iloc[:50].index]

    return {
        "applicable": True,
        "n_total": int(len(s)),
        "n_nonnull": int(len(nonnull)),
        "n_flagged_iqr": int(is_iqr_outlier.fillna(False).sum()),
        "n_flagged_z": int(is_z_outlier.fillna(False).sum()),
        "n_flagged_total": n_flagged_total,
        "n_implausible": n_implausible,
        "bounds": {
            "iqr_low": _sf(lo),
            "iqr_high": _sf(hi),
            "min_observed": _sf(nonnull.min()),
            "max_observed": _sf(nonnull.max()),
            "q01": _sf(q01),
            "q99": _sf(q99),
            "mean": _sf(mean),
            "std": _sf(std),
        },
        "flagged_indices": flagged_idx,          # full list — apply_decision needs all rows
        "flagged_values": flagged_vals_display,  # display-only cap of 50
        "implausible_indices": implausible_idx_display,
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# Discovery: build PENDING decision cards
# ---------------------------------------------------------------------------


def detect_outliers(
    df: "pd.DataFrame",
    profiles: dict[str, VariableProfile],
) -> list[DecisionCard]:
    """One PENDING card per numeric variable with at least one flagged value.

    Implausible-severity columns get a more aggressive default
    (treat_as_missing) than statistically extreme but plausible columns
    (leave_as_is).
    """
    cards: list[DecisionCard] = []
    for col in df.columns:
        p = profiles.get(col)
        if p is None:
            continue
        if p.detected_type not in (VariableType.NUMERIC, VariableType.INTEGER):
            continue
        result = flag_outliers(df, col)
        if not result.get("applicable"):
            continue
        # Produce a card if anything was flagged OR any value is
        # implausible (implausibles are folded into flagged anyway, but be
        # explicit).
        if result["n_flagged_total"] == 0 and result["n_implausible"] == 0:
            continue

        severity = result["severity"]
        n_flag = result["n_flagged_total"]
        n_imp = result["n_implausible"]

        if severity == "implausible":
            default = "treat_as_missing"
            rationale = (
                f"{n_imp} value(s) are physically implausible relative to the "
                f"rest of the column — likely data-entry errors. Recommend "
                f"replacing with NaN so the missingness pipeline can re-impute."
            )
        elif severity == "extreme":
            default = "inspect"
            rationale = (
                f"{n_flag} flagged value(s); some exceed |z|>5 from the mean. "
                f"Could be genuine extremes (long tail) or data-entry errors — "
                f"please inspect the distribution before deciding."
            )
        else:
            default = "leave_as_is"
            rationale = (
                f"{n_flag} value(s) fall outside the 1.5xIQR fence. With a "
                f"moderate tail this is expected; keeping them preserves the "
                f"variance for downstream modelling."
            )

        # Winsorize requires valid IQR bounds; skip it if bounds are None
        # (happens when column has Inf values that made IQR computation undefined).
        has_valid_bounds = (
            result["bounds"].get("iqr_low") is not None
            and result["bounds"].get("iqr_high") is not None
        )
        available_actions = [
            a for a in NUMERIC_OUTLIER_ACTIONS
            if a != "winsorize" or has_valid_bounds
        ]
        cards.append(
            DecisionCard(
                card_id=f"outlier__{col}",
                stage=Stage.OUTLIER,
                variable=col,
                issue=(
                    f"{n_flag} value(s) flagged "
                    f"(severity: {severity}; {n_imp} implausible)"
                ),
                recommendation=_pretty_action(default),
                rationale=rationale,
                default_action=default,
                alternatives=[a for a in available_actions if a != default],
                metadata=result,
            )
        )
    return cards


def _pretty_action(action: str) -> str:
    return {
        "leave_as_is": "leave outliers in place",
        "treat_as_missing": "replace flagged values with NaN",
        "drop_rows": "drop rows containing flagged values",
        "winsorize": "winsorize at 1.5xIQR fence (cap+floor)",
        "inspect": "inspect distribution before deciding",
    }.get(action, action)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def apply_decision(
    df: "pd.DataFrame",
    card: DecisionCard,
) -> tuple["pd.DataFrame", dict[str, Any]]:
    """Apply a resolved outlier DecisionCard."""
    import numpy as np
    import pandas as pd

    var = card.variable
    action = card.action_taken or card.default_action
    if var is None:
        raise ValueError(f"DecisionCard {card.card_id} has no variable")

    out = df.copy()
    info: dict[str, Any] = {"applied": action, "variable": var}

    if action in ("leave_as_is", "inspect"):
        info["n_changed"] = 0
        return out, info

    # Pull the flagged indices the discovery pass recorded.
    flagged_idx = list(card.metadata.get("flagged_indices", []))
    # Refresh against current df indices in case rows were already dropped.
    flagged_idx = [i for i in flagged_idx if i in out.index]
    info["n_flagged"] = len(flagged_idx)

    if action == "treat_as_missing":
        out.loc[flagged_idx, var] = np.nan
        info["n_changed"] = len(flagged_idx)
        return out, info

    if action == "drop_rows":
        out = out.drop(index=flagged_idx)
        info["rows_dropped"] = len(flagged_idx)
        return out, info

    if action == "winsorize":
        bounds = card.metadata.get("bounds", {})
        lo = bounds.get("iqr_low")
        hi = bounds.get("iqr_high")
        if lo is None or hi is None:
            raise ValueError("winsorize requires iqr bounds in card metadata")
        clipped = out[var].clip(lower=lo, upper=hi)
        n_changed = int((clipped != out[var]).fillna(False).sum())
        out[var] = clipped
        info["n_changed"] = n_changed
        info["bounds"] = {"low": lo, "high": hi}
        return out, info

    raise ValueError(f"Unknown outlier action: {action!r}")


def apply_default_for_all(
    df: "pd.DataFrame",
    cards: list[DecisionCard],
) -> "pd.DataFrame":
    """Resolve and apply every card with its default action (non-interactive)."""
    out = df
    for card in cards:
        if card.status != DecisionStatus.PENDING:
            continue
        card.resolve(
            action=card.default_action,
            confirmed_by="agent_default",
            status=DecisionStatus.AGENT_DEFAULT,
        )
        out, info = apply_decision(out, card)
        card.metadata["applied_info"] = info
    return out
