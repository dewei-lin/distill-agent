"""Stage 3 — Missingness handling.

For every variable with missing data, this module:

  1. Classifies the missingness pattern (MCAR / MAR / MNAR) using a
     lightweight heuristic that needs only pandas + numpy.
  2. Recommends an action (drop variable, drop rows, impute) based on the
     missing rate and the pattern.
  3. Emits one PENDING `DecisionCard` per variable for the analyst to
     confirm or override.

When the user confirms or overrides, `apply_decision` is called to
actually transform the data. The split between discovery and application
is what lets the same code drive both:

  * the Claude Code path, where the agent surfaces each card and waits;
  * the web app path, where each card maps to a UI element with
    confirm/override/inspect buttons;
  * the non-interactive Award-A path, where the orchestrator just calls
    `apply_default_for_all` and proceeds.

Pattern detection: lightweight by design. Proper MCAR tests (Little's) require
scipy and have a high implementation cost; for the use-case of "guide the
analyst's intuition", the quartile/level missingness-spread heuristic
below is sufficient and explainable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._pdcompat import is_text_dtype
from .types import (
    DecisionCard,
    DecisionStatus,
    MissingnessPattern,
    Stage,
    VariableProfile,
    VariableType,
)

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# A variable with >= this fraction missing is suggested for dropping outright.
DROP_VAR_THRESHOLD = 0.50

# A variable with <= this fraction missing is suggested for row-dropping.
DROP_ROWS_THRESHOLD = 0.02

# Spread (max - min) of missing rate across quartiles/levels of another
# variable that flags MAR.
MAR_SPREAD_THRESHOLD = 0.10

# Cap on the number of "other variables" to scan when classifying pattern;
# avoids quadratic blowup on wide datasets.
MAX_PATTERN_PROBES = 12

# When the variable is numeric, split candidate predictors into this many
# quantile bins for the spread test.
N_QUANTILE_BINS = 4

# Actions the user can override to.
ALL_NUMERIC_ACTIONS = [
    "drop_variable",
    "drop_rows",
    "mean_impute",
    "median_impute",
    "regression_impute",
    "flag_missing_indicator",
    "leave_as_is",
]
ALL_CATEGORICAL_ACTIONS = [
    "drop_variable",
    "drop_rows",
    "mode_impute",
    "flag_missing_indicator",
    "leave_as_is",
]
ALL_DATETIME_ACTIONS = [
    "drop_variable",
    "drop_rows",
    "flag_missing_indicator",
    "leave_as_is",
]


# ---------------------------------------------------------------------------
# Pattern classification
# ---------------------------------------------------------------------------


def classify_pattern(
    df: "pd.DataFrame",
    var: str,
    profiles: dict[str, VariableProfile] | None = None,
) -> tuple[MissingnessPattern, dict[str, Any]]:
    """Heuristic MCAR/MAR detection for `var` against the rest of `df`.

    Returns (pattern, evidence_dict). The evidence dict is attached to the
    DecisionCard so the analyst can see WHY the agent classified it the way
    it did.

    MCAR  — no candidate predictor shows a missing-rate spread above
            MAR_SPREAD_THRESHOLD.
    MAR   — at least one observed variable shows such a spread; we surface
            the strongest few in the evidence dict.
    MNAR  — never inferred from data alone; treated as MCAR or MAR for
            recommendation purposes. The report can note "MNAR possible"
            separately if the analyst flags it.
    """
    import numpy as np
    import pandas as pd

    s = df[var]
    is_missing = s.isna()
    n_missing = int(is_missing.sum())
    if n_missing == 0:
        return MissingnessPattern.UNKNOWN, {"note": "no missing values"}

    # Candidate predictors: all other columns that themselves aren't
    # mostly-missing (we can't probe missingness against a column we
    # ourselves can barely observe).
    candidates = [
        c for c in df.columns
        if c != var and df[c].isna().mean() < 0.5
    ]
    # Cap to avoid quadratic cost on wide data.
    candidates = candidates[:MAX_PATTERN_PROBES]

    evidence: list[dict[str, Any]] = []

    for c in candidates:
        other = df[c]
        if pd.api.types.is_numeric_dtype(other):
            # Quantile bins on observed values of `other`.
            observed_idx = other.notna()
            if observed_idx.sum() < 10:
                continue
            try:
                bins = pd.qcut(
                    other[observed_idx],
                    q=N_QUANTILE_BINS,
                    duplicates="drop",
                )
            except (ValueError, TypeError):
                continue
            if bins.nunique() < 2:
                continue
            grouped = is_missing[observed_idx].groupby(bins, observed=True)
            rates = grouped.mean()
            spread = float(rates.max() - rates.min())
        elif is_text_dtype(other):
            observed_idx = other.notna()
            if observed_idx.sum() < 10:
                continue
            grouped = is_missing[observed_idx].groupby(
                other[observed_idx].astype(str), observed=True
            )
            # Require at least 2 levels with >= 5 observations each.
            counts = grouped.count()
            if (counts >= 5).sum() < 2:
                continue
            rates = grouped.mean()
            spread = float(rates.max() - rates.min())
        elif pd.api.types.is_bool_dtype(other):
            observed_idx = other.notna()
            if observed_idx.sum() < 10:
                continue
            grouped = is_missing[observed_idx].groupby(other[observed_idx].astype(bool))
            rates = grouped.mean()
            spread = float(rates.max() - rates.min()) if len(rates) > 1 else 0.0
        else:
            continue

        if spread >= MAR_SPREAD_THRESHOLD:
            evidence.append({"predictor": c, "spread": round(spread, 3)})

    evidence.sort(key=lambda e: e["spread"], reverse=True)
    if evidence:
        return MissingnessPattern.MAR, {"signals": evidence[:3]}
    return MissingnessPattern.MCAR, {"signals": []}


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def recommend_action(
    profile: VariableProfile,
    pattern: MissingnessPattern,
) -> tuple[str, list[str], str]:
    """Pick the default action + alternatives + rationale for a missing
    variable, given its profile and detected pattern.

    Returns (default_action, alternatives, rationale).
    """
    rate = profile.missing_rate

    if rate >= DROP_VAR_THRESHOLD:
        rationale = (
            f"{rate:.0%} missing — too sparse to be useful as a predictor "
            f"and imputation would essentially synthesize the variable."
        )
        return "drop_variable", _alts_for(profile, exclude="drop_variable"), rationale

    if rate <= DROP_ROWS_THRESHOLD:
        rationale = (
            f"{rate:.1%} missing — small enough that dropping the affected "
            f"rows costs almost nothing and avoids any imputation bias."
        )
        return "drop_rows", _alts_for(profile, exclude="drop_rows"), rationale

    # Mid-range missingness: impute.
    if profile.detected_type in (VariableType.NUMERIC, VariableType.INTEGER):
        if pattern == MissingnessPattern.MAR:
            rationale = (
                f"{rate:.0%} missing and pattern is MAR — missingness "
                f"varies with other observed variables, so a model-based "
                f"imputation will reflect that relationship."
            )
            return "regression_impute", _alts_for(profile, exclude="regression_impute"), rationale
        rationale = (
            f"{rate:.0%} missing and pattern is MCAR — median imputation "
            f"preserves the marginal distribution and is robust to skew."
        )
        return "median_impute", _alts_for(profile, exclude="median_impute"), rationale

    if profile.detected_type == VariableType.CATEGORICAL:
        if pattern == MissingnessPattern.MAR:
            rationale = (
                f"{rate:.0%} missing and pattern is MAR — mode imputation "
                f"keeps the modal category share intact; consider the "
                f"missing-indicator alternative if the missingness itself "
                f"may be informative."
            )
            return "mode_impute", _alts_for(profile, exclude="mode_impute"), rationale
        rationale = (
            f"{rate:.0%} missing and pattern is MCAR — mode imputation "
            f"keeps the modal category share roughly intact."
        )
        return "mode_impute", _alts_for(profile, exclude="mode_impute"), rationale

    if profile.detected_type == VariableType.BOOLEAN:
        rationale = (
            f"{rate:.0%} missing — adding an explicit 'missing' indicator "
            f"is safer than guessing True/False, since both options change "
            f"downstream proportions materially."
        )
        return "flag_missing_indicator", ALL_NUMERIC_ACTIONS, rationale

    if profile.detected_type == VariableType.DATETIME:
        rationale = (
            f"{rate:.0%} missing — dates rarely impute meaningfully; either "
            f"drop the affected rows or add a missing-indicator."
        )
        return "drop_rows", ALL_DATETIME_ACTIONS, rationale

    # Free-text strings: just flag missingness.
    rationale = (
        f"{rate:.0%} missing in a free-text column — there is no sensible "
        f"impute; add a missing-indicator and leave values as-is."
    )
    return "flag_missing_indicator", ["drop_variable", "leave_as_is"], rationale


def _alts_for(profile: VariableProfile, exclude: str) -> list[str]:
    if profile.detected_type in (VariableType.NUMERIC, VariableType.INTEGER):
        base = ALL_NUMERIC_ACTIONS
    elif profile.detected_type == VariableType.CATEGORICAL:
        base = ALL_CATEGORICAL_ACTIONS
    elif profile.detected_type == VariableType.DATETIME:
        base = ALL_DATETIME_ACTIONS
    else:
        base = ["drop_variable", "drop_rows", "flag_missing_indicator", "leave_as_is"]
    return [a for a in base if a != exclude]


# ---------------------------------------------------------------------------
# Discovery: build PENDING decision cards
# ---------------------------------------------------------------------------


def detect_missingness(
    df: "pd.DataFrame",
    profiles: dict[str, VariableProfile],
) -> list[DecisionCard]:
    """One PENDING card per variable with at least one missing value."""
    cards: list[DecisionCard] = []
    for col in df.columns:
        p = profiles.get(col)
        if p is None or p.n_missing == 0:
            continue
        pattern, evidence = classify_pattern(df, col, profiles)
        default, alternatives, rationale = recommend_action(p, pattern)
        cards.append(
            DecisionCard(
                card_id=f"missing__{col}",
                stage=Stage.MISSINGNESS,
                variable=col,
                issue=f"{p.n_missing} of {p.n_total} rows missing ({p.missing_rate:.1%})",
                recommendation=_pretty_action(default, p),
                rationale=rationale,
                default_action=default,
                alternatives=alternatives,
                metadata={
                    "missing_n": p.n_missing,
                    "missing_rate": round(p.missing_rate, 4),
                    "pattern": pattern.value,
                    "evidence": evidence,
                    "detected_type": p.detected_type.value,
                },
            )
        )
    return cards


def _pretty_action(action: str, profile: VariableProfile) -> str:
    """Render the default action as a human-friendly recommendation string
    suitable for the card's `recommendation` field."""
    if action == "drop_variable":
        return f"drop variable '{profile.name}'"
    if action == "drop_rows":
        return f"drop rows missing '{profile.name}'"
    if action == "median_impute":
        med = profile.numeric.median if profile.numeric else None
        return f"impute with median{f' ({med:.3g})' if med is not None else ''}"
    if action == "mean_impute":
        mean = profile.numeric.mean if profile.numeric else None
        return f"impute with mean{f' ({mean:.3g})' if mean is not None else ''}"
    if action == "mode_impute":
        mode = (
            profile.categorical.top_values[0][0]
            if profile.categorical and profile.categorical.top_values
            else None
        )
        return f"impute with mode{f' ({mode!r})' if mode else ''}"
    if action == "regression_impute":
        return "impute with iterative regression (MICE)"
    if action == "flag_missing_indicator":
        return f"add '{profile.name}_was_missing' indicator and leave values as-is"
    if action == "leave_as_is":
        return "leave missing values in place"
    return action


# ---------------------------------------------------------------------------
# Application: execute a resolved decision
# ---------------------------------------------------------------------------


def apply_decision(
    df: "pd.DataFrame",
    card: DecisionCard,
    *,
    random_state: int = 42,
) -> tuple["pd.DataFrame", dict[str, Any]]:
    """Apply the action stored on a resolved DecisionCard.

    Returns (modified_df, info_dict). The info dict is suitable for merging
    into the card's metadata so the audit trail records exactly what
    happened (e.g. imputed median value, number of rows dropped).
    """
    import numpy as np
    import pandas as pd

    var = card.variable
    action = card.action_taken or card.default_action
    if var is None:
        raise ValueError(f"DecisionCard {card.card_id} has no variable")

    out = df.copy()
    info: dict[str, Any] = {"applied": action, "variable": var}

    if action == "drop_variable":
        out = out.drop(columns=[var])
        info["cols_dropped"] = 1
        return out, info

    if action == "drop_rows":
        before = len(out)
        out = out.dropna(subset=[var])
        info["rows_dropped"] = before - len(out)
        return out, info

    if action == "leave_as_is":
        return out, info

    if action == "flag_missing_indicator":
        indicator_col = f"{var}_was_missing"
        out[indicator_col] = df[var].isna()
        info["indicator_column"] = indicator_col
        return out, info

    n_missing = int(out[var].isna().sum())
    info["n_imputed"] = n_missing

    if action == "median_impute":
        val = out[var].median()
        out[var] = out[var].fillna(val)
        info["imputed_value"] = _to_jsonable(val)
        return out, info

    if action == "mean_impute":
        val = out[var].mean()
        out[var] = out[var].fillna(val)
        info["imputed_value"] = _to_jsonable(val)
        return out, info

    if action == "mode_impute":
        modes = out[var].mode(dropna=True)
        val = modes.iloc[0] if len(modes) else None
        out[var] = out[var].fillna(val)
        info["imputed_value"] = _to_jsonable(val)
        return out, info

    if action == "regression_impute":
        try:
            out = _regression_impute(out, var, random_state=random_state)
        except ImportError as e:
            # sklearn missing — fall back to a simpler imputer and record it.
            fb = _fallback_for(out, var)
            info["fallback_from"] = action
            info["fallback_reason"] = str(e)
            info["applied"] = fb
            out[var] = out[var].fillna(_fallback_value(out, var, fb))
            info["imputed_value"] = _to_jsonable(_fallback_value(out, var, fb))
        return out, info

    raise ValueError(f"Unknown missingness action: {action!r}")


def apply_default_for_all(
    df: "pd.DataFrame",
    cards: list[DecisionCard],
    *,
    random_state: int = 42,
) -> "pd.DataFrame":
    """Non-interactive convenience: resolve every pending card with its
    `default_action` and apply them sequentially.

    Cards are resolved in place (status=AGENT_DEFAULT, confirmed_by=
    "agent_default"). Returns the post-all-decisions DataFrame.
    """
    out = df
    for card in cards:
        if card.status != DecisionStatus.PENDING:
            continue
        card.resolve(
            action=card.default_action,
            confirmed_by="agent_default",
            status=DecisionStatus.AGENT_DEFAULT,
        )
        out, info = apply_decision(out, card, random_state=random_state)
        card.metadata.update({"applied_info": info})
    return out


# ---------------------------------------------------------------------------
# Imputation backends (lazy sklearn import)
# ---------------------------------------------------------------------------


def _regression_impute(df: "pd.DataFrame", var: str, *, random_state: int) -> "pd.DataFrame":
    """MICE (iterative regression) imputation for numeric `var`.

    MICE is numeric-only, so a non-numeric `var` falls back to mode
    imputation — the deterministic categorical default.
    """
    import pandas as pd

    if not pd.api.types.is_numeric_dtype(df[var]):
        out = df.copy()
        modes = out[var].mode(dropna=True)
        out[var] = out[var].fillna(modes.iloc[0] if len(modes) else None)
        return out

    try:
        # IterativeImputer is experimental; explicit opt-in import.
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "regression_impute requires scikit-learn; install with "
            "`pip install scikit-learn`"
        ) from e

    out = df.copy()
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    imputer = IterativeImputer(random_state=random_state, max_iter=10)
    out[numeric_cols] = imputer.fit_transform(out[numeric_cols])
    return out


def _fallback_for(df: "pd.DataFrame", var: str) -> str:
    """Pick a stdlib-only imputer when sklearn isn't available."""
    import pandas as pd

    if pd.api.types.is_numeric_dtype(df[var]):
        return "median_impute"
    return "mode_impute"


def _fallback_value(df: "pd.DataFrame", var: str, action: str) -> Any:
    import pandas as pd

    if action == "median_impute":
        return df[var].median()
    if action == "mean_impute":
        return df[var].mean()
    if action == "mode_impute":
        modes = df[var].mode(dropna=True)
        return modes.iloc[0] if len(modes) else None
    raise ValueError(f"no fallback value for action {action!r}")


def _to_jsonable(val: Any) -> Any:
    """Make pandas/numpy scalars JSON-serializable."""
    import numpy as np
    import pandas as pd

    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (pd.Timestamp,)):
        return val.isoformat()
    if isinstance(val, (np.bool_,)):
        return bool(val)
    try:
        return float(val) if hasattr(val, "__float__") else str(val)
    except Exception:
        return str(val)
