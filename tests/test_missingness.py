"""Tests for the missingness stage.

Note: tests for `knn_impute` and `regression_impute` are excluded here
because they require scikit-learn at runtime. Their core code path is
exercised by integration tests once sklearn is installed locally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from distill.missingness import (
    apply_decision,
    apply_default_for_all,
    classify_pattern,
    detect_missingness,
    recommend_action,
)
from distill.profiler import profile
from distill.types import (
    DecisionCard,
    DecisionStatus,
    MissingnessPattern,
    NumericStats,
    Stage,
    VariableProfile,
    VariableType,
)


# ---------------------------------------------------------------------------
# Pattern classification
# ---------------------------------------------------------------------------


def test_pattern_mcar():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "age": rng.normal(50, 12, n),
            "weight": rng.normal(75, 15, n),
            "score": rng.normal(100, 10, n),
        }
    )
    df.loc[rng.choice(n, size=40, replace=False), "weight"] = np.nan
    pattern, _ev = classify_pattern(df, "weight")
    assert pattern == MissingnessPattern.MCAR


def test_pattern_mar_identifies_predictor():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({"age": rng.normal(50, 15, n), "bmi": rng.normal(26, 4, n)})
    prob = 1 / (1 + np.exp(-(df["age"] - 60) / 5))
    df.loc[rng.random(n) < prob, "bmi"] = np.nan
    pattern, ev = classify_pattern(df, "bmi")
    assert pattern == MissingnessPattern.MAR
    assert any(e["predictor"] == "age" for e in ev["signals"])


def test_pattern_no_missing_is_unknown():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    pattern, _ = classify_pattern(df, "a")
    assert pattern == MissingnessPattern.UNKNOWN


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def _num_profile(rate: float) -> VariableProfile:
    n_total = 1000
    n_missing = int(rate * n_total)
    return VariableProfile(
        name="x",
        detected_type=VariableType.NUMERIC,
        stored_dtype="float64",
        n_total=n_total,
        n_missing=n_missing,
        n_unique=400,
        numeric=NumericStats(mean=10, median=10, std=2, min=5, max=15, q25=8, q75=12),
    )


def test_recommend_drop_variable_when_mostly_missing():
    p = _num_profile(0.6)
    action, alts, _ = recommend_action(p, MissingnessPattern.MCAR)
    assert action == "drop_variable"
    assert "drop_variable" not in alts


def test_recommend_drop_rows_when_few_missing():
    p = _num_profile(0.01)
    action, _, _ = recommend_action(p, MissingnessPattern.MCAR)
    assert action == "drop_rows"


def test_recommend_median_for_mcar_numeric():
    p = _num_profile(0.08)
    action, _, _ = recommend_action(p, MissingnessPattern.MCAR)
    assert action == "median_impute"


def test_recommend_regression_for_mar_numeric():
    p = _num_profile(0.08)
    action, _, _ = recommend_action(p, MissingnessPattern.MAR)
    assert action == "regression_impute"


def test_recommend_mode_for_mcar_categorical():
    p = VariableProfile(
        name="arm",
        detected_type=VariableType.CATEGORICAL,
        stored_dtype="object",
        n_total=1000,
        n_missing=100,
        n_unique=3,
    )
    action, _, _ = recommend_action(p, MissingnessPattern.MCAR)
    assert action == "mode_impute"


# ---------------------------------------------------------------------------
# Card discovery
# ---------------------------------------------------------------------------


def test_detect_one_card_per_variable_with_missing():
    df = pd.DataFrame(
        {
            "age": [25, 30, None, 45, 52, None, 38, 41, 60, 55],
            "weight": [70, 75, 68, None, 82, 79, 73, None, 88, 85],
            "arm": ["A", "B", "A", "C", "B", None, "A", "B", "C", "A"],
            "dense": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    cards = detect_missingness(df, profile(df))
    assert len(cards) == 3
    assert {c.variable for c in cards} == {"age", "weight", "arm"}
    assert all(c.status == DecisionStatus.PENDING for c in cards)
    assert all(c.stage == Stage.MISSINGNESS for c in cards)


def test_detect_card_metadata_populated():
    df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0, 5.0]})
    cards = detect_missingness(df, profile(df))
    assert len(cards) == 1
    c = cards[0]
    assert c.metadata["missing_n"] == 1
    assert c.metadata["missing_rate"] == pytest.approx(0.2)
    assert c.metadata["pattern"] in ("MCAR", "MAR", "MNAR", "unknown")


# ---------------------------------------------------------------------------
# Apply decision
# ---------------------------------------------------------------------------


def _resolved_card(variable: str, action: str) -> DecisionCard:
    card = DecisionCard(
        card_id=f"c_{action}",
        stage=Stage.MISSINGNESS,
        variable=variable,
        issue="-",
        recommendation="-",
        rationale="-",
        default_action=action,
    )
    card.resolve(action=action, confirmed_by="user", status=DecisionStatus.CONFIRMED)
    return card


def test_apply_drop_variable():
    df = pd.DataFrame({"x": [1.0, None, 3.0], "y": [10, 20, 30]})
    out, info = apply_decision(df, _resolved_card("x", "drop_variable"))
    assert "x" not in out.columns
    assert info["cols_dropped"] == 1


def test_apply_drop_rows():
    df = pd.DataFrame({"x": [1.0, None, 3.0], "y": [10, 20, 30]})
    out, info = apply_decision(df, _resolved_card("x", "drop_rows"))
    assert len(out) == 2
    assert info["rows_dropped"] == 1


def test_apply_median_impute():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    out, info = apply_decision(df, _resolved_card("x", "median_impute"))
    assert not out["x"].isna().any()
    assert info["imputed_value"] == 2.0


def test_apply_mode_impute():
    df = pd.DataFrame({"arm": ["A", "B", "A", "A", None, "B", "A"]})
    out, info = apply_decision(df, _resolved_card("arm", "mode_impute"))
    assert not out["arm"].isna().any()
    assert info["imputed_value"] == "A"


def test_apply_indicator():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    out, info = apply_decision(df, _resolved_card("x", "flag_missing_indicator"))
    assert "x_was_missing" in out.columns
    assert out["x_was_missing"].tolist() == [False, True, False]


def test_apply_unknown_action_raises():
    with pytest.raises(ValueError, match="Unknown missingness action"):
        apply_decision(pd.DataFrame({"x": [1.0]}), _resolved_card("x", "bogus"))


# ---------------------------------------------------------------------------
# Non-interactive default-for-all
# ---------------------------------------------------------------------------


def test_apply_default_for_all_runs_full_pipeline():
    df = pd.DataFrame(
        {
            "id": list(range(20)),
            "age": [
                25, 30, None, 45, 52, None, 38, 41, 60, 55,
                40, 50, None, 33, 29, 48, 62, None, 44, 39,
            ],
            "score": [10.0] * 20,
            "sparse": [None] * 15 + [1, 2, 3, 4, 5],
        }
    )
    cards = detect_missingness(df, profile(df))
    out = apply_default_for_all(df, cards)
    assert all(c.status == DecisionStatus.AGENT_DEFAULT for c in cards)
    assert "sparse" not in out.columns
    assert not out["age"].isna().any()
