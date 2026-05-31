"""Tests for the profiling stage."""

from __future__ import annotations

import pandas as pd
import pytest

from distill.profiler import profile, profile_summary_card
from distill.types import DecisionStatus, Stage, VariableType


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------


def test_profile_recognizes_native_numeric():
    df = pd.DataFrame({"x": [1.0, 2.5, 3.7, 4.1]})
    p = profile(df)["x"]
    assert p.detected_type == VariableType.NUMERIC
    assert p.numeric is not None
    assert p.numeric.min == pytest.approx(1.0)


def test_profile_recognizes_native_integer():
    df = pd.DataFrame({"age": [21, 35, 48, 62]})
    p = profile(df)["age"]
    assert p.detected_type == VariableType.INTEGER
    assert p.n_missing == 0


def test_profile_flags_numeric_stored_as_string():
    df = pd.DataFrame({"weight": ["72.4", "85.1", "61.0", "78.3", "90.0"]})
    p = profile(df)["weight"]
    assert p.detected_type == VariableType.NUMERIC
    assert any("numeric stored as string" in f for f in p.flags)


def test_profile_flags_date_stored_as_string():
    df = pd.DataFrame({"dob": ["1990-04-12", "1985-11-23", "2001-07-30", "1978-02-14"]})
    p = profile(df)["dob"]
    assert p.detected_type == VariableType.DATETIME
    assert any("datetime stored as string" in f for f in p.flags)


def test_profile_flags_boolean_stored_as_string():
    df = pd.DataFrame({"enrolled": ["yes", "no", "yes", "yes", "no", "no", "yes"]})
    p = profile(df)["enrolled"]
    assert p.detected_type == VariableType.BOOLEAN
    assert any("boolean stored as string" in f for f in p.flags)


def test_profile_low_cardinality_string_is_categorical():
    df = pd.DataFrame({"arm": ["A", "B", "A", "C", "B", "A", "B", "C"]})
    p = profile(df)["arm"]
    assert p.detected_type == VariableType.CATEGORICAL
    assert p.categorical is not None
    assert len(p.categorical.top_values) > 0


def test_profile_high_cardinality_string_is_free_text():
    df = pd.DataFrame({"note": [f"row-{i}-note" for i in range(80)]})
    p = profile(df)["note"]
    assert p.detected_type == VariableType.STRING


def test_profile_flags_zero_one_integer_as_possible_bool():
    df = pd.DataFrame({"flag": [0, 1, 0, 1, 1, 0, 1]})
    p = profile(df)["flag"]
    assert p.detected_type == VariableType.INTEGER
    assert any("0/1 encoded" in f for f in p.flags)


# ---------------------------------------------------------------------------
# Cross-cutting flags
# ---------------------------------------------------------------------------


def test_profile_flags_all_unique_as_identifier():
    df = pd.DataFrame({"sid": [f"S{i:04d}" for i in range(50)]})
    p = profile(df)["sid"]
    assert any("all-unique" in f for f in p.flags)


def test_profile_flags_near_constant():
    df = pd.DataFrame({"country": ["US"] * 100})
    p = profile(df)["country"]
    assert any("near-constant" in f for f in p.flags)


def test_profile_flags_nearly_empty():
    df = pd.DataFrame({"sparse": [None] * 99 + ["value"]})
    p = profile(df)["sparse"]
    assert any("nearly-empty" in f for f in p.flags)


# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------


def test_profile_counts_missing():
    df = pd.DataFrame({"age": [25.0, None, 30.0, None, 45.0]})
    p = profile(df)["age"]
    assert p.n_total == 5
    assert p.n_missing == 2
    assert p.missing_rate == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Summary card
# ---------------------------------------------------------------------------


def test_profile_summary_card_is_agent_auto():
    df = pd.DataFrame(
        {"a": [1, 2, 3, None], "b": ["yes", "no", "yes", "no"]}
    )
    profiles = profile(df)
    card = profile_summary_card(df, profiles)
    assert card.stage == Stage.PROFILE
    assert card.status == DecisionStatus.AGENT_AUTO
    assert card.metadata["n_rows"] == 4
    assert card.metadata["n_cols"] == 2
    assert "type_counts" in card.metadata
