"""Tests for the format resolution stage."""

from __future__ import annotations

import pandas as pd

from distill.format import resolve_formats, to_snake_case
from distill.profiler import profile
from distill.types import DecisionStatus, Stage


# ---------------------------------------------------------------------------
# Snake case
# ---------------------------------------------------------------------------


def test_snake_case_basic():
    assert to_snake_case("Subject ID") == "subject_id"
    assert to_snake_case("BMI (kg/m^2)") == "bmi_kg_m_2"
    assert to_snake_case("camelCaseField") == "camel_case_field"
    assert to_snake_case("PascalCase") == "pascal_case"
    assert to_snake_case("  spaced out  ") == "spaced_out"
    assert to_snake_case("ALL_CAPS") == "all_caps"
    assert to_snake_case("9-Lives") == "_9_lives"  # leading digit prefix
    assert to_snake_case("---") == "_"


# ---------------------------------------------------------------------------
# Column rename
# ---------------------------------------------------------------------------


def test_resolve_renames_columns():
    df = pd.DataFrame({"Subject ID": [1, 2], "BMI (kg/m^2)": [22.1, 27.4]})
    out, cards = resolve_formats(df, profile(df))
    assert list(out.columns) == ["subject_id", "bmi_kg_m_2"]
    rename_cards = [c for c in cards if c.card_id == "format_column_names"]
    assert len(rename_cards) == 1
    assert rename_cards[0].status == DecisionStatus.AGENT_AUTO


def test_resolve_no_rename_when_already_snake():
    df = pd.DataFrame({"subject_id": [1, 2], "bmi": [22.1, 27.4]})
    out, cards = resolve_formats(df, profile(df))
    assert list(out.columns) == ["subject_id", "bmi"]
    assert not any(c.card_id == "format_column_names" for c in cards)


# ---------------------------------------------------------------------------
# Whitespace stripping
# ---------------------------------------------------------------------------


def test_resolve_strips_whitespace():
    df = pd.DataFrame({"arm": [" A", "B ", "  A  ", "B"]})
    out, cards = resolve_formats(df, profile(df))
    assert out["arm"].tolist() == ["A", "B", "A", "B"]
    assert any(c.card_id == "format_strip_ws__arm" for c in cards)


# ---------------------------------------------------------------------------
# Numeric coercion
# ---------------------------------------------------------------------------


def test_resolve_coerces_numeric_stored_as_string():
    df = pd.DataFrame({"weight": ["72.4", "85.1", "61.0", "78.3", "90.0"]})
    out, cards = resolve_formats(df, profile(df))
    assert pd.api.types.is_numeric_dtype(out["weight"])
    assert out["weight"].iloc[0] == 72.4
    assert any(c.card_id == "format_coerce_numeric__weight" for c in cards)


def test_resolve_numeric_records_unparseable_count():
    df = pd.DataFrame({"v": ["1", "2", "3", "4", "5", "6", "7", "oops", "9", "10"]})
    out, cards = resolve_formats(df, profile(df))
    coerce = [c for c in cards if c.card_id == "format_coerce_numeric__v"]
    assert len(coerce) == 1
    assert coerce[0].metadata["n_unparseable_to_nan"] == 1


# ---------------------------------------------------------------------------
# Datetime coercion
# ---------------------------------------------------------------------------


def test_resolve_coerces_datetime():
    df = pd.DataFrame({"dob": ["1990-04-12", "1985-11-23", "2001-07-30", "1978-02-14"]})
    out, cards = resolve_formats(df, profile(df))
    assert pd.api.types.is_datetime64_any_dtype(out["dob"])
    assert any(c.card_id == "format_coerce_datetime__dob" for c in cards)


# ---------------------------------------------------------------------------
# Boolean coercion
# ---------------------------------------------------------------------------


def test_resolve_coerces_yes_no_to_boolean():
    df = pd.DataFrame({"active": ["yes", "no", "yes", "no", "yes", "no", "yes"]})
    out, cards = resolve_formats(df, profile(df))
    assert out["active"].dtype.name == "boolean"
    assert out["active"].iloc[0] is True or out["active"].iloc[0]
    assert any(c.card_id == "format_coerce_boolean__active" for c in cards)


def test_resolve_coerces_0_1_to_boolean():
    df = pd.DataFrame({"flag": [0, 1, 0, 1, 1, 0, 1, 0]})
    out, cards = resolve_formats(df, profile(df))
    assert out["flag"].dtype.name == "boolean"
    assert any(c.card_id == "format_coerce_boolean__flag" for c in cards)


# ---------------------------------------------------------------------------
# Categorical label normalization
# ---------------------------------------------------------------------------


def test_resolve_normalizes_categorical_case_variants():
    df = pd.DataFrame({"arm": ["A", "a", "A", "B", "b", "A", "A"]})
    out, cards = resolve_formats(df, profile(df))
    # "A" appears 4x as "A" and 1x as "a"; "A" wins.
    assert set(out["arm"].dropna().unique()) <= {"A", "B"}
    norm = [c for c in cards if c.card_id == "format_normalize_labels__arm"]
    assert len(norm) == 1


def test_resolve_does_not_normalize_genuinely_distinct_labels():
    df = pd.DataFrame({"arm": ["A", "B", "C", "A", "B", "C"]})
    out, cards = resolve_formats(df, profile(df))
    assert set(out["arm"].unique()) == {"A", "B", "C"}
    assert not any(c.card_id == "format_normalize_labels__arm" for c in cards)


# ---------------------------------------------------------------------------
# Determinism + idempotence
# ---------------------------------------------------------------------------


def test_resolve_idempotent_on_clean_input():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": ["A", "B", "A"]})
    out1, cards1 = resolve_formats(df, profile(df))
    out2, cards2 = resolve_formats(out1, profile(out1))
    pd.testing.assert_frame_equal(out1, out2)
    assert cards2 == []
