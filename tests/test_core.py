"""Smoke tests for the core types, state, and IO scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from distill import (
    DecisionCard,
    DecisionLog,
    DecisionStatus,
    PipelineState,
    Session,
    Stage,
    VariableProfile,
    VariableType,
)
from distill.io import SUPPORTED_EXTENSIONS, load, save_clean


# ---------------------------------------------------------------------------
# Decision card lifecycle
# ---------------------------------------------------------------------------


def test_decision_card_resolve_round_trip():
    card = DecisionCard(
        card_id="missing_age_001",
        stage=Stage.MISSINGNESS,
        variable="age",
        issue="4.2% missing (52 rows)",
        recommendation="median imputation (62.0)",
        rationale="MCAR; missingness uncorrelated with other vars.",
        alternatives=["drop_rows", "regression_impute", "leave_as_is"],
        default_action="median_impute",
    )
    assert card.status == DecisionStatus.PENDING
    card.resolve(
        action="median_impute",
        confirmed_by="user",
        status=DecisionStatus.CONFIRMED,
    )
    assert card.status == DecisionStatus.CONFIRMED
    assert card.action_taken == "median_impute"
    assert card.resolved_at is not None


def test_decision_card_serializes_to_json():
    card = DecisionCard(
        card_id="c1",
        stage=Stage.FORMAT,
        variable="dob",
        issue="dates stored as object",
        recommendation="parse with dayfirst=False",
        rationale="ISO-style values dominate samples.",
        default_action="parse_iso",
    )
    payload = card.to_dict()
    # Make sure it round-trips via plain JSON.
    s = json.dumps(payload)
    decoded = json.loads(s)
    assert decoded["card_id"] == "c1"
    assert decoded["stage"] == "format"


# ---------------------------------------------------------------------------
# DecisionLog
# ---------------------------------------------------------------------------


def test_decision_log_filters_and_dump(tmp_path: Path):
    log = DecisionLog()
    log.append(
        DecisionCard(
            card_id="p1",
            stage=Stage.PROFILE,
            issue="28 vars",
            recommendation="continue",
            rationale="ok",
            default_action="continue",
            status=DecisionStatus.AGENT_AUTO,
        )
    )
    log.append(
        DecisionCard(
            card_id="m1",
            stage=Stage.MISSINGNESS,
            variable="bmi",
            issue="18% missing",
            recommendation="regression impute",
            rationale="MAR; bmi ~ age + sex.",
            default_action="regression_impute",
        )
    )
    assert len(log) == 2
    assert len(log.pending()) == 1
    assert len(log.by_stage(Stage.MISSINGNESS)) == 1
    out = log.dump(tmp_path / "decisions.json")
    text = out.read_text(encoding="utf-8")
    parsed = json.loads(text)
    assert parsed["n_entries"] == 2
    assert parsed["entries"][0]["card_id"] == "p1"


# ---------------------------------------------------------------------------
# VariableProfile
# ---------------------------------------------------------------------------


def test_variable_profile_missing_rate():
    vp = VariableProfile(
        name="age",
        detected_type=VariableType.INTEGER,
        stored_dtype="float64",
        n_total=1000,
        n_missing=52,
        n_unique=80,
    )
    assert vp.missing_rate == pytest.approx(0.052)


# ---------------------------------------------------------------------------
# PipelineState
# ---------------------------------------------------------------------------


def test_pipeline_state_factory_and_row_count_recording(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    fake_input = tmp_path / "raw.csv"
    fake_input.write_text("a,b\n1,4\n2,5\n3,6\n", encoding="utf-8")

    state = PipelineState.new(input_path=fake_input, df=df, target_column=None)
    assert state.n_rows_in == 3
    assert state.n_cols_in == 2
    assert state.session_id  # non-empty

    entry = state.record_stage_io(
        Stage.PROFILE, rows_in=3, cols_in=2, rows_out=3, cols_out=2, reason="profile-only"
    )
    assert entry.rows_removed == 0
    assert len(state.row_counts) == 1


# ---------------------------------------------------------------------------
# Session directory
# ---------------------------------------------------------------------------


def test_session_creates_dir_and_exposes_paths(tmp_path: Path):
    s = Session(session_id="testsess", root=tmp_path)
    assert s.dir.is_dir()
    assert s.clean_csv.name == "clean.csv"
    assert s.decisions_json.parent == s.dir


# ---------------------------------------------------------------------------
# IO round-trip
# ---------------------------------------------------------------------------


def test_csv_load_and_save_roundtrip(tmp_path: Path):
    src = tmp_path / "raw.csv"
    src.write_text("a,b\n1,foo\n2,bar\n", encoding="utf-8")
    df = load(src)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2

    out = save_clean(df, original_path=src, out_dir=tmp_path / "out")
    assert "csv" in out
    assert out["csv"].read_text(encoding="utf-8").startswith("a,b")


def test_load_rejects_unknown_extension(tmp_path: Path):
    bad = tmp_path / "weird.zzz"
    bad.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load(bad)


def test_supported_extensions_includes_common_formats():
    for ext in [".csv", ".xlsx", ".tsv", ".parquet"]:
        assert ext in SUPPORTED_EXTENSIONS
