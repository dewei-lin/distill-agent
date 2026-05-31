"""Core data types for DataClean Agent.

These models are the contract between the Python core, the Claude Code
subagents, and the web-app frontend. Anything that crosses a process or
file boundary (decision log JSON, variable profile JSON, web SSE payloads)
is represented here.

Implementation note: we use stdlib `dataclasses` rather than pydantic so
the core library stays dep-light (pandas/numpy/scipy/sklearn only). The
web backend uses pydantic for its own request/response models — it's a
FastAPI dep anyway — but the core never needs it.

In-process objects that hold pandas frames live in `state.py`, not here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .action_labels import label_for, options_for


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Stage(str, Enum):
    """Pipeline stages, in canonical execution order.

    Duplicate detection runs *before* missingness: removing duplicated
    rows first means missing-value rates and imputations are measured on
    the genuine, de-duplicated sample rather than being skewed by repeated
    records.
    """

    PROFILE = "profile"
    COMPANION = "companion"
    FORMAT = "format"
    DUPLICATE = "duplicate"
    MISSINGNESS = "missingness"
    OUTLIER = "outlier"
    BINNING = "binning"
    SCALING = "scaling"
    ENCODING = "encoding"


class OutputFormat(str, Enum):
    """User-selected output formats produced by the generated clean_script.py."""
    CSV     = "csv"      # clean.csv — universal, opens anywhere
    PARQUET = "parquet"  # clean.parquet — typed, compressed, fast pandas/DuckDB
    HF      = "hf"       # clean_dataset/ — images/audio/text embedded, feeds Trainer/DataLoader
    HDF5    = "hdf5"     # clean.h5 — multi-frame arrays (video, multi-page PDF)


class VariableType(str, Enum):
    """The agent's best-guess semantic type for a variable, independent of
    how pandas happened to store it."""

    NUMERIC = "numeric"        # continuous float
    INTEGER = "integer"        # discrete whole number
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    STRING = "string"          # free text, high cardinality
    DATETIME = "datetime"
    UNKNOWN = "unknown"


class DecisionStatus(str, Enum):
    PENDING = "pending"              # surfaced to user, no answer yet
    CONFIRMED = "confirmed"          # user accepted the recommendation
    OVERRIDDEN = "overridden"        # user picked a different alternative
    AGENT_DEFAULT = "agent_default"  # non-interactive run; default_action applied
    AGENT_AUTO = "agent_auto"        # mechanical decision; never surfaced
    SKIPPED = "skipped"              # stage was skipped (e.g. no target column)


class MissingnessPattern(str, Enum):
    MCAR = "MCAR"  # missing completely at random
    MAR = "MAR"    # missing at random — depends on observed variables
    MNAR = "MNAR"  # missing not at random — depends on the missing value itself
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Variable profile
# ---------------------------------------------------------------------------


@dataclass
class NumericStats:
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    q25: float | None = None
    q75: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CategoricalStats:
    # Top-k most frequent values, as (value_as_string, count) pairs.
    top_values: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"top_values": [list(t) for t in self.top_values]}


@dataclass
class VariableProfile:
    """A point-in-time snapshot of a single variable.

    Built by the profiling stage and refreshed at the end of the pipeline so
    the report can show before/after.
    """

    name: str
    detected_type: VariableType
    stored_dtype: str            # raw pandas dtype string, e.g. "object"
    n_total: int
    n_missing: int
    n_unique: int
    sample_values: list[str] = field(default_factory=list)
    numeric: NumericStats | None = None
    categorical: CategoricalStats | None = None
    # Any extra agent-detected flags (e.g. "looks like a date stored as str")
    flags: list[str] = field(default_factory=list)

    @property
    def missing_rate(self) -> float:
        return self.n_missing / self.n_total if self.n_total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "detected_type": self.detected_type.value,
            "stored_dtype": self.stored_dtype,
            "n_total": self.n_total,
            "n_missing": self.n_missing,
            "n_unique": self.n_unique,
            "missing_rate": self.missing_rate,
            "sample_values": list(self.sample_values),
            "numeric": self.numeric.to_dict() if self.numeric else None,
            "categorical": self.categorical.to_dict() if self.categorical else None,
            "flags": list(self.flags),
        }


# ---------------------------------------------------------------------------
# Decision cards — the HITL contract
# ---------------------------------------------------------------------------


@dataclass
class DecisionCard:
    """A single point of judgment surfaced to the analyst.

    Every modification of the dataset that requires human judgment is
    represented as one of these. Mechanical changes use this same model with
    `status = AGENT_AUTO` so the audit trail is uniform.
    """

    card_id: str
    stage: Stage
    issue: str                           # one-line summary of what was found
    recommendation: str                  # one-line summary of agent's pick
    rationale: str                       # why the agent recommends it
    default_action: str                  # alternative applied if no human answers
    variable: str | None = None          # None for cross-variable decisions
    alternatives: list[str] = field(default_factory=list)

    # Per-stage structured payload (e.g. {"pattern": "MCAR", "missing_n": 52}).
    metadata: dict[str, Any] = field(default_factory=dict)

    # Resolution
    status: DecisionStatus = DecisionStatus.PENDING
    action_taken: str | None = None
    confirmed_by: str | None = None      # "user", "agent_default", "agent_auto"
    resolved_at: datetime | None = None
    user_note: str | None = None

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def resolve(
        self,
        *,
        action: str,
        confirmed_by: str,
        status: DecisionStatus,
        note: str | None = None,
    ) -> None:
        """Mark this card as resolved. Caller decides which status applies."""
        self.action_taken = action
        self.confirmed_by = confirmed_by
        self.status = status
        self.user_note = note
        self.resolved_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "stage": self.stage.value,
            "variable": self.variable,
            "issue": self.issue,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
            "alternatives": list(self.alternatives),
            "default_action": self.default_action,
            # Plain-language rendering of every action, so neither the UI
            # nor the decision-log JSON ever exposes a raw identifier.
            "recommended_action": self.default_action,
            "recommended_label": label_for(self.stage.value, self.default_action),
            "options": options_for(
                self.stage.value, self.default_action, self.alternatives
            ),
            "action_taken_label": (
                label_for(self.stage.value, self.action_taken)
                if self.action_taken
                else None
            ),
            "metadata": dict(self.metadata),
            "status": self.status.value,
            "action_taken": self.action_taken,
            "confirmed_by": self.confirmed_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "user_note": self.user_note,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Pipeline row-count tracking (used by the CONSORT flowchart)
# ---------------------------------------------------------------------------


@dataclass
class StageRowCount:
    """Row/column counts at the boundary of one pipeline stage."""

    stage: Stage
    rows_in: int
    cols_in: int
    rows_out: int
    cols_out: int
    rows_removed: int = 0
    cols_removed: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "rows_in": self.rows_in,
            "cols_in": self.cols_in,
            "rows_out": self.rows_out,
            "cols_out": self.cols_out,
            "rows_removed": self.rows_removed,
            "cols_removed": self.cols_removed,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Companion files (non-tabular uploads attached to a session)
# ---------------------------------------------------------------------------


@dataclass
class DataCompanion:
    """A non-tabular file attached to a cleaning session (image, audio, etc.).

    These files are never merged into the DataFrame; they are registered so
    the audit report can document what was uploaded alongside the data.
    """

    path: Path
    mime_type: str          # e.g. "image/png", "application/pdf"
    description: str = ""   # user-provided label or auto-detected from extension
    # Subject identifier extracted from the filename stem at intake.  Used by
    # script_gen to build the companion manifest for the reproducible script.
    subject_id: str = ""

    def __post_init__(self) -> None:
        if not self.subject_id:
            self.subject_id = Path(self.path).stem

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.path.name,
            "mime_type": self.mime_type,
            "description": self.description,
            "subject_id": self.subject_id,
        }
