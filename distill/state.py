"""Pipeline state and decision log.

PipelineState holds the live, in-memory state of a cleaning session: the
current DataFrame, accumulated VariableProfiles, and per-stage row counts.
It is not serializable (it holds a pandas DataFrame).

DecisionLog is the append-only audit trail of everything the agent did or
asked. It IS serializable — `dump(path)` writes the canonical
`decisions.json` artifact.

Session is a thin helper that manages the on-disk `run_artifacts/<id>/`
directory for a single cleaning session.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .types import (
    DataCompanion,
    DecisionCard,
    DecisionStatus,
    Stage,
    StageRowCount,
    VariableProfile,
)

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------


class DecisionLog:
    """Append-only ordered list of every decision in a session."""

    def __init__(self) -> None:
        self._entries: list[DecisionCard] = []

    # ----- mutation -----

    def append(self, card: DecisionCard) -> DecisionCard:
        self._entries.append(card)
        return card

    def extend(self, cards: Iterable[DecisionCard]) -> None:
        for c in cards:
            self.append(c)

    # ----- queries -----

    @property
    def entries(self) -> list[DecisionCard]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def by_stage(self, stage: Stage) -> list[DecisionCard]:
        return [e for e in self._entries if e.stage == stage]

    def by_id(self, card_id: str) -> DecisionCard | None:
        for e in self._entries:
            if e.card_id == card_id:
                return e
        return None

    def pending(self) -> list[DecisionCard]:
        return [e for e in self._entries if e.status == DecisionStatus.PENDING]

    def resolved(self) -> list[DecisionCard]:
        return [e for e in self._entries if e.status != DecisionStatus.PENDING]

    def overrides(self) -> list[DecisionCard]:
        return [e for e in self._entries if e.status == DecisionStatus.OVERRIDDEN]

    # ----- serialization -----

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "n_entries": len(self._entries),
            "entries": [e.to_dict() for e in self._entries],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def dump(self, path: Path) -> Path:
        """Write the log to `path` as UTF-8 JSON. Returns the path written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------


class PipelineState:
    """Mutable, in-memory state for a single cleaning session.

    Not Pydantic — it holds the live pandas DataFrame, which we don't want
    to serialize wholesale. Anything that needs to be persisted goes through
    DecisionLog or the artifact writers in `distill.io`.
    """

    def __init__(
        self,
        *,
        session_id: str,
        input_path: Path,
        df: "pd.DataFrame",
        target_column: str | None = None,
        intake_metadata: str = "",
        companions: list[DataCompanion] | None = None,
        companion_match_template: str | None = None,
        companion_match_mode: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.input_path = Path(input_path)
        self.df = df
        self.target_column = target_column
        # Free-text context the analyst provided at intake (codebook notes,
        # study description, etc.). Stored verbatim; surfaced in the report.
        self.intake_metadata: str = intake_metadata
        # Non-tabular files registered at intake (images, PDFs, etc.).
        self.companions: list[DataCompanion] = companions or []
        # Template used to derive the companion match key from a row.
        # Plain column name: "subject_id"
        # Compound: "{jurisdiction}_{period_id}"
        self.companion_match_template: str | None = companion_match_template
        # "presence_check" → add has_companion bool column only.
        # "full_join"      → emit code to load & merge actual companion files.
        # None             → companions documented in report but not joined.
        self.companion_match_mode: str | None = companion_match_mode
        # Output formats selected by the analyst at session end.
        # Set by the agent after asking the post-cleaning format question.
        # Defaults to CSV only; updated before emit_script() is called.
        self.output_formats: list[str] = ["csv"]
        # AI/doc-sourced variable descriptions (shown in the inspector panel),
        # keyed by variable name. Populated before report rendering so the
        # audit report can document what each variable means. Each value is a
        # dict: {"text": str, "source": "documentation" | "ai_inferred"}.
        self.variable_descriptions: dict[str, dict[str, str]] = {}
        self.started_at = datetime.now(timezone.utc)

        # initial dimensions (frozen at construction time)
        self.n_rows_in: int = len(df)
        self.n_cols_in: int = len(df.columns)

        # mutated as pipeline progresses
        self.current_stage: Stage | None = None
        self.profiles_initial: dict[str, VariableProfile] = {}
        self.profiles_final: dict[str, VariableProfile] = {}
        self.row_counts: list[StageRowCount] = []
        self.log = DecisionLog()

    # ----- factory -----

    @classmethod
    def new(
        cls,
        *,
        input_path: Path | str,
        df: "pd.DataFrame",
        target_column: str | None = None,
        session_id: str | None = None,
        intake_metadata: str = "",
        companions: list[DataCompanion] | None = None,
        companion_match_template: str | None = None,
        companion_match_mode: str | None = None,
    ) -> "PipelineState":
        return cls(
            session_id=session_id or _new_session_id(),
            input_path=Path(input_path),
            df=df,
            target_column=target_column,
            intake_metadata=intake_metadata,
            companions=companions or [],
            companion_match_template=companion_match_template,
            companion_match_mode=companion_match_mode,
        )

    # ----- convenience -----

    def record_stage_io(
        self,
        stage: Stage,
        rows_in: int,
        cols_in: int,
        rows_out: int,
        cols_out: int,
        reason: str = "",
    ) -> StageRowCount:
        entry = StageRowCount(
            stage=stage,
            rows_in=rows_in,
            cols_in=cols_in,
            rows_out=rows_out,
            cols_out=cols_out,
            rows_removed=max(rows_in - rows_out, 0),
            cols_removed=max(cols_in - cols_out, 0),
            reason=reason,
        )
        self.row_counts.append(entry)
        return entry


# ---------------------------------------------------------------------------
# Session — on-disk directory management
# ---------------------------------------------------------------------------


class Session:
    """Manages the `run_artifacts/<session_id>/` directory for a session."""

    def __init__(self, session_id: str, root: Path | str = "run_artifacts") -> None:
        self.session_id = session_id
        self.dir = Path(root) / session_id
        self.dir.mkdir(parents=True, exist_ok=True)

    # ----- canonical artifact paths -----

    @property
    def clean_csv(self) -> Path:
        return self.dir / "clean.csv"

    @property
    def report_md(self) -> Path:
        return self.dir / "report.md"

    @property
    def report_pdf(self) -> Path:
        return self.dir / "report.pdf"

    @property
    def flowchart_svg(self) -> Path:
        return self.dir / "flowchart.svg"

    @property
    def flowchart_png(self) -> Path:
        return self.dir / "flowchart.png"

    @property
    def decisions_json(self) -> Path:
        return self.dir / "decisions.json"

    @property
    def clean_script(self) -> Path:
        return self.dir / "clean_script.py"

    @property
    def requirements_txt(self) -> Path:
        return self.dir / "requirements.txt"

    @property
    def environment_yml(self) -> Path:
        return self.dir / "environment.yml"

    @property
    def profile_json(self) -> Path:
        return self.dir / "profile.json"

    # ----- helpers -----

    def write_profile(self, profiles: dict[str, VariableProfile]) -> Path:
        payload = {name: p.to_dict() for name, p in profiles.items()}
        self.profile_json.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        return self.profile_json


def _new_session_id() -> str:
    """Generate a short, time-ordered, human-friendly session ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:6]}"
