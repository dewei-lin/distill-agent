"""Distill Agent — core library.

Reusable, deterministic data-cleaning primitives shared by:
  - the Claude Code skill at .claude/skills/distill/
  - the FastAPI web demo at web/backend/

The library is intentionally framework-agnostic: it takes a pandas
DataFrame in and returns transformations + DecisionCard objects out.
The agent layer (Claude Code or the web app) decides which cards to
surface to the user and which to auto-confirm.
"""

from __future__ import annotations

from .missingness import (
    apply_decision as apply_missingness_decision,
    apply_default_for_all as apply_missingness_defaults,
    classify_pattern,
    detect_missingness,
    recommend_action as recommend_missingness_action,
)
from .duplicates import (
    apply_decision as apply_duplicate_decision,
    apply_default_for_all as apply_duplicate_defaults,
    detect_duplicates,
)
from .outliers import (
    apply_decision as apply_outlier_decision,
    apply_default_for_all as apply_outlier_defaults,
    detect_outliers,
    flag_outliers,
)
from .flowchart import render_flowchart
from .outputs import write_all_outputs, write_clean_data, write_decision_log
from .profiler import profile, profile_summary_card
from .report import render_report
from .script_gen import emit_script
from .state import DecisionLog, PipelineState, Session
from .types import (
    CategoricalStats,
    DecisionCard,
    DecisionStatus,
    MissingnessPattern,
    NumericStats,
    Stage,
    StageRowCount,
    VariableProfile,
    VariableType,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # state
    "DecisionLog",
    "PipelineState",
    "Session",
    # types
    "CategoricalStats",
    "DecisionCard",
    "DecisionStatus",
    "MissingnessPattern",
    "NumericStats",
    "Stage",
    "StageRowCount",
    "VariableProfile",
    "VariableType",
    # stages
    "profile",
    "profile_summary_card",
    "classify_pattern",
    "detect_missingness",
    "recommend_missingness_action",
    "apply_missingness_decision",
    "apply_missingness_defaults",
    "flag_outliers",
    "detect_outliers",
    "apply_outlier_decision",
    "apply_outlier_defaults",
    "detect_duplicates",
    "apply_duplicate_decision",
    "apply_duplicate_defaults",
    # outputs
    "write_clean_data",
    "write_decision_log",
    "write_all_outputs",
    "render_report",
    "render_flowchart",
    "emit_script",
]
