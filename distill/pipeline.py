"""Pipeline orchestration.

`run_pipeline` runs all six cleaning stages end to end and writes the five
output artifacts. It is the single orchestration path shared by:

  * the command-line interface (`distill.cli`);
  * the FastAPI web demo (`web/backend`);
  * the non-interactive "Do the data analysis" entry point.

Human-in-the-loop is handled through one optional `decide` callback. For
every judgment decision the pipeline surfaces, it calls `decide(card)`;
the callback is expected to resolve the card (confirm or override). If no
callback is given — or the callback leaves a card unresolved — the card's
`default_action` is applied and recorded as `agent_default`. That single
seam is what lets the same orchestration drive a silent CLI run, an
interactive web session, and the Claude Code subagent flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .binning import apply_decision as _apply_bin
from .binning import assess_binning
from .companions import apply_decision as _apply_companion
from .companions import detect_companion_matching
from .encoding import apply_decision as _apply_enc
from .encoding import assess_encoding
from .duplicates import apply_decision as _apply_dup
from .duplicates import detect_duplicates
from .format import apply_renames_to_target, resolve_formats
from .io import load
from .missingness import apply_decision as _apply_miss
from .missingness import detect_missingness
from .outliers import apply_decision as _apply_out
from .outliers import detect_outliers
from .outputs import write_all_outputs
from .profiler import profile, profile_summary_card
from .scaling import apply_decision as _apply_scale
from .scaling import assess_scaling
from .state import PipelineState, Session
from .types import DataCompanion, DecisionCard, DecisionStatus, Stage

if TYPE_CHECKING:
    pass

# Per-stage application function for resolved cards.
_STAGE_APPLY: dict[Stage, Callable] = {
    Stage.COMPANION: _apply_companion,
    Stage.MISSINGNESS: _apply_miss,
    Stage.OUTLIER: _apply_out,
    Stage.DUPLICATE: _apply_dup,
    Stage.BINNING: _apply_bin,
    Stage.SCALING: _apply_scale,
    Stage.ENCODING: _apply_enc,
}

# A decision callback: given a PENDING card, resolve it in place.
DecideFn = Callable[[DecisionCard], None]
# A progress callback: (event_name, payload).
EventFn = Callable[[str, dict[str, Any]], None]


@dataclass
class PipelineResult:
    """Everything a caller needs after a run."""

    state: PipelineState
    session: Session
    artifacts: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, Any]:
        s = self.state
        return {
            "session_id": s.session_id,
            "rows_in": s.n_rows_in,
            "rows_out": len(s.df),
            "cols_in": s.n_cols_in,
            "cols_out": len(s.df.columns),
            "n_decisions": len(s.log),
            "n_overrides": sum(
                1 for c in s.log if c.status == DecisionStatus.OVERRIDDEN
            ),
            "artifacts": {k: str(v) for k, v in self.artifacts.items()
                          if not isinstance(v, dict)},
        }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pipeline(
    input_path: Path | str,
    *,
    target_column: str | None = None,
    out_root: Path | str = "run_artifacts",
    session_id: str | None = None,
    decide: DecideFn | None = None,
    on_event: EventFn | None = None,
    write_outputs: bool = True,
    intake_metadata: str = "",
    companions: list[DataCompanion] | None = None,
) -> PipelineResult:
    """Run the full eight-stage cleaning pipeline.

    Parameters
    ----------
    input_path : dataset to clean (CSV / Excel / SPSS / ...).
    target_column : outcome variable, if any — recorded for downstream use.
    out_root : directory under which the session folder is created.
    session_id : explicit session id (default: time-ordered auto id).
    decide : optional callback to resolve each judgment DecisionCard.
        If None, every card uses its default_action.
    on_event : optional progress callback, called as
        on_event(event_name, payload) at each stage boundary.
    write_outputs : if True, write all five artifacts at the end.
    intake_metadata : free-text context the analyst provided at upload
        (study description, codebook notes, etc.).
    companions : non-tabular files registered at intake.
    """
    input_path = Path(input_path)
    df = load(input_path)
    state = PipelineState.new(
        input_path=input_path,
        df=df,
        target_column=target_column,
        session_id=session_id,
        intake_metadata=intake_metadata,
        companions=companions or [],
    )
    session = Session(session_id=state.session_id, root=out_root)

    def emit(name: str, payload: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(name, payload)

    emit("pipeline_start", {
        "session_id": state.session_id,
        "rows": state.n_rows_in, "cols": state.n_cols_in,
        "dataset": input_path.name,
    })

    # ----- Stage 1: Profiling -----
    shape = state.df.shape
    profiles = profile(state.df)
    state.profiles_initial = profiles
    state.log.append(profile_summary_card(state.df, profiles))
    _record(state, Stage.PROFILE, shape, "diagnostic only — no changes")
    emit("stage_done", {"stage": "profile", "n_variables": len(profiles)})

    # ----- Stage 1.5: Companion matching -----
    shape = state.df.shape
    companion_cards = detect_companion_matching(state)
    if companion_cards:
        _resolve_and_apply(state, companion_cards, decide)
        _record(state, Stage.COMPANION, shape,
                "subjects without companion handled")
        emit("stage_done", {
            "stage": "companion",
            "n_companions": len(state.companions),
            "n_unmatched": companion_cards[0].metadata.get("n_unmatched", 0),
        })
    else:
        _skip = DecisionCard(
            card_id="companion_matching",
            stage=Stage.COMPANION,
            issue="No companion files registered, or no match key set.",
            recommendation="skip",
            rationale="Stage skipped.",
            default_action="skip",
            status=DecisionStatus.SKIPPED,
        )
        _skip.resolve(action="skip", confirmed_by="agent_auto",
                      status=DecisionStatus.SKIPPED)
        state.log.append(_skip)
        _record(state, Stage.COMPANION, shape, "no companions — skipped")
        emit("stage_done", {"stage": "companion", "skipped": True})

    # ----- Stage 2: Format resolution (autonomous) -----
    shape = state.df.shape
    state.df, fmt_cards = resolve_formats(state.df, profiles)
    state.log.extend(fmt_cards)
    state.target_column = apply_renames_to_target(state.target_column, fmt_cards)
    _record(state, Stage.FORMAT, shape, "mechanical fixes (types, whitespace, names)")
    emit("stage_done", {"stage": "format", "n_fixes": len(fmt_cards)})
    profiles = profile(state.df)  # refresh after format changes

    # ----- Stage 3: Duplicates -----
    # Runs before missingness so missing-rate measurement and imputation
    # see the genuine de-duplicated sample, not repeated records.
    shape = state.df.shape
    _resolve_and_apply(state, detect_duplicates(state.df, profiles), decide)
    _record(state, Stage.DUPLICATE, shape, "duplicate rows removed")
    emit("stage_done", {"stage": "duplicate", "rows": len(state.df)})
    profiles = profile(state.df)  # refresh after duplicate removal

    # ----- Stage 4: Missingness -----
    shape = state.df.shape
    _resolve_and_apply(state, detect_missingness(state.df, profiles), decide)
    _record(state, Stage.MISSINGNESS, shape, "rows/variables dropped or imputed")
    emit("stage_done", {"stage": "missingness", "rows": len(state.df)})

    # ----- Stage 5: Outliers -----
    shape = state.df.shape
    _resolve_and_apply(state, detect_outliers(state.df, profiles), decide)
    _record(state, Stage.OUTLIER, shape, "flagged outliers handled")
    emit("stage_done", {"stage": "outlier", "rows": len(state.df)})

    # ----- Stage 6: Binning (only if warranted) -----
    profiles = profile(state.df)  # refresh after outlier changes
    shape = state.df.shape
    binning_cards = assess_binning(state.df, profiles)
    if binning_cards:
        _resolve_and_apply(state, binning_cards, decide)
    _record(state, Stage.BINNING, shape, "numeric columns discretised (if warranted)")
    emit("stage_done", {"stage": "binning", "n_suggestions": len(binning_cards)})

    # ----- Stage 7: Scaling (only if warranted) -----
    profiles = profile(state.df)  # refresh after any binning changes
    shape = state.df.shape
    scaling_cards = assess_scaling(state.df, profiles)
    if scaling_cards:
        _resolve_and_apply(state, scaling_cards, decide)
    _record(state, Stage.SCALING, shape, "numeric columns scaled (if warranted)")
    emit("stage_done", {"stage": "scaling", "n_suggestions": len(scaling_cards)})

    # ----- Stage 8: Encoding (only if warranted) -----
    profiles = profile(state.df)  # refresh after scaling changes
    shape = state.df.shape
    encoding_cards = assess_encoding(state.df, profiles)
    if encoding_cards:
        _resolve_and_apply(state, encoding_cards, decide)
    _record(state, Stage.ENCODING, shape, "categorical columns encoded (if warranted)")
    emit("stage_done", {"stage": "encoding", "n_suggestions": len(encoding_cards)})

    state.profiles_final = profile(state.df)

    # ----- Outputs -----
    result = PipelineResult(state=state, session=session)
    if write_outputs:
        out = write_all_outputs(state, session)
        result.artifacts = out["artifacts"]
        result.errors = out["errors"]
    emit("pipeline_done", result.summary)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_and_apply(
    state: PipelineState,
    cards: list[DecisionCard],
    decide: DecideFn | None,
) -> None:
    """Resolve every PENDING card (via `decide` or its default) and apply it.

    Cards are applied to `state.df` in order and appended to the log.
    Already-resolved cards (AGENT_AUTO / SKIPPED) are applied and logged
    as-is.
    """
    for card in cards:
        if card.status == DecisionStatus.PENDING:
            if decide is not None:
                decide(card)
            if card.status == DecisionStatus.PENDING:
                # No callback, or callback declined to resolve — use default.
                card.resolve(
                    action=card.default_action,
                    confirmed_by="agent_default",
                    status=DecisionStatus.AGENT_DEFAULT,
                )

        apply_fn = _STAGE_APPLY.get(card.stage)
        if apply_fn is not None:
            state.df, info = apply_fn(state.df, card)
            card.metadata["applied_info"] = info
        state.log.append(card)


def _record(
    state: PipelineState,
    stage: Stage,
    before_shape: tuple[int, int],
    reason: str,
) -> None:
    r0, c0 = before_shape
    r1, c1 = state.df.shape
    state.record_stage_io(stage, r0, c0, r1, c1, reason)
