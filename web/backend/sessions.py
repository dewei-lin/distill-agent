"""Web session management for the Distill Agent demo backend.

This module is the *framework-agnostic* core of the web app. It holds one
`WebSession` per uploaded dataset and exposes the cleaning pipeline as a
sequence of discrete, resumable steps:

    upload -> profiling -> format -> [duplicate, missingness, outlier,
    each: detect then resolve] -> finalize

The web frontend orchestrates that sequence with plain request/response
calls — no threads, no SSE, no blocking callbacks. Each judgment stage is
two steps: `detect_stage` returns the pending decision cards; the user
answers them; `resolve_stage` applies the answers.

Keeping this logic out of `app.py` means it can be unit-tested without a
running FastAPI server (which is how it is tested).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import uuid as _uuid

from distill.binning import apply_decision as apply_binning
from distill.binning import assess_binning
from distill.companions import apply_decision as apply_companion
from distill.companions import detect_companion_matching, validate_template
from distill.duplicates import apply_decision as apply_duplicate
from distill.duplicates import detect_duplicates
from distill.encoding import apply_decision as apply_encoding
from distill.encoding import assess_encoding, decompose_datetime, frequency_encode
from distill.format import apply_renames_to_target, resolve_formats
from distill.io import load
from distill.missingness import apply_decision as apply_missingness
from distill.missingness import detect_missingness
from distill.outliers import apply_decision as apply_outlier
from distill.outliers import detect_outliers
from distill.outputs import write_all_outputs
from distill.profiler import profile, profile_summary_card
from distill.scaling import apply_decision as apply_scaling
from distill.scaling import assess_scaling
from distill.scaling import scale_column
from distill.binning import bin_column
from distill.state import PipelineState, Session
from distill.types import DecisionCard, DecisionStatus, Stage

# Core cleaning stages (duplicates → missingness → outlier).
_JUDGMENT_STAGES: list[tuple[Stage, Any]] = [
    (Stage.DUPLICATE, apply_duplicate),
    (Stage.MISSINGNESS, apply_missingness),
    (Stage.OUTLIER, apply_outlier),
]
_APPLY = dict(_JUDGMENT_STAGES)

# Feature engineering sub-stage apply functions.
_FE_APPLY: dict[Stage, Any] = {
    Stage.BINNING: apply_binning,
    Stage.SCALING: apply_scaling,
    Stage.ENCODING: apply_encoding,
}

HIST_BINS = 15


class WebSession:
    """One cleaning session backing the web UI."""

    def __init__(
        self,
        session_id: str,
        input_path: Path,
        df: pd.DataFrame,
        out_root: Path,
        target_column: str | None = None,
        doc_text: str = "",
        ingestion: dict[str, Any] | None = None,
        companions: list[Any] | None = None,
        hold_out_df: "pd.DataFrame | None" = None,
    ) -> None:
        self.state = PipelineState.new(
            input_path=input_path, df=df, target_column=target_column,
            session_id=session_id,
            companions=companions or [],
        )
        self.session = Session(session_id=session_id, root=out_root)
        self.created_at = time.time()
        # Pristine copy of the intake DataFrame + identity, kept so "Start Over"
        # can rewind the pipeline to stage 1 with the original data intact
        # (no re-upload). Stored once and never mutated.
        self._original_df = df.copy()
        self._original_target = target_column
        self._original_input_path = input_path
        # Persist the pre-cleaning intake snapshot so the reproducibility check
        # always runs against the joined, unmodified source — even when the
        # session was created from multiple files merged at ingestion.
        try:
            raw_intake_path = self.session.dir / "raw_intake.csv"
            df.to_csv(raw_intake_path, index=False, lineterminator="\n")
        except Exception:
            pass
        self.profiles: dict[str, Any] = {}
        self.stage_cards: dict[Stage, list[DecisionCard]] = {}
        self.finalized = False
        # Documentation text + ingestion report, set by the upload flow.
        self.doc_text = doc_text or ""
        self.ingestion = ingestion or {}
        # Companion files (images, audio, etc.) uploaded alongside the data.
        self.companions: list[Any] = companions or []
        # Hold-out / validation DataFrame (same schema; cleaned separately).
        self.hold_out_df: "pd.DataFrame | None" = hold_out_df
        # "reference_only" (default) or "bundle_all"
        self.companion_mode: str = "reference_only"
        # Cache for variable descriptions (doc-sourced or LLM-inferred) so
        # clicking a variable multiple times never re-calls the LLM.
        self._desc_cache: dict[str, tuple[str | None, str | None]] = {}

    # ----- start over: rewind to stage 1, keep the data -----

    def reset_pipeline(self) -> None:
        """Rewind the session to its post-intake state without re-uploading.

        Restores the pristine intake DataFrame and rebuilds a fresh
        PipelineState, discarding every cleaning decision, profile, and stage
        card. The uploaded data, companions, target, and documentation are
        preserved so the analyst can re-run the cleaning pipeline from scratch.
        """
        self.state = PipelineState.new(
            input_path=self._original_input_path,
            df=self._original_df.copy(),
            target_column=self._original_target,
            session_id=self.session.session_id,
            companions=self.companions or [],
        )
        self.profiles = {}
        self.stage_cards = {}
        self.finalized = False
        self.companion_mode = "reference_only"
        self._desc_cache = {}

    # ----- stage 1: profiling -----

    def run_profiling(self) -> dict[str, Any]:
        self.profiles = profile(self.state.df)
        self.state.profiles_initial = self.profiles
        self.state.log.append(profile_summary_card(self.state.df, self.profiles))
        self._record(Stage.PROFILE, self.state.df.shape, "diagnostic only")
        result: dict[str, Any] = {
            "session_id": self.state.session_id,
            "n_rows": int(self.state.df.shape[0]),
            "n_cols": int(self.state.df.shape[1]),
            "target_column": self.state.target_column,
            "profile": [p.to_dict() for p in self.profiles.values()],
            "ingestion": self.ingestion,
            "dataset_name": self.state.input_path.name,
        }
        if self.companions:
            n_img = sum(1 for c in self.companions if "image" in c.mime_type)
            n_other = len(self.companions) - n_img
            result["companion_card"] = {
                "n_total": len(self.companions),
                "n_images": n_img,
                "n_other": n_other,
                "files": [c.to_dict() for c in self.companions[:20]],
            }
        return result

    # ----- stage 1.5: companion matching -----

    def run_companion_stage(
        self, mode: str, match_template: str | None = None
    ) -> dict[str, Any]:
        """Detect companion-subject coverage and return any pending decision card.

        This is the canonical Stage COMPANION entry point for the web flow.
        Must be called after run_profiling() and before run_format().

        *mode* controls how companion files are packaged in the output
        (``"reference_only"`` or ``"bundle_all"``).  Subject matching is
        independent: it runs whenever *match_template* is provided.
        """
        before = self.state.df.shape
        self.set_companion_mode(mode)
        card = self.set_companion_match_template(match_template) if match_template else None

        if card is None and not self.state.log.by_id("companion_matching"):
            from distill.types import DecisionCard as _DC
            _skip = _DC(
                card_id="companion_matching",
                stage=Stage.COMPANION,
                issue="Companion files registered — no subject matching requested.",
                recommendation="skip",
                rationale="No match template provided; matching skipped.",
                default_action="skip",
                status=DecisionStatus.SKIPPED,
            )
            _skip.resolve(action="skip", confirmed_by="agent_auto",
                          status=DecisionStatus.SKIPPED)
            self.state.log.append(_skip)

        self._record(Stage.COMPANION, before,
                     "companion matching" if card else "companion stage skipped")
        return {
            "stage": Stage.COMPANION.value,
            "mode": self.companion_mode,
            "match_template": self.state.companion_match_template,
            "n_companions": len(self.companions),
            "card": card.to_dict() if card else None,
        }

    def set_companion_mode(self, mode: str) -> None:
        # Only "reference_only" is supported. "bundle_all" was removed — when
        # companions are present the recommended output is the Python script,
        # not a bundled ZIP.
        if mode not in ("reference_only",):
            mode = "reference_only"
        self.companion_mode = mode
        if not self.state.companion_match_template:
            self.state.companion_match_mode = None

    def set_companion_match_template(self, template: str) -> "DecisionCard | None":
        """Validate *template* against current columns and run matching analysis.

        *template* may be a plain column name (``"subject_id"``) or a Python
        format string referencing one or more columns
        (``"{jurisdiction}_{period_id}"``).

        Returns the PENDING DecisionCard when unmatched subjects exist, an
        AGENT_AUTO card when every subject matched, or None when companions/
        template are absent.
        """
        missing = validate_template(template, list(self.state.df.columns))
        if missing:
            raise ValueError(
                f"Template {template!r} references unknown column(s): {missing}. "
                f"Available: {list(self.state.df.columns)}"
            )
        self.state.companion_match_template = template
        self.state.companion_match_mode = "presence_check"
        return self._analyze_companion_matching()

    def propose_companion_match(self) -> dict[str, Any]:
        """Ask the LLM to propose a match template from uploaded documentation.

        Uses the session's doc_text (ingested at upload time) plus column names
        and companion filename stems as context.  Falls back gracefully when no
        API key is configured.
        """
        from .llm import propose_companion_match as _propose
        stems = [c.subject_id for c in self.companions[:30]]
        return _propose(
            doc_text=self.doc_text or "",
            columns=list(self.state.df.columns),
            filename_stems=stems,
        )

    def _analyze_companion_matching(self) -> "DecisionCard | None":
        if not self.companions or not self.state.companion_match_template:
            return None
        self.state.log._entries = [
            e for e in self.state.log._entries if e.card_id != "companion_matching"
        ]
        cards = detect_companion_matching(self.state)
        if not cards:
            return None
        card = cards[0]
        self.state.log.append(card)
        return card

    def resolve_companion_matching(
        self, action: str, note: str | None = None
    ) -> dict[str, Any]:
        """Apply the analyst's decision about companion matching.

        Incomplete coverage actions : keep_rows | add_indicator | drop_rows | flag_caveat
        Complete coverage actions   : exclude_path | include_path
        """
        card = self.state.log.by_id("companion_matching")
        if card is None:
            raise ValueError("No companion_matching card in the decision log.")
        if card.status != DecisionStatus.PENDING:
            raise ValueError(f"Card already resolved with action={card.action_taken!r}.")
        card.resolve(action=action, confirmed_by="user",
                     status=DecisionStatus.CONFIRMED, note=note)
        self.state.df, info = apply_companion(self.state.df, card)
        return {
            "action": action,
            "n_dropped": info.get("rows_dropped", 0),
            "n_rows_remaining": len(self.state.df),
            "include_companion_path": card.metadata.get("include_companion_path", False),
            "include_companion_indicator": card.metadata.get("include_companion_indicator", False),
        }

    # ----- stage 2: format resolution (autonomous) -----

    def run_format(self) -> dict[str, Any]:
        before = self.state.df.shape
        self.state.df, cards = resolve_formats(self.state.df, self.profiles)
        self.state.log.extend(cards)
        self.state.target_column = apply_renames_to_target(
            self.state.target_column, cards
        )
        self._record(Stage.FORMAT, before, "mechanical fixes")
        self.profiles = profile(self.state.df)  # refresh
        return {
            "cards": [c.to_dict() for c in cards],
            "columns": [str(c) for c in self.state.df.columns],
            "target_column": self.state.target_column,
        }

    # ----- judgment stages: detect then resolve -----

    def detect_stage(self, stage: Stage) -> dict[str, Any]:
        """Run a cleaning judgment stage's detector and return decision cards."""
        if stage == Stage.MISSINGNESS:
            cards = detect_missingness(self.state.df, self.profiles)
        elif stage == Stage.OUTLIER:
            cards = detect_outliers(self.state.df, self.profiles)
        elif stage == Stage.DUPLICATE:
            cards = detect_duplicates(self.state.df, self.profiles)
        else:
            raise ValueError(f"{stage} is not a supported judgment stage")
        self.stage_cards[stage] = cards
        return {
            "stage": stage.value,
            "cards": [c.to_dict() for c in cards],
            "n_pending": sum(1 for c in cards if c.status == DecisionStatus.PENDING),
        }

    # ----- feature engineering (compound stage) -----

    def detect_feature_engineering(self) -> dict[str, Any]:
        """Run all FE assessments and return cards + column metadata.

        Returns cards from binning, scaling, and encoding assessments
        together with the full column list for the manual selector.
        """
        self.profiles = profile(self.state.df)  # refresh before assessing
        bin_cards = assess_binning(self.state.df, self.profiles)
        scale_cards = assess_scaling(self.state.df, self.profiles)
        enc_cards = assess_encoding(self.state.df, self.profiles)
        self.stage_cards[Stage.BINNING] = bin_cards
        self.stage_cards[Stage.SCALING] = scale_cards
        self.stage_cards[Stage.ENCODING] = enc_cards

        all_cards = bin_cards + scale_cards + enc_cards
        columns = [
            {
                "name": str(col),
                "type": (
                    self.profiles[col].detected_type.value
                    if col in self.profiles
                    else "unknown"
                ),
                "n_unique": int(self.profiles[col].n_unique) if col in self.profiles else 0,
            }
            for col in self.state.df.columns
        ]
        return {
            "cards": [c.to_dict() for c in all_cards],
            "columns": columns,
            "n_suggestions": len(all_cards),
        }

    def resolve_feature_engineering(
        self, resolutions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Apply analyst decisions for all FE sub-stages."""
        all_cards: list[DecisionCard] = []
        for stage in (Stage.BINNING, Stage.SCALING, Stage.ENCODING):
            all_cards.extend(self.stage_cards.get(stage, []))

        answers = {r["card_id"]: r for r in resolutions}
        before = self.state.df.shape

        for card in all_cards:
            if card.status == DecisionStatus.PENDING:
                ans = answers.get(card.card_id)
                if ans is None:
                    card.resolve(
                        action=card.default_action,
                        confirmed_by="agent_default",
                        status=DecisionStatus.AGENT_DEFAULT,
                    )
                else:
                    action = ans.get("action", card.default_action)
                    is_default = action == card.default_action
                    card.resolve(
                        action=action,
                        confirmed_by="user",
                        status=(
                            DecisionStatus.CONFIRMED
                            if is_default
                            else DecisionStatus.OVERRIDDEN
                        ),
                        note=ans.get("note"),
                    )
            apply_fn = _FE_APPLY.get(card.stage)
            if apply_fn is not None:
                self.state.df, info = apply_fn(self.state.df, card)
                card.metadata["applied_info"] = info
            self.state.log.append(card)

        self._record(Stage.BINNING, before, "feature engineering decisions applied")
        return {
            "n_rows": int(self.state.df.shape[0]),
            "n_cols": int(self.state.df.shape[1]),
            "columns": [str(c) for c in self.state.df.columns],
            "applied": [
                {
                    "card_id": c.card_id,
                    "action": c.action_taken,
                    "status": c.status.value,
                }
                for c in all_cards
            ],
        }

    def apply_on_demand_transform(
        self, col: str, operation: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply a user-initiated transform immediately from the FE selector.

        The original column is always preserved. Every transform is logged
        as a user-confirmed DecisionCard.
        """
        if col not in self.state.df.columns:
            raise ValueError(f"Column '{col}' not found in dataset")

        before = self.state.df.shape
        new_cols: list[str] = []
        action_desc = ""
        stage = Stage.ENCODING

        if operation in ("bin_quantile", "bin_equal_width", "bin_natural_breaks"):
            method = operation[4:]  # strip "bin_"
            n_bins = int(params.get("n_bins", 4))
            self.state.df = bin_column(self.state.df, col, method, n_bins)
            new_col = f"{col}_bin"
            new_cols = [new_col]
            action_desc = f"Binned '{col}' ({method}, {n_bins} bins) → '{new_col}'"
            stage = Stage.BINNING

        elif operation in ("scale_standard", "scale_minmax", "scale_robust", "scale_log"):
            method = operation[6:]  # strip "scale_"
            self.state.df = scale_column(self.state.df, col, method)
            new_col = f"{col}_scaled"
            new_cols = [new_col]
            action_desc = f"Scaled '{col}' ({method}) → '{new_col}'"
            stage = Stage.SCALING

        elif operation == "frequency_encode":
            self.state.df = frequency_encode(self.state.df, col)
            new_col = f"{col}_freq"
            new_cols = [new_col]
            action_desc = f"Frequency encoded '{col}' → '{new_col}'"
            stage = Stage.ENCODING

        elif operation == "datetime_decompose":
            components = params.get(
                "components", ["year", "month", "day", "dayofweek"]
            )
            self.state.df = decompose_datetime(self.state.df, col, components)
            new_cols = [
                f"{col}_{c}"
                for c in components
                if f"{col}_{c}" in self.state.df.columns
            ]
            action_desc = f"Decomposed datetime '{col}' into {components}"
            stage = Stage.ENCODING

        else:
            raise ValueError(f"Unknown transform operation: {operation!r}")

        # Log as a confirmed user decision card.
        card = DecisionCard(
            card_id=f"ondemand_{col}_{_uuid.uuid4().hex[:6]}",
            stage=stage,
            variable=col,
            issue="User-requested transform",
            recommendation=action_desc,
            rationale="Applied on user request via the feature engineering selector.",
            default_action=operation,
        )
        card.resolve(
            action=operation,
            confirmed_by="user",
            status=DecisionStatus.CONFIRMED,
        )
        self.state.log.append(card)
        self._record(stage, before, action_desc)

        return {
            "n_rows": int(self.state.df.shape[0]),
            "n_cols": int(self.state.df.shape[1]),
            "new_cols": new_cols,
            "action": action_desc,
            "columns": [str(c) for c in self.state.df.columns],
        }

    def resolve_stage(
        self,
        stage: Stage,
        resolutions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply the user's answers for a judgment stage.

        `resolutions` is a list of {card_id, action, note?}. Cards not
        mentioned fall back to their default_action.
        """
        cards = self.stage_cards.get(stage)
        if cards is None:
            raise ValueError(f"detect_stage({stage}) must run before resolve")
        answers = {r["card_id"]: r for r in resolutions}
        before = self.state.df.shape

        for card in cards:
            if card.status == DecisionStatus.PENDING:
                ans = answers.get(card.card_id)
                if ans is None:
                    card.resolve(
                        action=card.default_action,
                        confirmed_by="agent_default",
                        status=DecisionStatus.AGENT_DEFAULT,
                    )
                else:
                    action = ans.get("action", card.default_action)
                    is_default = action == card.default_action
                    card.resolve(
                        action=action,
                        confirmed_by="user",
                        status=(DecisionStatus.CONFIRMED if is_default
                                else DecisionStatus.OVERRIDDEN),
                        note=ans.get("note"),
                    )
            apply_fn = _APPLY.get(card.stage)
            if apply_fn is not None:
                self.state.df, info = apply_fn(self.state.df, card)
                card.metadata["applied_info"] = info
                # Free large per-row arrays — apply_decision has consumed them.
                for _k in ("flagged_indices", "implausible_indices", "flagged_values"):
                    card.metadata.pop(_k, None)
            self.state.log.append(card)

        self._record(stage, before, f"{stage.value} decisions applied")
        return {
            "stage": stage.value,
            "n_rows": int(self.state.df.shape[0]),
            "n_cols": int(self.state.df.shape[1]),
            "applied": [
                {"card_id": c.card_id, "action": c.action_taken,
                 "status": c.status.value}
                for c in cards
            ],
        }

    # ----- finalize: write the five artifacts -----

    def finalize(self, output_formats: list[str] | None = None) -> dict[str, Any]:
        self.state.profiles_final = profile(self.state.df)
        if output_formats:
            self.state.output_formats = output_formats
        # Capture variable descriptions (doc-sourced or AI-inferred) so the
        # audit report documents what each variable means.
        self.populate_descriptions()
        result = write_all_outputs(self.state, self.session, repro_check_timeout=30)
        self.finalized = True
        artifacts: dict[str, str] = {}
        for k, v in result["artifacts"].items():
            if isinstance(v, Path):
                artifacts[k] = v.name
            elif isinstance(v, str):
                artifacts[k] = Path(v).name
            elif isinstance(v, dict):
                # format-keyed file dict, e.g. {"csv": Path(...), "xlsx": Path(...)}
                for fmt, path in v.items():
                    if isinstance(path, (str, Path)):
                        artifacts[f"{k}_{fmt}"] = Path(path).name
        repro = result.get("checks", {}).get("reproducibility")
        return {
            "artifacts": artifacts,
            "errors": result["errors"],
            "reproducibility_check": repro,
            "summary": {
                "dataset": self.state.input_path.name,
                "rows_in": self.state.n_rows_in,
                "rows_out": int(len(self.state.df)),
                "cols_in": self.state.n_cols_in,
                "cols_out": int(len(self.state.df.columns)),
                "n_decisions": len(self.state.log),
                "n_overrides": sum(
                    1 for c in self.state.log
                    if c.status == DecisionStatus.OVERRIDDEN
                ),
            },
        }

    # ----- variable inspector data (right panel) -----

    def variable_data(self, name: str) -> dict[str, Any]:
        """Histogram + summary stats for one variable, for the inspector."""
        if name not in self.state.df.columns:
            raise KeyError(name)
        s = self.state.df[name]
        n_total = len(s)
        n_missing = int(s.isna().sum())
        out: dict[str, Any] = {
            "name": name,
            "n_total": n_total,
            "n_missing": n_missing,
            "missing_rate": (n_missing / n_total) if n_total else 0.0,
        }
        if pd.api.types.is_numeric_dtype(s):
            # Drop both NaN *and* Inf — np.histogram chokes on Inf edges.
            num = pd.to_numeric(s, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            out["n_inf"] = int(((s == np.inf) | (s == -np.inf)).sum())
            if len(num) > 0:
                counts, edges = np.histogram(num, bins=HIST_BINS)
                def _sf(v: float) -> float:
                    import math
                    f = float(v)
                    return 0.0 if (math.isnan(f) or math.isinf(f)) else f
                out.update({
                    "kind": "numeric",
                    "mean": _sf(num.mean()),
                    "median": _sf(num.median()),
                    "std": _sf(num.std(ddof=1)) if len(num) > 1 else 0.0,
                    "min": _sf(num.min()),
                    "max": _sf(num.max()),
                    "bins": [int(c) for c in counts],
                    "bin_edges": [_sf(e) for e in edges],
                })
            else:
                out["kind"] = "numeric"
                out["bins"] = []
        else:
            vc = s.dropna().astype(str).value_counts().head(HIST_BINS)
            out.update({
                "kind": "categorical",
                "n_unique": int(s.nunique(dropna=True)),
                "top_values": [[str(k), int(v)] for k, v in vc.items()],
            })
        return out

    def describe_from_docs(self, name: str) -> str | None:
        """Search documentation text for a description of `name`.

        Looks for a line/paragraph containing the variable name and returns
        the surrounding context (up to ~250 chars). Returns None if no match.
        """
        import re
        text = self.doc_text or ""
        if not text:
            return None
        # Build a flexible pattern: underscores may appear as spaces or hyphens
        flexible = re.escape(name).replace(r"\_", r"[_\s\-]?")
        pat = re.compile(r"(?i)\b" + flexible + r"\b")
        m = pat.search(text)
        if not m:
            return None
        # Grab the surrounding paragraph (split on blank lines or line breaks)
        start = max(0, m.start() - 15)
        end = min(len(text), m.end() + 280)
        snippet = text[start:end].strip()
        # Trim to at most 2 non-empty lines
        lines = [l.strip() for l in snippet.splitlines() if l.strip()]
        for i, line in enumerate(lines):
            if pat.search(line):
                return " ".join(lines[i : i + 2])[:280]
        return snippet[:280]

    def describe_variable(self, name: str) -> tuple[str | None, str | None]:
        """Return (description, source) for a variable, generating once and caching.

        `source` is "documentation" when the description came from the uploaded
        codebook, "ai_inferred" when the LLM filled the gap, or None when no
        description could be produced. Results are memoised in `_desc_cache` so
        repeated inspector clicks and the finalize sweep never re-call the LLM.
        """
        from .llm import answer_question, chat_available

        if name in self._desc_cache:
            return self._desc_cache[name]

        desc: str | None = None
        source: str | None = None

        if chat_available():
            p = self.profiles.get(name)
            if p:
                doc_snippet = self.describe_from_docs(name) or ""
                has_docs = bool(self.doc_text)
                if doc_snippet:
                    source_hint = "Use the documentation excerpt as the primary source."
                    source = "documentation"
                elif has_docs:
                    source_hint = (
                        "Documentation was provided but this variable was not found in it. "
                        "End your description with: (AI estimate — verify against codebook.)"
                    )
                    source = "ai_inferred"
                else:
                    source_hint = (
                        "No documentation was provided. "
                        "End your description with: (AI estimate — needs verification.)"
                    )
                    source = "ai_inferred"

                inferred = answer_question(
                    f"Write 1–2 sentences describing the variable '{name}' for a data analyst. "
                    f"Cover all three: "
                    f"(a) expand any abbreviations in the variable name; "
                    f"(b) state whether it is continuous/numeric or categorical/leveled; "
                    f"(c) what role it likely plays in this analysis "
                    f"(e.g. outcome, predictor, identifier, confounder). "
                    f"{source_hint}",
                    {
                        "variable": name,
                        "profile": p.to_dict(),
                        "dataset": self.state.input_path.name,
                        "documentation_excerpt": doc_snippet or "(not found in docs)",
                    },
                )
                if inferred.get("available") and inferred.get("answer"):
                    desc = inferred["answer"]

        if desc is None:
            desc = self.describe_from_docs(name)
            source = "documentation" if desc else None

        self._desc_cache[name] = (desc, source)
        return desc, source

    def populate_descriptions(self, *, budget_s: float = 15.0, max_workers: int = 4) -> None:
        """Generate descriptions for every current column and store on the state.

        Called at finalize so the audit report can document what each variable
        means. Uses the same cache as the inspector, so already-viewed variables
        cost nothing.

        Bounded by a hard wall-clock budget: cached descriptions are collected
        for free, then uncached columns are described concurrently in a thread
        pool until ``budget_s`` elapses. Any columns not finished within the
        budget are simply left out — the inspector fills them in lazily on
        demand, and a missing description must never block or slow hand-off.
        Failures on any single variable are swallowed for the same reason.
        """
        import concurrent.futures as _cf
        import time as _time

        out: dict[str, dict[str, str]] = {}
        cols = [str(c) for c in self.state.df.columns]

        # 1) Cached columns (already viewed in the inspector) are free.
        pending: list[str] = []
        for name in cols:
            if name in self._desc_cache:
                desc, source = self._desc_cache[name]
                if desc:
                    out[name] = {"text": desc, "source": source or "ai_inferred"}
            else:
                pending.append(name)

        # 2) Describe the rest concurrently under a shared wall-clock budget.
        #    `as_completed(timeout=budget_s)` stops *waiting* once the budget
        #    elapses (it raises TimeoutError), so this method always returns
        #    promptly and finalize never hangs. Python threads can't be force
        #    -killed, so any in-flight LLM call simply finishes in the
        #    background and caches its own result for the next inspector click.
        if pending:
            ex = _cf.ThreadPoolExecutor(max_workers=max_workers)
            futures = {ex.submit(self.describe_variable, n): n for n in pending}
            try:
                for fut in _cf.as_completed(futures, timeout=budget_s):
                    name = futures[fut]
                    try:
                        desc, source = fut.result()
                    except Exception:
                        desc, source = None, None
                    if desc:
                        out[name] = {"text": desc, "source": source or "ai_inferred"}
            except _cf.TimeoutError:
                pass  # budget spent — leave the rest to lazy inspector fills
            finally:
                # Don't join: return now, let stragglers finish in the background.
                ex.shutdown(wait=False, cancel_futures=True)

        self.state.variable_descriptions = out

    # ----- helpers -----

    def _record(self, stage: Stage, before_shape, reason: str) -> None:
        r0, c0 = before_shape
        r1, c1 = self.state.df.shape
        self.state.record_stage_io(stage, r0, c0, r1, c1, reason)

    def status(self) -> dict[str, Any]:
        return {
            "session_id": self.state.session_id,
            "dataset": self.state.input_path.name,
            "n_rows": int(self.state.df.shape[0]),
            "n_cols": int(self.state.df.shape[1]),
            "target_column": self.state.target_column,
            "n_decisions": len(self.state.log),
            "finalized": self.finalized,
        }


class SessionStore:
    """In-memory store of active web sessions (fine for a single-process
    demo server)."""

    def __init__(self, out_root: Path, ttl_seconds: float = 6 * 3600) -> None:
        self.out_root = Path(out_root)
        self.ttl = ttl_seconds
        self._sessions: dict[str, WebSession] = {}

    def create(
        self,
        input_path: Path,
        target_column: str | None = None,
    ) -> WebSession:
        """Create a session from a single file path (loads it directly)."""
        return self.create_from_df(input_path, load(input_path), target_column)

    def create_from_df(
        self,
        input_path: Path,
        df: pd.DataFrame,
        target_column: str | None = None,
        doc_text: str = "",
        ingestion: dict[str, Any] | None = None,
        companions: list[Any] | None = None,
        hold_out_df: "pd.DataFrame | None" = None,
    ) -> WebSession:
        """Create a session from an already-loaded DataFrame."""
        from distill.state import _new_session_id

        sid = _new_session_id()
        ws = WebSession(
            sid, input_path, df, self.out_root, target_column,
            doc_text=doc_text, ingestion=ingestion, companions=companions,
            hold_out_df=hold_out_df,
        )
        self._sessions[sid] = ws
        self._evict_stale()
        return ws

    def get(self, session_id: str) -> WebSession:
        ws = self._sessions.get(session_id)
        if ws is None:
            raise KeyError(session_id)
        return ws

    def _evict_stale(self) -> None:
        now = time.time()
        stale = [
            sid for sid, ws in self._sessions.items()
            if now - ws.created_at > self.ttl
        ]
        for sid in stale:
            del self._sessions[sid]
