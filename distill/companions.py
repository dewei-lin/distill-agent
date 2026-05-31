"""Companion-file matching detection and resolution (Stage COMPANION).

Runs immediately after the profiling pass. Computes which subjects in the
structured data have a matching companion file, surfaces a PENDING
DecisionCard when coverage is incomplete, and applies the analyst's choice.

Match templates
---------------
A *match template* describes how to derive the companion filename stem from
a structured data row.  Two forms are accepted:

* Plain column name:  ``"subject_id"``
  The match key is ``str(row["subject_id"])``.

* Format string:  ``"{jurisdiction}_{period_id}"``
  Python ``str.format_map`` is applied to the row, e.g.
  ``"{jurisdiction}_{period_id}"`` on a row with
  ``jurisdiction="WV", period_id="uTjgI1Sv"`` yields ``"WV_uTjgI1Sv"``.

Skipped silently when no companions are registered or no match template is set.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .types import DecisionCard, DecisionStatus, Stage

if TYPE_CHECKING:
    import pandas as pd
    from .state import PipelineState


# ---------------------------------------------------------------------------
# Template helpers (public so sessions.py and script_gen can reuse them)
# ---------------------------------------------------------------------------


def get_template_columns(template: str) -> list[str]:
    """Return the list of column names referenced by *template*.

    For a plain column name returns ``[template]``.
    For a format string returns all ``{name}`` placeholders.
    """
    refs = re.findall(r"\{(\w+)\}", template)
    return refs if refs else [template]


def validate_template(template: str, available_columns: list[str]) -> list[str]:
    """Return column names referenced in *template* that are not in *available_columns*."""
    return [c for c in get_template_columns(template) if c not in available_columns]


def evaluate_template(template: str, row: "pd.Series") -> str:
    """Derive the companion match key for one row using *template*."""
    if "{" in template and "}" in template:
        return template.format_map({c: str(row.get(c, "")) for c in get_template_columns(template)})
    return str(row.get(template, ""))


def build_row_keys(df: "pd.DataFrame", template: str) -> "pd.Series":
    """Return a Series of per-row match keys derived from *template*."""
    if "{" in template and "}" in template:
        cols = get_template_columns(template)
        parts = [df[c].fillna("").astype(str) for c in cols]
        # Reconstruct the template by replacing {col} with the series values.
        result = parts[0].copy()
        result[:] = ""
        t = template
        for c in cols:
            t = t.replace(f"{{{c}}}", "__PLACEHOLDER__")
        segments = t.split("__PLACEHOLDER__")
        result = segments[0]
        for i, c in enumerate(cols):
            result = result + df[c].fillna("").astype(str) + (segments[i + 1] if i + 1 < len(segments) else "")
        return result
    return df[template].dropna().astype(str).reindex(df.index, fill_value="")


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------


def detect_companion_matching(state: "PipelineState") -> list[DecisionCard]:
    """Return a single DecisionCard describing companion coverage.

    Returns an empty list when companions or match_template are absent.
    """
    template = getattr(state, "companion_match_template", None)
    if not state.companions or not template:
        return []

    missing_cols = validate_template(template, list(state.df.columns))
    if missing_cols:
        raise ValueError(
            f"Match template {template!r} references columns not in dataset: {missing_cols}. "
            f"Available: {list(state.df.columns)}"
        )

    companion_ids: frozenset[str] = frozenset(c.subject_id for c in state.companions)
    row_keys: set[str] = set(build_row_keys(state.df, template).dropna())
    row_keys.discard("")

    matched = row_keys & companion_ids
    unmatched_subjects = sorted(row_keys - companion_ids)
    orphan_companions = sorted(companion_ids - row_keys)

    n_total = len(row_keys)
    n_matched = len(matched)
    n_unmatched = len(unmatched_subjects)

    if n_unmatched:
        pct = 100 * n_unmatched / n_total if n_total else 0
        issue = (
            f"{n_unmatched} of {n_total} subject(s) ({pct:.0f}%) "
            f"have no matching companion file."
        )
        rationale = (
            "Some subjects have no companion file. Keep every row, add a presence "
            "indicator (has_<type>) so downstream code can tell which rows lack a "
            "companion, or drop the unmatched rows if companion data is required "
            "for every record."
        )
        recommendation = "keep_rows"
        default_action = "keep_rows"
        alternatives = ["add_indicator", "drop_rows", "flag_caveat"]
        coverage = "incomplete"
    else:
        issue = f"All {n_total} subjects have a matching companion file."
        rationale = (
            "Coverage is complete. Companion file paths are reconstructed on demand "
            "by the generated script and kept out of the cleaned table by default. "
            "Include the file-path column only if you want it stored in the data."
        )
        recommendation = "exclude_path"
        default_action = "exclude_path"
        alternatives = ["include_path"]
        coverage = "complete"

    card = DecisionCard(
        card_id="companion_matching",
        stage=Stage.COMPANION,
        issue=issue,
        recommendation=recommendation,
        rationale=rationale,
        default_action=default_action,
        variable=template,
        alternatives=alternatives,
        metadata={
            "match_template": template,
            "template_columns": get_template_columns(template),
            "coverage": coverage,
            "n_total": n_total,
            "n_matched": n_matched,
            "n_unmatched": n_unmatched,
            "unmatched_subjects": unmatched_subjects[:50],
            "n_orphan_companions": len(orphan_companions),
            "orphan_companions": orphan_companions[:20],
            "_companion_ids": list(companion_ids),
        },
        status=DecisionStatus.PENDING,
    )
    return [card]


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def apply_decision(
    df: "pd.DataFrame", card: DecisionCard
) -> "tuple[pd.DataFrame, dict]":
    """Apply the resolved companion-matching decision.

    Signature matches the other stage apply functions so it can be registered
    in pipeline._STAGE_APPLY and called by _resolve_and_apply.
    """
    action = card.action_taken or card.default_action
    info: dict = {"action": action, "rows_dropped": 0}

    # Record output-shaping flags the script emitter reads when deciding which
    # companion columns survive into the cleaned tabular output.
    card.metadata["include_companion_path"] = action == "include_path"
    card.metadata["include_companion_indicator"] = action == "add_indicator"

    if action == "drop_rows":
        companion_ids = frozenset(card.metadata.get("_companion_ids", []))
        template = card.metadata.get("match_template", card.variable or "")
        if template and companion_ids:
            before = len(df)
            row_keys = build_row_keys(df, template)
            df = df[row_keys.isin(companion_ids)].reset_index(drop=True)
            info["rows_dropped"] = before - len(df)

    return df, info
