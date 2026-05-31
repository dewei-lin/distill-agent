"""Output artifact (c) — the CONSORT-style cleaning flowchart.

Renders a publication-ready flow diagram of the cleaning pipeline, in the
visual style of a CONSORT participant-flow diagram: a vertical chain of
stage boxes from the raw dataset to the clean dataset, with side
"exclusion" boxes wherever a stage removed rows or columns.

Drawn with matplotlib (pure pip, no system binary needed) and exported to
both SVG and PNG so it can drop straight into a paper, report, or slide.

The diagram is built from `state.row_counts` — the list of `StageRowCount`
entries the orchestrator records as each stage runs. If that list is
empty (a pipeline that never recorded stage IO), a minimal two-box
raw -> clean diagram is produced instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .types import Stage

if TYPE_CHECKING:
    from .state import PipelineState, Session


_STAGE_LABEL = {
    Stage.PROFILE: "Profiling pass",
    Stage.COMPANION: "Companion matching",
    Stage.FORMAT: "Format resolution",
    Stage.MISSINGNESS: "Missingness pass",
    Stage.OUTLIER: "Outlier detection",
    Stage.DUPLICATE: "Duplicate detection",
    Stage.BINNING: "Feature engineering",
    Stage.SCALING: "Feature engineering",
    Stage.ENCODING: "Feature engineering",
}

# Stages that never remove rows or columns — skip them from the flowchart
# unless they happen to record an unexpected removal.
_NON_MODIFYING = {Stage.PROFILE, Stage.FORMAT, Stage.BINNING, Stage.SCALING, Stage.ENCODING}

# Colors (hex) — kept here so the look is consistent and easy to retheme.
_C_RAW = "#185FA5"
_C_STAGE = "#E6F1FB"
_C_STAGE_EDGE = "#185FA5"
_C_CLEAN = "#0F6E56"
_C_EXCL = "#FAEEDA"
_C_EXCL_EDGE = "#C8862A"
_C_TEXT = "#1a1a1a"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_flowchart(
    state: "PipelineState",
    session: "Session",
) -> tuple[Path, Path]:
    """Render the cleaning flowchart to SVG + PNG.

    Returns (svg_path, png_path). Raises ImportError if matplotlib is not
    installed (the caller in `outputs.py` handles that gracefully).
    """
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    steps = _build_steps(state)
    n = len(steps)

    # Layout in data coordinates: x in [0, 10], y one unit per step.
    fig_h = max(3.5, 1.35 * n)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n)
    ax.axis("off")

    box_w, box_h = 3.8, 0.74
    main_cx = 2.9          # centre x of the main flow column
    excl_cx = 7.35         # centre x of the exclusion column
    excl_w = 4.2

    for i, step in enumerate(steps):
        cy = n - 0.5 - i   # first step at top

        # ----- main flow box -----
        if step["kind"] == "raw":
            face, edge, tcol = _C_RAW, _C_RAW, "white"
        elif step["kind"] == "clean":
            face, edge, tcol = _C_CLEAN, _C_CLEAN, "white"
        else:
            face, edge, tcol = _C_STAGE, _C_STAGE_EDGE, _C_TEXT

        ax.add_patch(
            FancyBboxPatch(
                (main_cx - box_w / 2, cy - box_h / 2),
                box_w, box_h,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=face, edgecolor=edge, linewidth=1.3,
            )
        )
        ax.text(
            main_cx, cy + 0.12, step["title"],
            ha="center", va="center", fontsize=10, fontweight="bold", color=tcol,
        )
        ax.text(
            main_cx, cy - 0.16,
            f"{step['rows']:,} rows  ×  {step['cols']} cols",
            ha="center", va="center", fontsize=8.5, color=tcol,
        )

        # ----- arrow to next box -----
        if i < n - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (main_cx, cy - box_h / 2),
                    (main_cx, cy - 1 + box_h / 2),
                    arrowstyle="-|>", mutation_scale=14,
                    linewidth=1.2, color="#5b6b7a",
                )
            )

        # ----- exclusion side box (dynamic height for multi-line breakdown) -----
        if step.get("excl_text"):
            n_lines = step.get("excl_lines", 1)
            excl_h = max(0.52, 0.30 * n_lines)
            ax.add_patch(
                FancyBboxPatch(
                    (excl_cx - excl_w / 2, cy - excl_h / 2),
                    excl_w, excl_h,
                    boxstyle="round,pad=0.02,rounding_size=0.05",
                    facecolor=_C_EXCL, edgecolor=_C_EXCL_EDGE, linewidth=1.0,
                )
            )
            ax.text(
                excl_cx, cy, step["excl_text"],
                ha="center", va="center", fontsize=7.2, color="#5a3d10",
                linespacing=1.4,
            )
            # connector from main box to exclusion box
            ax.add_patch(
                FancyArrowPatch(
                    (main_cx + box_w / 2, cy),
                    (excl_cx - excl_w / 2, cy),
                    arrowstyle="-|>", mutation_scale=11,
                    linewidth=1.0, color=_C_EXCL_EDGE,
                )
            )

    ax.text(
        5.0, n - 0.02,
        f"DataClean Agent — cleaning flow for {state.input_path.name}",
        ha="center", va="bottom", fontsize=9, style="italic", color="#5b6b7a",
    )

    fig.tight_layout(pad=0.6)
    svg_path = session.flowchart_svg
    png_path = session.flowchart_png
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


# ---------------------------------------------------------------------------
# Step construction
# ---------------------------------------------------------------------------


def _var_breakdown(
    state: "PipelineState", stage: "Stage"
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Return per-variable (rows removed, cols removed) lists for a stage.

    Reads `applied_info` stored on each resolved DecisionCard. Returns
    (row_items, col_items) where each item is (variable_name, count).
    """
    row_items: list[tuple[str, int]] = []
    col_items: list[tuple[str, int]] = []
    for card in state.log.by_stage(stage):
        info = card.metadata.get("applied_info", {})
        var = card.variable or "—"
        rows = info.get("rows_dropped", 0)
        cols = info.get("cols_dropped", 0)
        if rows and rows > 0:
            row_items.append((var, int(rows)))
        if cols and cols > 0:
            col_items.append((var, int(cols)))
    return row_items, col_items


def _build_steps(state: "PipelineState") -> list[dict[str, Any]]:
    """Turn the pipeline state into an ordered list of flow steps.

    Each step: {kind, title, rows, cols, excl_text, excl_lines}.
    kind is "raw" | "stage" | "clean".
    excl_lines is the line count of excl_text (used for dynamic box height).

    Non-modifying stages (profiling, format, FE transforms) are skipped
    entirely unless they unexpectedly changed the row/col count.
    """
    steps: list[dict[str, Any]] = [
        {
            "kind": "raw",
            "title": "Raw dataset",
            "rows": state.n_rows_in,
            "cols": state.n_cols_in,
            "excl_text": None,
            "excl_lines": 0,
        }
    ]

    for sc in state.row_counts:
        # Skip non-modifying stages that made no structural change.
        if sc.stage in _NON_MODIFYING and sc.rows_removed == 0 and sc.cols_removed == 0:
            continue

        excl_text: str | None = None
        excl_lines = 0

        bits = []
        if sc.rows_removed > 0:
            bits.append(f"−{sc.rows_removed:,} rows")
        if sc.cols_removed > 0:
            bits.append(f"−{sc.cols_removed} col(s)")

        if bits:
            excl_text = ",  ".join(bits)
            excl_lines = 1

            # Per-variable breakdown derived from the decision log.
            row_items, col_items = _var_breakdown(state, sc.stage)

            if row_items:
                # Merge: if same variable appears multiple times, sum.
                merged: dict[str, int] = {}
                for v, n in row_items:
                    merged[v] = merged.get(v, 0) + n
                sorted_items = sorted(merged.items(), key=lambda x: -x[1])
                parts = [f"{v}: {n:,}" for v, n in sorted_items[:5]]
                if len(sorted_items) > 5:
                    parts.append(f"+{len(sorted_items)-5} more")
                excl_text += "\n" + "  ".join(parts)
                excl_lines += 1

            if col_items:
                merged_c: dict[str, int] = {}
                for v, n in col_items:
                    merged_c[v] = merged_c.get(v, 0) + n
                parts_c = [f"{v}" for v, _ in sorted(merged_c.items(), key=lambda x: -x[1])[:4]]
                excl_text += "\ncols: " + ",  ".join(parts_c)
                excl_lines += 1

        steps.append(
            {
                "kind": "stage",
                "title": _STAGE_LABEL.get(sc.stage, sc.stage.value.title()),
                "rows": sc.rows_out,
                "cols": sc.cols_out,
                "excl_text": excl_text,
                "excl_lines": excl_lines,
            }
        )

    # Final "clean dataset" box.
    if state.row_counts:
        last = state.row_counts[-1]
        final_rows, final_cols = last.rows_out, last.cols_out
    else:
        final_rows, final_cols = len(state.df), len(state.df.columns)
    steps.append(
        {
            "kind": "clean",
            "title": "Clean dataset",
            "rows": final_rows,
            "cols": final_cols,
            "excl_text": None,
            "excl_lines": 0,
        }
    )
    return steps
