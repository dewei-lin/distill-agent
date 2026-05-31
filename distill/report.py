"""Output artifact (b) — the Cleaning Audit Report.

Renders a human-readable report in two formats:

  * `report.md`  — Markdown, always produced.
  * `report.pdf` — produced via reportlab when installed; skipped gracefully
    if not.

The report contains:
  * an executive summary (rows/cols in vs out, decision counts);
  * the CONSORT cleaning flowchart (PNG embedded);
  * a plain-language "What Changed" section per stage — no raw log dumps;
  * a variable-summary table (original vs final type, missingness);
  * a final-dataset profile with summary statistics per variable.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .profiler import profile as _profile
from .types import DecisionStatus, Stage

if TYPE_CHECKING:
    from .state import PipelineState, Session


_STAGE_DISPLAY = {
    Stage.PROFILE:    "Profiling",
    Stage.COMPANION:  "Companion matching",
    Stage.FORMAT:     "Format resolution",
    Stage.DUPLICATE:  "Duplicate detection",
    Stage.MISSINGNESS: "Missingness",
    Stage.OUTLIER:    "Outlier detection",
    Stage.BINNING:    "Feature engineering — binning",
    Stage.SCALING:    "Feature engineering — scaling",
    Stage.ENCODING:   "Feature engineering — encoding",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_report(
    state: "PipelineState",
    session: "Session",
) -> tuple[Path, Path | None]:
    """Write `report.md` (always) and `report.pdf` (if reportlab present)."""
    data = _collect(state, session)

    md_path = session.report_md
    md_path.write_text(_render_markdown(data), encoding="utf-8")

    pdf_path: Path | None = None
    try:
        pdf_path = _render_pdf(data, session.report_pdf)
    except ImportError:
        pass

    return md_path, pdf_path


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def _collect(state: "PipelineState", session: "Session") -> dict[str, Any]:
    """Gather everything both renderers need into one structured dict."""
    final_profiles = _profile(state.df)
    initial_profiles = state.profiles_initial or {}

    rename_map: dict[str, str] = {}
    for card in state.log:
        if card.card_id == "format_column_names":
            rename_map = dict(card.metadata.get("renames", {}))
            break

    entries = list(state.log)
    n_total = len(entries)
    n_auto = sum(
        1 for e in entries
        if e.status in (DecisionStatus.AGENT_AUTO, DecisionStatus.SKIPPED)
    )
    n_overrides = sum(1 for e in entries if e.status == DecisionStatus.OVERRIDDEN)
    n_human = sum(
        1 for e in entries
        if e.status in (
            DecisionStatus.CONFIRMED,
            DecisionStatus.OVERRIDDEN,
            DecisionStatus.AGENT_DEFAULT,
        )
    )

    true_missing: dict[str, float] = {}
    for card in entries:
        if card.stage == Stage.MISSINGNESS and card.variable and "missing_rate" in card.metadata:
            true_missing[card.variable] = float(card.metadata["missing_rate"])

    # Variable summary table
    var_rows: list[dict[str, Any]] = []
    for orig_name, p0 in initial_profiles.items():
        final_name = rename_map.get(orig_name, orig_name)
        pf = final_profiles.get(final_name)
        orig_missing = true_missing.get(final_name, p0.missing_rate)
        if pf is None:
            var_rows.append({
                "name": orig_name,
                "type": f"{p0.detected_type.value} → (dropped)",
                "missing": f"{orig_missing * 100:.1f}% → —",
                "change": "dropped",
            })
            continue
        type_str = (
            p0.detected_type.value
            if p0.detected_type == pf.detected_type
            else f"{p0.detected_type.value} → {pf.detected_type.value}"
        )
        miss_str = (
            f"{orig_missing * 100:.1f}%"
            if abs(orig_missing - pf.missing_rate) < 1e-9
            else f"{orig_missing * 100:.1f}% → {pf.missing_rate * 100:.1f}%"
        )
        change_notes: list[str] = []
        if final_name != orig_name:
            change_notes.append("renamed")
        for card in entries:
            if card.variable == final_name and card.action_taken:
                from .action_labels import label_for
                lbl = label_for(card.stage.value, card.action_taken)
                if lbl and lbl not in change_notes:
                    change_notes.append(lbl)
        var_rows.append({
            "name": f"{orig_name} → {final_name}" if final_name != orig_name else orig_name,
            "type": type_str,
            "missing": miss_str,
            "change": "; ".join(change_notes) if change_notes else "—",
        })

    # Also list new derived columns (e.g. age_bin, income_scaled) not in initial profiles
    initial_names = set(initial_profiles.keys()) | set(rename_map.values())
    for col_name, pf in final_profiles.items():
        if col_name not in initial_names:
            var_rows.append({
                "name": col_name,
                "type": pf.detected_type.value,
                "missing": f"{pf.missing_rate * 100:.1f}%",
                "change": "derived (new column)",
            })

    # Plain-language stage narratives (no raw log dumps)
    stage_narratives = _build_stage_narratives(state, entries, rename_map)

    # Final profile data for the "Dataset Profile" section, enriched with the
    # AI/doc-sourced variable descriptions captured at finalize.
    profile_data = _build_profile_data(
        final_profiles, getattr(state, "variable_descriptions", {}) or {}
    )

    # Flowchart paths
    flowchart_png = session.flowchart_png
    flowchart_svg = session.flowchart_svg

    return {
        "dataset": state.input_path.name,
        "session_id": state.session_id,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "rows_in": state.n_rows_in,
        "rows_out": len(state.df),
        "cols_in": state.n_cols_in,
        "cols_out": len(state.df.columns),
        "n_total": n_total,
        "n_auto": n_auto,
        "n_human": n_human,
        "n_overrides": n_overrides,
        "var_rows": var_rows,
        "stage_narratives": stage_narratives,
        "profile_data": profile_data,
        "flowchart_png": flowchart_png if flowchart_png.is_file() else None,
        "flowchart_svg": flowchart_svg if flowchart_svg.is_file() else None,
        "intake_metadata": state.intake_metadata or "",
    }


def _build_stage_narratives(
    state: "PipelineState",
    entries: list,
    rename_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Build one readable paragraph per stage that did something meaningful."""
    from .types import DecisionStatus

    narratives: list[dict[str, Any]] = []

    # ---- Format resolution ----
    fmt_cards = [e for e in entries if e.stage == Stage.FORMAT and
                 e.status not in (DecisionStatus.SKIPPED,)]
    if fmt_cards:
        renames = rename_map
        coercions = [e for e in fmt_cards if e.action_taken == "coerced"]
        placeholder = [e for e in fmt_cards if e.action_taken == "converted"]
        bits = []
        if renames:
            bits.append(f"{len(renames)} column name(s) standardised")
        if coercions:
            bits.append(f"{len(coercions)} column(s) re-typed to their correct data type")
        if placeholder:
            bits.append(f"{len(placeholder)} column(s) had placeholder tokens (e.g. '?') converted to missing")
        n_other = len(fmt_cards) - len(coercions) - len(placeholder)
        if n_other > 0:
            bits.append(f"{n_other} other mechanical fix(es) applied")
        narratives.append({
            "title": "Format resolution",
            "summary": "; ".join(bits) + "." if bits else "No changes.",
            "items": [],
        })

    # ---- Companion matching ----
    comp_card = next(
        (e for e in entries if e.stage == Stage.COMPANION
         and e.card_id == "companion_matching"),
        None,
    )
    if comp_card and comp_card.status != DecisionStatus.SKIPPED:
        from .action_labels import label_for
        meta = comp_card.metadata
        n_total = meta.get("n_total", 0)
        n_matched = meta.get("n_matched", 0)
        n_unmatched = meta.get("n_unmatched", 0)
        n_orphan = meta.get("n_orphan_companions", 0)
        action = comp_card.action_taken or comp_card.default_action
        lbl = label_for(Stage.COMPANION.value, action)
        unmatched_list = meta.get("unmatched_subjects", [])
        items = []
        if n_unmatched:
            preview = ", ".join(str(s) for s in unmatched_list[:10])
            if len(unmatched_list) > 10:
                preview += f", … (+{len(unmatched_list) - 10} more)"
            items.append(f"Subjects with no companion: {preview}")
        if n_orphan:
            items.append(
                f"{n_orphan} companion file(s) had no matching subject in the data"
            )
        narratives.append({
            "title": "Companion matching",
            "summary": (
                f"{n_matched} of {n_total} subject(s) matched "
                f"({n_unmatched} unmatched). Decision: {lbl}."
            ),
            "items": items,
        })

    # ---- Duplicate detection ----
    dup_cards = [e for e in entries if e.stage == Stage.DUPLICATE]
    if dup_cards:
        total_rows = sum(
            e.metadata.get("applied_info", {}).get("rows_dropped", 0)
            for e in dup_cards
        )
        action_taken = dup_cards[0].action_taken if dup_cards else "—"
        from .action_labels import label_for
        lbl = label_for(Stage.DUPLICATE.value, action_taken) if action_taken else "—"
        narratives.append({
            "title": "Duplicate detection",
            "summary": (
                f"{total_rows:,} duplicate row(s) removed."
                if total_rows > 0
                else f"No rows removed. Decision: {lbl}."
            ),
            "items": [],
        })

    # ---- Missingness ----
    miss_cards = [e for e in entries if e.stage == Stage.MISSINGNESS and e.variable]
    if miss_cards:
        items = []
        rows_dropped = cols_dropped = 0
        for card in miss_cards:
            info = card.metadata.get("applied_info", {})
            rd = info.get("rows_dropped", 0)
            cd = info.get("cols_dropped", 0)
            rows_dropped += rd
            cols_dropped += cd
            rate = card.metadata.get("missing_rate", 0)
            n_miss = card.metadata.get("missing_n", 0)
            from .action_labels import label_for
            lbl = label_for(Stage.MISSINGNESS.value, card.action_taken or card.default_action)
            detail = f"{card.variable}: {n_miss:,} missing ({rate * 100:.1f}%) — {lbl}"
            if card.status == DecisionStatus.OVERRIDDEN:
                detail += " _(analyst choice)_"
            items.append(detail)
        parts = []
        if rows_dropped > 0:
            parts.append(f"{rows_dropped:,} rows removed")
        if cols_dropped > 0:
            parts.append(f"{cols_dropped} column(s) dropped")
        imputed = [c for c in miss_cards
                   if (c.action_taken or c.default_action or "").endswith("impute")
                   or (c.action_taken or c.default_action or "") == "flag_missing_indicator"]
        if imputed:
            parts.append(f"{len(imputed)} variable(s) imputed or flagged")
        narratives.append({
            "title": "Missingness",
            "summary": ("; ".join(parts) + ".") if parts else "No changes.",
            "items": items,
        })

    # ---- Outlier detection ----
    out_cards = [e for e in entries if e.stage == Stage.OUTLIER and e.variable]
    if out_cards:
        items = []
        rows_dropped = 0
        for card in out_cards:
            info = card.metadata.get("applied_info", {})
            rd = info.get("rows_dropped", 0)
            rows_dropped += rd
            n_flag = card.metadata.get("n_flagged_total", 0)
            from .action_labels import label_for
            lbl = label_for(Stage.OUTLIER.value, card.action_taken or card.default_action)
            detail = f"{card.variable}: {n_flag:,} value(s) flagged — {lbl}"
            if card.status == DecisionStatus.OVERRIDDEN:
                detail += " _(analyst choice)_"
            items.append(detail)
        parts = []
        if rows_dropped > 0:
            parts.append(f"{rows_dropped:,} rows removed")
        changed = [c for c in out_cards
                   if (c.action_taken or c.default_action) not in ("leave_as_is", "inspect")]
        if changed:
            parts.append(f"{len(changed)} variable(s) modified")
        narratives.append({
            "title": "Outlier detection",
            "summary": ("; ".join(parts) + ".") if parts else "No flagged outliers acted on.",
            "items": items,
        })

    # ---- Feature engineering (binning + scaling + encoding combined) ----
    fe_stages = {Stage.BINNING, Stage.SCALING, Stage.ENCODING}
    fe_cards = [e for e in entries if e.stage in fe_stages and e.variable]
    if fe_cards:
        items = []
        for card in fe_cards:
            action = card.action_taken or card.default_action
            if action in ("leave_as_is", "skip", "proceed"):
                continue
            info = card.metadata.get("applied_info", {})
            new_col = info.get("new_col", "")
            from .action_labels import label_for
            lbl = label_for(card.stage.value, action)
            detail = f"{card.variable}: {lbl}"
            if new_col:
                detail += f" → `{new_col}`"
            elif card.stage == Stage.ENCODING and action == "datetime_decompose":
                detail += " (new columns created)"
            if card.status == DecisionStatus.OVERRIDDEN:
                detail += " _(analyst choice)_"
            items.append(detail)
        if items:
            narratives.append({
                "title": "Feature engineering",
                "summary": f"{len(items)} transformation(s) applied.",
                "items": items,
            })

    return narratives


def _build_profile_data(
    profiles: dict,
    descriptions: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Convert VariableProfile objects into render-ready dicts."""
    descriptions = descriptions or {}
    rows = []
    for name, p in profiles.items():
        desc = descriptions.get(name) or {}
        entry: dict[str, Any] = {
            "name": name,
            "type": p.detected_type.value,
            "n_total": p.n_total,
            "n_missing": p.n_missing,
            "missing_rate": p.missing_rate,
            "n_unique": p.n_unique,
            "description": desc.get("text"),
            "description_source": desc.get("source"),
        }
        if p.numeric:
            entry.update({
                "kind": "numeric",
                "mean": p.numeric.mean,
                "median": p.numeric.median,
                "std": p.numeric.std,
                "min": p.numeric.min,
                "max": p.numeric.max,
                "q25": p.numeric.q25,
                "q75": p.numeric.q75,
            })
        elif p.categorical:
            entry.update({
                "kind": "categorical",
                "top_values": p.categorical.top_values[:5],
            })
        else:
            entry["kind"] = "other"
        rows.append(entry)
    return rows


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def _render_markdown(d: dict[str, Any]) -> str:
    lines: list[str] = []
    W = lines.append

    W("# Cleaning Audit Report")
    W("")
    W(f"**Dataset:** {d['dataset']}  ")
    W(f"**Session:** {d['session_id']}  ")
    W(f"**Generated:** {d['generated']}  ")
    W("")

    if d.get("intake_metadata"):
        W(f"> {d['intake_metadata']}")
        W("")

    W("---")
    W("")

    # --- Executive summary ---
    W("## Summary")
    W("")
    W("| Metric | Value |")
    W("| --- | --- |")
    W(f"| Rows (in → out) | {d['rows_in']:,} → {d['rows_out']:,} |")
    W(f"| Columns (in → out) | {d['cols_in']} → {d['cols_out']} |")
    W(f"| Total decisions logged | {d['n_total']:,} |")
    W(f"| Automated (mechanical) | {d['n_auto']:,} |")
    W(f"| Human-confirmed / default | {d['n_human']:,} |")
    W(f"| Analyst overrides | {d['n_overrides']:,} |")
    W("")

    # --- Flowchart ---
    if d.get("flowchart_png"):
        W("## Cleaning Flowchart")
        W("")
        W("![Cleaning Flowchart](flowchart.png)")
        W("")

    # --- What Changed ---
    if d.get("stage_narratives"):
        W("## What Changed")
        W("")
        for stage in d["stage_narratives"]:
            W(f"### {stage['title']}")
            W("")
            W(stage["summary"])
            if stage.get("items"):
                W("")
                for item in stage["items"]:
                    W(f"- {item}")
            W("")

    # --- Variable summary table ---
    W("## Variable Summary")
    W("")
    W("| Variable | Type | Missingness | What happened |")
    W("| --- | --- | --- | --- |")
    for r in d["var_rows"]:
        W(f"| {r['name']} | {r['type']} | {r['missing']} | {r['change']} |")
    W("")

    # --- Final dataset profile (no decision log — just clean stats) ---
    W("## Final Dataset Profile")
    W("")
    W("Summary statistics for every variable in the cleaned dataset.")
    W("")
    for p in d["profile_data"]:
        W(f"### `{p['name']}`")
        W("")
        if p.get("description"):
            tag = " _(AI-inferred)_" if p.get("description_source") == "ai_inferred" else ""
            W(f"_{p['description']}_{tag}")
            W("")
        miss_pct = p["missing_rate"] * 100
        W(f"**Type:** {p['type']}  |  "
          f"**Missing:** {miss_pct:.1f}% ({p['n_missing']:,} of {p['n_total']:,})  |  "
          f"**Unique values:** {p['n_unique']:,}")
        W("")
        if p["kind"] == "numeric":
            W("| Mean | Median | Std | Min | Max | Q25 | Q75 |")
            W("| --- | --- | --- | --- | --- | --- | --- |")
            W(f"| {_f(p['mean'])} | {_f(p['median'])} | {_f(p['std'])} "
              f"| {_f(p['min'])} | {_f(p['max'])} | {_f(p['q25'])} | {_f(p['q75'])} |")
        elif p["kind"] == "categorical" and p.get("top_values"):
            W("| Value | Count |")
            W("| --- | --- |")
            for val, cnt in p["top_values"]:
                W(f"| {val} | {cnt:,} |")
        W("")

    # --- Methods narrative ---
    W("## Methods")
    W("")
    W(_methods_narrative(d))
    W("")
    W("---")
    W("")
    W("*Generated by DataClean Agent. Every transformation above is "
      "reproducible from the accompanying `clean_script.py`.*")
    return "\n".join(lines) + "\n"


def _f(v: float | None) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


def _methods_narrative(d: dict[str, Any]) -> str:
    rows_delta = d["rows_in"] - d["rows_out"]
    cols_delta = d["cols_in"] - d["cols_out"]
    parts = [
        f"The dataset `{d['dataset']}` entered the pipeline with "
        f"{d['rows_in']:,} rows and {d['cols_in']} variables.",
    ]
    if rows_delta > 0:
        parts.append(f" {rows_delta:,} rows were removed during cleaning")
    elif rows_delta < 0:
        parts.append(f" {-rows_delta:,} rows were added")
    else:
        parts.append(" No rows were removed")
    if cols_delta > 0:
        parts.append(f", and {cols_delta} variable(s) were dropped")
    parts.append(
        f", leaving {d['rows_out']:,} rows and {d['cols_out']} variables."
    )
    parts.append(
        f" Of {d['n_total']:,} logged decisions, {d['n_auto']:,} were applied automatically"
        f" and {d['n_human']:,} required a judgment call."
    )
    if d["n_overrides"]:
        parts.append(f" The analyst overrode the agent's recommendation {d['n_overrides']:,} time(s).")
    parts.append(
        " Every decision — automated or human — is captured in the accompanying"
        " `decisions.json`, and `clean_script.py` replays the entire process"
        " from the raw input."
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# PDF renderer (reportlab)
# ---------------------------------------------------------------------------


def _render_pdf(d: dict[str, Any], path: Path) -> Path:
    """Render the report to PDF using reportlab. Raises ImportError if
    reportlab is not installed."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]
    body = ParagraphStyle(
        "body", parent=styles["Normal"], fontSize=9, leading=13, alignment=TA_LEFT
    )
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.grey)
    mono = ParagraphStyle("mono", parent=body, fontSize=8, fontName="Courier")
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=6)

    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        title="Cleaning Audit Report",
    )
    flow: list[Any] = []

    # Header
    flow.append(Paragraph("Cleaning Audit Report", h1))
    meta = f"Dataset: {d['dataset']}  ·  Session: {d['session_id']}  ·  {d['generated']}"
    flow.append(Paragraph(meta, small))
    if d.get("intake_metadata"):
        flow.append(Spacer(1, 4))
        flow.append(Paragraph(d["intake_metadata"], small))
    flow.append(Spacer(1, 12))

    # Summary table
    flow.append(Paragraph("Summary", h2))
    summary_rows = [
        ["Rows (in → out)", f"{d['rows_in']:,} → {d['rows_out']:,}"],
        ["Columns (in → out)", f"{d['cols_in']} → {d['cols_out']}"],
        ["Total decisions logged", f"{d['n_total']:,}"],
        ["Automated (mechanical)", f"{d['n_auto']:,}"],
        ["Human-confirmed / default", f"{d['n_human']:,}"],
        ["Analyst overrides", f"{d['n_overrides']:,}"],
    ]
    t = Table(summary_rows, colWidths=[2.6 * inch, 3.4 * inch])
    t.setStyle(_grid_style(colors))
    flow.append(t)
    flow.append(Spacer(1, 14))

    # Flowchart
    if d.get("flowchart_png") and Path(d["flowchart_png"]).is_file():
        flow.append(Paragraph("Cleaning Flowchart", h2))
        flow.append(Spacer(1, 6))
        try:
            img = Image(str(d["flowchart_png"]))
            avail_w = 6.4 * inch
            scale = avail_w / img.imageWidth
            img.drawWidth = avail_w
            img.drawHeight = img.imageHeight * scale
            flow.append(img)
        except Exception:
            flow.append(Paragraph("[Flowchart image unavailable]", small))
        flow.append(Spacer(1, 14))

    # What Changed
    if d.get("stage_narratives"):
        flow.append(Paragraph("What Changed", h2))
        flow.append(Spacer(1, 4))
        for stage in d["stage_narratives"]:
            flow.append(Paragraph(stage["title"], h3))
            flow.append(Paragraph(stage["summary"], body))
            for item in stage.get("items", []):
                flow.append(Paragraph(f"• {item}", bullet))
            flow.append(Spacer(1, 6))
        flow.append(Spacer(1, 4))

    # Variable summary table
    flow.append(Paragraph("Variable Summary", h2))
    var_data = [["Variable", "Type", "Missingness", "What happened"]]
    for r in d["var_rows"]:
        var_data.append([
            Paragraph(str(r["name"]), small),
            Paragraph(str(r["type"]), small),
            Paragraph(str(r["missing"]), small),
            Paragraph(str(r["change"]), small),
        ])
    vt = Table(var_data, colWidths=[1.9 * inch, 1.5 * inch, 1.3 * inch, 1.5 * inch])
    vt.setStyle(_grid_style(colors, header=True))
    flow.append(vt)
    flow.append(Spacer(1, 14))

    # Final Dataset Profile
    flow.append(Paragraph("Final Dataset Profile", h2))
    flow.append(Spacer(1, 4))
    desc_style = ParagraphStyle(
        "desc", parent=body, fontSize=8.5, textColor=colors.HexColor("#333333"),
        leading=12, spaceAfter=2,
    )
    for p in d["profile_data"]:
        flow.append(Paragraph(p["name"], h3))
        if p.get("description"):
            import html as _html
            tag = " (AI-inferred)" if p.get("description_source") == "ai_inferred" else ""
            flow.append(Paragraph(
                f"<i>{_html.escape(str(p['description']))}</i>{tag}", desc_style
            ))
        miss_pct = p["missing_rate"] * 100
        meta_line = (
            f"Type: {p['type']}  ·  "
            f"Missing: {miss_pct:.1f}% ({p['n_missing']:,}/{p['n_total']:,})  ·  "
            f"Unique: {p['n_unique']:,}"
        )
        flow.append(Paragraph(meta_line, small))
        if p["kind"] == "numeric":
            num_data = [
                ["Mean", "Median", "Std", "Min", "Max", "Q25", "Q75"],
                [
                    _f(p.get("mean")), _f(p.get("median")), _f(p.get("std")),
                    _f(p.get("min")),  _f(p.get("max")),
                    _f(p.get("q25")),  _f(p.get("q75")),
                ],
            ]
            nt = Table(num_data, colWidths=[0.86 * inch] * 7)
            nt.setStyle(_grid_style(colors, header=True))
            flow.append(Spacer(1, 3))
            flow.append(nt)
        elif p["kind"] == "categorical" and p.get("top_values"):
            cat_data = [["Value", "Count"]] + [
                [str(v), f"{c:,}"] for v, c in p["top_values"]
            ]
            ct = Table(cat_data, colWidths=[3.2 * inch, 1.0 * inch])
            ct.setStyle(_grid_style(colors, header=True))
            flow.append(Spacer(1, 3))
            flow.append(ct)
        flow.append(Spacer(1, 8))

    # Methods
    flow.append(Paragraph("Methods", h2))
    flow.append(Paragraph(_methods_narrative(d), body))
    flow.append(Spacer(1, 10))
    flow.append(Paragraph(
        "Generated by DataClean Agent. Every transformation is reproducible "
        "from the accompanying clean_script.py.", small))

    doc.build(flow)
    return path


def _grid_style(colors, header: bool = False):
    from reportlab.platypus import TableStyle
    cmds = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.white, colors.HexColor("#f5f7fa")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#185FA5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    return TableStyle(cmds)
