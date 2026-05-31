"""Dataset ingestion — multi-file loading + documentation-aware planning.

The web app lets an analyst upload more than one data file, plus optional
documentation (a codebook / data dictionary / README as PDF, Markdown or
plain text) and any number of non-tabular companion files (images, etc.).
This module turns that pile of uploads into one tidy DataFrame:

  1. classify every upload as data / documentation / companion;
  2. extract plain text from the documentation;
  3. ask an LLM to read that documentation and produce an *ingestion
     plan* — which file is the main table, how several files relate
     (single / stack / join), and which coded values mean "missing"
     (e.g. -99, 9999, "Unknown", "Refused");
  4. apply the plan to produce one DataFrame and a catalogue of the
     companion files.

Everything degrades gracefully. With no documentation, or no API key, a
deterministic heuristic planner is used instead; a single tabular file is
simply loaded the standard way. The core `distill` library is never
involved in this step, so the cleaning pipeline stays LLM-free.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Any

import pandas as pd

from distill.io import SUPPORTED_EXTENSIONS, load
from distill.types import DataCompanion

# Extensions we treat as readable documentation.
DOC_EXTENSIONS = frozenset({".pdf", ".md", ".markdown", ".txt", ".rst", ".docx"})

# MIME types inferred from file extension for companion registration.
_MIME_MAP: dict[str, str] = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".bmp": "image/bmp", ".tiff": "image/tiff",
    ".svg": "image/svg+xml", ".webp": "image/webp",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".mp4": "video/mp4",
    ".zip": "application/zip", ".gz": "application/gzip",
    ".pdf": "application/pdf",
}

# Model used for the ingestion plan — cheap and fast; structured output.
_INGEST_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

_MAX_DOC_CHARS = 16000  # documentation text handed to the LLM
_PREVIEW_ROWS = 5


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Everything the web session needs after ingestion."""

    df: pd.DataFrame
    primary_path: Path
    doc_text: str = ""
    report: dict[str, Any] = field(default_factory=dict)
    # Free-text context typed by the analyst at upload time.
    user_note: str = ""
    # Non-tabular files registered during ingestion.
    companions: list[DataCompanion] = field(default_factory=list)


# ---------------------------------------------------------------------------
# File classification
# ---------------------------------------------------------------------------


def classify_files(paths: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    """Split uploads into (tabular data, documentation, companion) lists."""
    tabular, docs, companions = [], [], []
    for p in paths:
        ext = p.suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            tabular.append(p)
        elif ext in DOC_EXTENSIONS:
            docs.append(p)
        else:
            companions.append(p)
    return tabular, docs, companions


# ---------------------------------------------------------------------------
# Documentation text extraction
# ---------------------------------------------------------------------------


def extract_doc_text(doc_paths: list[Path]) -> str:
    """Concatenate readable text from every documentation file."""
    chunks: list[str] = []
    for p in doc_paths:
        try:
            text = _extract_one(p)
        except Exception as e:  # never let a bad doc abort ingestion
            text = f"[Could not read {p.name}: {type(e).__name__}]"
        if text and text.strip():
            chunks.append(f"===== {p.name} =====\n{text.strip()}")
    return "\n\n".join(chunks)


def _extract_one(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".md", ".markdown", ".txt", ".rst"):
        return path.read_text(encoding="utf-8", errors="replace")
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return "[PDF documentation uploaded but pypdf is not installed]"
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        try:
            import docx  # python-docx
        except ImportError:
            return "[DOCX documentation uploaded but python-docx is not installed]"
        d = docx.Document(str(path))
        return "\n".join(par.text for par in d.paragraphs)
    return ""


# ---------------------------------------------------------------------------
# File previews (fed to the planner)
# ---------------------------------------------------------------------------


def _preview(path: Path, df: pd.DataFrame) -> dict[str, Any]:
    head = df.head(_PREVIEW_ROWS)
    return {
        "filename": path.name,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "sample_rows": head.astype(str).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Role-based proposal (new human-in-the-loop ingestion flow)
# ---------------------------------------------------------------------------

# Valid roles an analyst can assign to each uploaded file.
FILE_ROLES = frozenset({"primary_data", "target", "hold_out", "template", "companion"})

_INGEST_SYSTEM = """\
You are the ingestion planner inside Distill Agent, a data-cleaning tool.
You are given previews of uploaded data files, optional documentation, and
optional analyst notes.

Assign EXACTLY ONE role to every file, then describe how to combine them.
Respond with ONE JSON object only — no prose outside the JSON.

FILE ROLES:
  "primary_data"  — table(s) to be joined together and cleaned. Assign this
                    role to ALL files that will be horizontally merged by a
                    shared identifier (e.g. covariates.csv + dosing.csv that
                    share participant_id). If files have identical schemas,
                    they will be stacked (rows combined) instead of joined.
                    One or more files may be primary_data.
  "target"        — labels / outcomes to JOIN onto primary_data (when the
                    outcome file is distinct from the features).
  "hold_out"      — validation or test set with the same schema as
                    primary_data; cleaned separately, NOT concatenated.
  "template"      — submission template, README, or metadata — EXCLUDE.
  "companion"     — non-tabular supplementary file (image, audio, etc.).

RESPONSE SCHEMA:
{
  "files": [
    {
      "filename": "<exact filename as given>",
      "role": "primary_data|target|hold_out|template|companion"
    }
  ],
  "join_key": "<single column name, or a JSON list for compound keys: [\"col1\",\"col2\"]>",
  "key_map": {"<filename>": "<local col name if it differs from join_key>"} or {},
  "na_values": ["<values the docs say encode missing data, e.g. -99, 9999>"],
  "target_column": "<outcome column name, or null>",
  "notes": "<2-3 plain-language sentences addressed to the analyst>"
}

RULES:
- When multiple files share features linked by a common ID, give them ALL "primary_data".
- Files named sample_submission*, *template*, *readme* are almost always "template".
- A val/ or test/ directory with the same schema as train/ is "hold_out".
- Use "target" only when the outcome/label file is clearly separate from features.
- na_values: ONLY list values the documentation EXPLICITLY says encode missing data.
- notes: start with "I propose …", explain your reasoning briefly."""


def _llm_available() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return bool(key) and key.strip() not in {
        "", "your_anthropic_api_key_here", "sk-ant-xxx",
    }


def _llm_propose(
    previews: list[dict], doc_text: str, user_note: str = ""
) -> dict[str, Any]:
    """Ask the LLM to propose file roles. Raises on any failure."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    doc_block = doc_text[:_MAX_DOC_CHARS] if doc_text else "(no documentation uploaded)"
    note_block = f"\nANALYST NOTE:\n{user_note.strip()}" if user_note.strip() else ""
    user_msg = (
        f"DATA FILE PREVIEWS:\n{json.dumps(previews, default=str)[:8000]}\n\n"
        f"DOCUMENTATION:\n{doc_block}"
        f"{note_block}\n\n"
        "Return the ingestion proposal as a single JSON object."
    )
    resp = client.messages.create(
        model=_INGEST_MODEL,
        max_tokens=900,
        system=_INGEST_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(
        b.text for b in resp.content if getattr(b, "type", "") == "text"
    )
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM did not return a JSON object")
    return json.loads(text[start : end + 1])


# ---------------------------------------------------------------------------
# Proposal (returned to the frontend for human confirmation)
# ---------------------------------------------------------------------------


def _heuristic_roles(loaded: dict[Path, pd.DataFrame]) -> list[dict[str, Any]]:
    """Fallback role assignment when LLM is unavailable."""
    paths = list(loaded.keys())
    if not paths:
        return []

    # Largest file is primary; files with same columns as primary are hold-outs;
    # files with few columns that look like labels are targets; everything else single.
    primary = max(paths, key=lambda p: len(loaded[p]))
    primary_cols = set(str(c) for c in loaded[primary].columns)
    roles = []
    for p in paths:
        cols = set(str(c) for c in loaded[p].columns)
        name_lower = p.name.lower()
        if p == primary:
            role = "primary_data"
        elif any(kw in name_lower for kw in ("submission", "template", "sample_sub", "readme")):
            role = "template"
        elif cols == primary_cols:
            role = "hold_out"
        elif len(cols) <= 3 and cols < primary_cols | cols:
            role = "target"
        else:
            role = "primary_data"  # additional feature table to be joined
        roles.append({"filename": p.name, "role": role, "join_key": None,
                      "n_rows": len(loaded[p]), "n_cols": loaded[p].shape[1],
                      "columns": [str(c) for c in loaded[p].columns]})
    return roles


def propose_ingestion(
    loaded: dict[Path, pd.DataFrame],
    doc_text: str,
    user_note: str = "",
) -> dict[str, Any]:
    """Propose file roles for human review — does NOT merge anything yet.

    Returns a proposal dict::

        {
          "files": [
            {"filename": "covariates.csv", "role": "primary_data",
             "join_key": null, "n_rows": 4286, "n_cols": 15,
             "columns": [...]},
            {"filename": "dose_sys_train.csv", "role": "target",
             "join_key": "period_id", "n_rows": 4286, "n_cols": 2,
             "columns": [...]},
            ...
          ],
          "join_key": "period_id",
          "key_map": {},
          "na_values": [],
          "target_column": "dose_sys",
          "notes": "I propose ...",
          "used_llm": true
        }

    The frontend shows this as a decision card; the analyst confirms or
    overrides roles, then POSTs the confirmed list to /ingestion/confirm.
    """
    previews = [_preview(p, df) for p, df in loaded.items()]

    # Enrich previews with the same metadata we'll surface in the card.
    for prev, (p, df) in zip(previews, loaded.items()):
        prev["n_rows"] = len(df)
        prev["n_cols"] = df.shape[1]

    heuristic_roles = _heuristic_roles(loaded)

    if not _llm_available() or (not doc_text and not user_note and len(loaded) <= 1):
        return {
            "files": heuristic_roles,
            "join_key": None,
            "key_map": {},
            "na_values": [],
            "target_column": None,
            "notes": "",
            "used_llm": False,
        }

    try:
        llm = _llm_propose(previews, doc_text, user_note)
    except Exception as e:
        return {
            "files": heuristic_roles,
            "join_key": None,
            "key_map": {},
            "na_values": [],
            "target_column": None,
            "notes": "",
            "used_llm": False,
            "llm_error": f"{type(e).__name__}: {e}",
        }

    # Merge LLM roles with heuristic metadata (n_rows, n_cols, columns).
    meta = {r["filename"]: r for r in heuristic_roles}
    merged_files = []
    for f in llm.get("files", []):
        fname = f.get("filename", "")
        base = meta.get(fname) or meta.get(Path(fname).name) or {}
        merged_files.append({
            "filename": fname,
            "role": f.get("role", "primary_data") if f.get("role") in FILE_ROLES else "primary_data",
            "join_key": f.get("join_key"),
            "n_rows": base.get("n_rows"),
            "n_cols": base.get("n_cols"),
            "columns": base.get("columns", []),
        })

    # Ensure every loaded file appears (LLM might have missed some).
    covered = {f["filename"] for f in merged_files}
    covered_bases = {Path(n).name for n in covered}
    for r in heuristic_roles:
        if r["filename"] not in covered and r["filename"] not in covered_bases:
            merged_files.append(r)

    return {
        "files": merged_files,
        "join_key": llm.get("join_key"),
        "key_map": llm.get("key_map") or {},
        "na_values": [str(v) for v in (llm.get("na_values") or []) if str(v).strip()],
        "target_column": llm.get("target_column"),
        "notes": llm.get("notes", ""),
        "used_llm": True,
    }


# ---------------------------------------------------------------------------
# Apply confirmed roles → produce DataFrame
# ---------------------------------------------------------------------------


def _make_companion(path: Path) -> DataCompanion:
    ext = path.suffix.lower()
    mime = _MIME_MAP.get(ext, f"application/octet-stream")
    return DataCompanion(path=path, mime_type=mime)


def _apply_na_values(df: pd.DataFrame, na_values: list[str]) -> pd.DataFrame:
    """Replace any cell equal to a documented missing-code with NA."""
    norm = {str(v).strip().lower() for v in na_values if str(v).strip()}
    if not norm:
        return df
    out = df.copy()
    for col in out.columns:
        def _is_code(v: Any) -> bool:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return False
            return str(v).strip().lower() in norm

        mask = out[col].map(_is_code)
        if mask.any():
            out.loc[mask, col] = pd.NA
    return out


def _match_path(name: str | None, paths: list[Path]) -> Path | None:
    if not name:
        return None
    for p in paths:
        if p.name == name or p.name.lower() == str(name).lower():
            return p
    return None


def apply_confirmed_roles(
    confirmed: dict[str, Any],
    loaded: dict[Path, pd.DataFrame],
    *,
    companions: list[Path],
    load_errors: dict[Path, str],
) -> tuple[pd.DataFrame, Path, pd.DataFrame | None, dict[str, Any], list[DataCompanion]]:
    """Apply analyst-confirmed file roles to produce the training DataFrame.

    Returns ``(train_df, primary_path, hold_out_df, report, companion_objects)``.
    ``hold_out_df`` is ``None`` when no hold-out files were assigned.
    """
    # Build a role lookup: filename -> role (accept both full path and basename).
    role_map: dict[str, str] = {}
    join_key_map: dict[str, str | None] = {}
    for f in confirmed.get("files", []):
        name = f["filename"]
        role = f.get("role", "primary_data")
        role_map[name] = role
        role_map[Path(name).name] = role
        join_key_map[name] = f.get("join_key")
        join_key_map[Path(name).name] = f.get("join_key")

    def _role(p: Path) -> str:
        return role_map.get(p.name) or role_map.get(str(p)) or "primary_data"

    primary_paths = [p for p in loaded if _role(p) == "primary_data"]
    target_paths  = [p for p in loaded if _role(p) == "target"]
    hold_out_paths = [p for p in loaded if _role(p) == "hold_out"]
    extra_companions = [p for p in loaded if _role(p) in ("template", "companion")]

    if not primary_paths and loaded:
        primary_paths = [max(loaded, key=lambda p: len(loaded[p]))]

    primary = primary_paths[0]
    na_values = [str(v) for v in (confirmed.get("na_values") or []) if str(v).strip()]
    cleaned = {p: _apply_na_values(df, na_values) for p, df in loaded.items()}

    # ---- join / stack multiple primary_data files ----
    df = cleaned[primary]
    key_map: dict[str, str] = confirmed.get("key_map") or {}
    note_lines: list[str] = []

    # Normalise join_keys to a list (handles str, list[str], or None from LLM).
    raw_jk = confirmed.get("join_keys") or confirmed.get("join_key")
    if isinstance(raw_jk, list):
        explicit_keys: list[str] = [str(k).strip() for k in raw_jk if str(k).strip()]
    elif isinstance(raw_jk, str) and raw_jk.strip():
        # comma-separated compound key written as a string
        explicit_keys = [k.strip() for k in raw_jk.split(",") if k.strip()]
    else:
        explicit_keys = []

    def _join_right(df: pd.DataFrame, right: pd.DataFrame, right_name: str) -> pd.DataFrame:
        """Merge right onto df using explicit_keys, shared cols, col-append, or companion fallback."""
        local_renames = {v: k for k, v in key_map.items() if v in right.columns and k not in right.columns}
        if local_renames:
            right = right.rename(columns=local_renames)
        # Identical schema → vertical stack (row combining).
        if set(right.columns) == set(df.columns):
            result = pd.concat([df, right[df.columns]], ignore_index=True)
            note_lines.append(f"Stacked '{right_name}' onto '{primary.name}' (same schema) → {len(result):,} rows.")
            return result
        shared = [c for c in df.columns if c in right.columns]
        on_cols: list[str] = list(dict.fromkeys(explicit_keys + shared)) or shared
        if on_cols and all(c in right.columns for c in on_cols):
            result = df.merge(right, on=on_cols, how="left")
            note_lines.append(f"Joined '{right_name}' onto '{primary.name}' on ({', '.join(on_cols)}) → {len(result):,} rows.")
            return result
        if shared:
            result = df.merge(right, on=shared, how="left")
            note_lines.append(f"Joined '{right_name}' onto '{primary.name}' on shared ({', '.join(shared)}) → {len(result):,} rows.")
            return result
        if len(right) == len(df):
            for col in right.columns:
                if col not in df.columns:
                    df[col] = right[col].values
            note_lines.append(f"Appended columns from '{right_name}' (no shared key; same row count).")
            return df
        note_lines.append(
            f"Could not join '{right_name}': no shared columns and row counts differ "
            f"({len(right):,} vs {len(df):,}). Kept as companion."
        )
        return df

    for extra_p in primary_paths[1:]:
        df = _join_right(df, cleaned[extra_p], extra_p.name)

    # ---- join target files onto primary ----
    for tp in target_paths:
        df = _join_right(df, cleaned[tp], tp.name)

    # ---- hold-out DataFrame ----
    hold_out_df: pd.DataFrame | None = None
    if hold_out_paths:
        hold_out_frames = [cleaned[p] for p in hold_out_paths]
        hold_out_df = (
            pd.concat(hold_out_frames, ignore_index=True)
            if len(hold_out_frames) > 1
            else hold_out_frames[0]
        )
        hold_names = ", ".join(p.name for p in hold_out_paths)
        note_lines.append(
            f"Hold-out set registered ({hold_names}, "
            f"{len(hold_out_df):,} rows) — will be cleaned with the same pipeline."
        )

    if na_values:
        note_lines.append(
            "Treated as missing per documentation: "
            + ", ".join(sorted(set(na_values))) + "."
        )
    if extra_companions:
        note_lines.append(
            f"{len(extra_companions)} file(s) kept aside "
            "(template / companion): "
            + ", ".join(p.name for p in extra_companions) + "."
        )
    if load_errors:
        note_lines.append(
            f"{len(load_errors)} file(s) could not be read and were skipped."
        )

    plan_notes = (confirmed.get("notes") or "").strip()
    summary = plan_notes if plan_notes else " ".join(note_lines)
    if not summary:
        summary = f"Loaded '{primary.name}' ({len(df):,} rows, {df.shape[1]} columns)."

    companion_objects = [_make_companion(p) for p in companions + extra_companions]

    report = {
        "strategy": "confirmed",
        "summary": summary,
        "detail": note_lines,
        "files_loaded": [p.name for p in loaded],
        "na_values": sorted(set(na_values)),
        "key_map": key_map,
        "target_column": confirmed.get("target_column"),
        "used_llm": bool(confirmed.get("used_llm")),
        "hold_out_files": [p.name for p in hold_out_paths],
        "companions": [c.to_dict() for c in companion_objects],
        "skipped": [{"name": p.name, "error": err} for p, err in load_errors.items()],
    }
    return df, primary, hold_out_df, report, companion_objects


# kept for internal single-file fast-path
def apply_plan(
    plan: dict[str, Any],
    loaded: dict[Path, pd.DataFrame],
    *,
    companions: list[Path],
    load_errors: dict[Path, str],
) -> tuple[pd.DataFrame, Path, dict[str, Any], list[DataCompanion]]:
    """Turn an ingestion plan + loaded files into one DataFrame + a report."""
    if not loaded:
        detail = "; ".join(f"{p.name}: {e}" for p, e in load_errors.items())
        raise ValueError(
            "No readable data file was uploaded."
            + (f" ({detail})" if detail else "")
        )

    paths = list(loaded.keys())
    na_values = [str(v) for v in (plan.get("na_values") or [])]
    cleaned = {p: _apply_na_values(df, na_values) for p, df in loaded.items()}

    strategy = plan.get("strategy", "single")
    primary = _match_path(plan.get("primary_file"), paths) or max(
        paths, key=lambda p: len(loaded[p])
    )
    if len(cleaned) == 1:
        strategy = "single"

    note_lines: list[str] = []

    if strategy == "concat":
        frames = [
            cleaned[p].assign(source_file=p.name) for p in paths
        ]
        df = pd.concat(frames, ignore_index=True)
        note_lines.append(
            f"Stacked {len(paths)} files with matching columns into one "
            f"table of {len(df):,} rows (a 'source_file' column records "
            f"where each row came from)."
        )
    elif strategy == "join":
        key = plan.get("join_key")
        key_map: dict[str, str] = plan.get("key_map") or {}
        # Apply column renames so every file uses the canonical key name.
        renamed: dict[Path, pd.DataFrame] = {}
        for p, frame in cleaned.items():
            local_key = key_map.get(p.name)
            if local_key and local_key != key and local_key in frame.columns:
                renamed[p] = frame.rename(columns={local_key: key})
            else:
                renamed[p] = frame
        if key and all(key in d.columns for d in renamed.values()):
            ordered = [primary] + [p for p in paths if p != primary]

            def _smart_merge(left: pd.DataFrame, right_path: Path) -> pd.DataFrame:
                right = renamed[right_path]
                # Merge on ALL shared columns to avoid pandas suffix collisions
                # (e.g. two files that share both period_id and jurisdiction).
                shared = [c for c in left.columns if c in right.columns]
                on_cols: list[str] = list(dict.fromkeys([key] + shared))
                return left.merge(right, on=on_cols, how="outer")

            df = reduce(_smart_merge, ordered[1:], renamed[ordered[0]])
            on_desc = ", ".join(
                list(dict.fromkeys(
                    [key] + [c for c in renamed[ordered[0]].columns
                             if c in renamed[ordered[1]].columns]
                )) if len(ordered) > 1 else [key]
            )
            key_note = (
                f" (column names unified via key_map: {key_map})" if key_map else ""
            )
            note_lines.append(
                f"Joined {len(paths)} files on ({on_desc}){key_note} into one "
                f"table of {len(df):,} rows."
            )
        else:
            strategy = "single"
            df = cleaned[primary]
            note_lines.append(
                "Could not find a shared key to join the files, so I used "
                f"'{primary.name}' as the main table."
            )
    else:  # single
        df = cleaned[primary]
        extras = [p for p in paths if p != primary]
        if extras:
            companions = companions + extras
            note_lines.append(
                f"Used '{primary.name}' as the main table; "
                f"{len(extras)} other data file(s) were kept aside as "
                "companion files."
            )

    if na_values:
        note_lines.append(
            "Per the documentation, treated these as missing: "
            + ", ".join(sorted(set(na_values)))
            + "."
        )
    if load_errors:
        note_lines.append(
            f"{len(load_errors)} uploaded file(s) could not be read and "
            "were skipped."
        )

    plan_notes = (plan.get("notes") or "").strip()
    summary = plan_notes if plan_notes else " ".join(note_lines)
    if not summary:
        summary = f"Loaded '{primary.name}' ({len(df):,} rows, {df.shape[1]} columns)."

    companion_objects = [_make_companion(p) for p in companions]

    report = {
        "strategy": strategy,
        "summary": summary,
        "detail": note_lines,
        "files_loaded": [p.name for p in paths],
        "na_values": sorted(set(na_values)),
        "key_map": plan.get("key_map") or {},
        "target_column": plan.get("target_column"),
        "used_llm": bool(plan.get("used_llm")),
        "companions": [c.to_dict() for c in companion_objects],
        "skipped": [
            {"name": p.name, "error": err} for p, err in load_errors.items()
        ],
    }
    return df, primary, report, companion_objects


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def _expand_zips(paths: list[Path]) -> list[Path]:
    """Replace any .zip files with their extracted contents.

    Extracts into a sibling directory named ``<stem>_extracted/``.
    Skips path components containing ``..`` to prevent zip-slip attacks.
    Falls back to keeping the original zip path if extraction fails.
    """
    import zipfile

    result: list[Path] = []
    for p in paths:
        if p.suffix.lower() != ".zip":
            result.append(p)
            continue
        extract_dir = p.parent / (p.stem + "_extracted")
        extract_dir.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(p) as zf:
                safe_names = [
                    n for n in zf.namelist()
                    if not any(part == ".." for part in Path(n).parts)
                    and not Path(n).is_absolute()
                ]
                zf.extractall(extract_dir, members=safe_names)
                for name in safe_names:
                    extracted = extract_dir / name
                    if extracted.is_file():
                        result.append(extracted)
        except Exception:
            result.append(p)  # unreadable zip → pass through as companion
    return result


@dataclass
class IngestionProposal:
    """Returned by ``stage_ingest`` when multiple files require human review."""
    loaded: dict  # Path → pd.DataFrame (kept in session until confirmed)
    companions: list  # list[Path]
    load_errors: dict  # Path → str
    doc_text: str
    user_note: str
    proposal: dict[str, Any]  # the role proposal shown to the user


def stage_ingest(
    data_paths: list[Path],
    doc_paths: list[Path],
    user_note: str = "",
) -> tuple[IngestionProposal, bool]:
    """Classify, read docs, and propose file roles — but do NOT merge yet.

    Returns ``(proposal_obj, needs_confirmation)``.

    When ``needs_confirmation`` is False (single file, no ambiguity) the
    caller may immediately call ``ingest_confirmed`` with the auto-approved
    proposal.  When True, the frontend should show the ingestion decision card.
    """
    data_paths = _expand_zips(list(data_paths))
    doc_paths  = _expand_zips(list(doc_paths))

    tabular, docs_from_data, companions = classify_files(list(data_paths))
    doc_tab, docs_from_docs, doc_companions = classify_files(list(doc_paths))

    tabular    = tabular + doc_tab
    docs       = docs_from_data + docs_from_docs
    companions = companions + doc_companions

    doc_text = extract_doc_text(docs)

    loaded: dict[Path, pd.DataFrame] = {}
    load_errors: dict[Path, str] = {}
    for p in tabular:
        try:
            loaded[p] = load(p)
        except Exception as e:
            load_errors[p] = f"{type(e).__name__}: {e}"

    proposal = propose_ingestion(loaded, doc_text, user_note=user_note)
    proposal["doc_files"] = [p.name for p in docs]
    proposal["has_documentation"] = bool(doc_text.strip())

    needs_confirmation = len(loaded) > 1
    return IngestionProposal(
        loaded=loaded,
        companions=companions,
        load_errors=load_errors,
        doc_text=doc_text,
        user_note=user_note,
        proposal=proposal,
    ), needs_confirmation


def ingest_confirmed(
    ip: IngestionProposal,
    confirmed_files: list[dict[str, Any]] | None = None,
    confirmed: dict[str, Any] | None = None,
) -> IngestionResult:
    """Apply analyst-confirmed (or auto-approved) file roles and produce a DataFrame.

    ``confirmed_files`` is the analyst's final role list — a list of dicts
    with ``filename`` and ``role`` keys.  Pass ``None`` to accept the proposal.

    ``confirmed`` is the fully-assembled plan dict (from ``app.py``); when
    provided it takes precedence over rebuilding from the proposal.
    """
    if confirmed is None:
        confirmed = dict(ip.proposal)
    if confirmed_files is not None:
        confirmed["files"] = confirmed_files

    df, primary, hold_out_df, report, companion_objects = apply_confirmed_roles(
        confirmed,
        ip.loaded,
        companions=ip.companions,
        load_errors=ip.load_errors,
    )
    report["doc_files"] = ip.proposal.get("doc_files", [])
    report["has_documentation"] = ip.proposal.get("has_documentation", False)
    report["has_user_note"] = bool(ip.user_note.strip())

    result = IngestionResult(
        df=df,
        primary_path=primary,
        doc_text=ip.doc_text,
        report=report,
        user_note=ip.user_note,
        companions=companion_objects,
    )
    result.hold_out_df = hold_out_df  # type: ignore[attr-defined]
    return result


def ingest(
    data_paths: list[Path],
    doc_paths: list[Path],
    user_note: str = "",
) -> IngestionResult:
    """Legacy single-step entry point — auto-confirms the proposal.

    Used by the sample dataset path and any caller that doesn't need
    the human-in-the-loop card.  New code should use ``stage_ingest``
    + ``ingest_confirmed``.
    """
    ip, _ = stage_ingest(data_paths, doc_paths, user_note=user_note)
    return ingest_confirmed(ip)
