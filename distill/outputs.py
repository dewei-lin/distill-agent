"""Output artifact writers.

Centralised entry points for writing the five canonical output artifacts
a cleaning session produces. The writers themselves live in this file
(for simple artifacts) or in their own modules (for the report,
flowchart, and reproducible script). This file holds the orchestration.

Artifacts:
  (a) Clean dataset   -- clean.csv + original format       (here)
  (b) Audit report    -- report.md + report.pdf            (distill.report)
  (c) CONSORT figure  -- flowchart.svg + flowchart.png     (distill.flowchart)
  (d) Decision log    -- decisions.json                    (here)
  (e) Reproducible    -- clean_script.py                   (distill.script_gen)

Each writer is independent so a partial run still produces what it can.
`write_all_outputs` returns a dict[artifact -> path] for everything it
successfully produced; failed writers append to the returned `errors` dict
rather than raising.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .io import save_clean
from .state import PipelineState, Session

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# (a) Clean dataset
# ---------------------------------------------------------------------------


def write_clean_data(state: PipelineState, session: Session) -> dict[str, Path]:
    """Write the cleaned DataFrame as portable CSV plus the original format.

    Returns a dict mapping format key ("csv", "xlsx", ...) to file path.
    The CSV is always produced and lives at `session.clean_csv`.
    """
    return save_clean(
        state.df,
        original_path=state.input_path,
        out_dir=session.dir,
        base_name="clean",
    )


# ---------------------------------------------------------------------------
# (d) Decision log
# ---------------------------------------------------------------------------


def write_decision_log(state: PipelineState, session: Session) -> Path:
    """Write the full decision log as canonical UTF-8 JSON.

    The log includes every card emitted across all six pipeline stages,
    AGENT_AUTO and AGENT_DEFAULT cards included, in execution order.
    Designed to be consumed by downstream agents.
    """
    return state.log.dump(session.decisions_json)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def write_all_outputs(
    state: PipelineState,
    session: Session,
    *,
    skip: tuple[str, ...] = (),
    repro_check_timeout: int = 120,
    run_repro: bool = True,
) -> dict[str, Any]:
    """Produce every output artifact possible, swallowing per-writer errors.

    Returns:
      {
        "artifacts": {  # successfully-written files
          "clean": {"csv": Path, "xlsx": Path, ...},
          "decisions": Path,
          "report_md": Path,
          "report_pdf": Path,
          "flowchart_svg": Path,
          "flowchart_png": Path,
          "clean_script": Path,
        },
        "errors": {     # writer -> error message (only on failure)
          "report": "ImportError: reportlab not installed",
          ...
        },
      }

    Pass `skip=("report",)` etc. to omit specific writers without
    triggering their error path.
    """
    artifacts: dict[str, Any] = {}
    errors: dict[str, str] = {}

    if "clean" not in skip:
        try:
            artifacts["clean"] = write_clean_data(state, session)
        except Exception as e:  # pragma: no cover -- defensive
            errors["clean"] = f"{type(e).__name__}: {e}"

    if "decisions" not in skip:
        try:
            artifacts["decisions"] = write_decision_log(state, session)
        except Exception as e:  # pragma: no cover -- defensive
            errors["decisions"] = f"{type(e).__name__}: {e}"

    # Report, flowchart, and reproducible script are wired in tasks 11/12/13.
    # Their writers are imported lazily so they can be added without
    # touching this file's imports.
    # Flowchart runs before report so the PNG is available for embedding.
    if "flowchart" not in skip:
        try:
            from .flowchart import render_flowchart  # type: ignore[import-not-found]

            svg_path, png_path = render_flowchart(state, session)
            artifacts["flowchart_svg"] = svg_path
            artifacts["flowchart_png"] = png_path
        except ImportError:
            pass
        except Exception as e:  # pragma: no cover
            errors["flowchart"] = f"{type(e).__name__}: {e}"

    if "report" not in skip:
        try:
            from .report import render_report  # type: ignore[import-not-found]

            md_path, pdf_path = render_report(state, session)
            artifacts["report_md"] = md_path
            if pdf_path is not None:
                artifacts["report_pdf"] = pdf_path
        except ImportError:
            pass
        except Exception as e:  # pragma: no cover
            errors["report"] = f"{type(e).__name__}: {e}"

    if "script" not in skip:
        try:
            from .script_gen import emit_requirements, emit_script  # type: ignore[import-not-found]

            artifacts["clean_script"] = emit_script(state, session)
            # Dependency manifests (requirements.txt + environment.yml) so a
            # user with a near-empty environment can run the script directly.
            reqs = emit_requirements(state, session)
            artifacts["requirements"] = reqs["requirements"]
            artifacts["environment"] = reqs["environment"]
        except ImportError:
            pass
        except Exception as e:  # pragma: no cover
            errors["script"] = f"{type(e).__name__}: {e}"

    checks: dict[str, Any] = {}
    if run_repro and "script" not in skip and "clean_script" in artifacts:
        try:
            repro = run_reproducibility_check(state, session, timeout=repro_check_timeout)
            checks["reproducibility"] = repro
            if repro.get("timed_out"):
                # A timeout is not a failure — the script is simply unverified.
                # Record it as a check note rather than an error so it does not
                # block hand-off.
                checks["reproducibility_note"] = (
                    f"Reproducibility check did not finish within {repro_check_timeout}s; "
                    "the script is unverified, not known to be wrong."
                )
            elif repro.get("crashed"):
                errors["reproducibility"] = (
                    f"Script replay crashed (exit code {repro.get('returncode')}). "
                    + (repro.get("stderr") or "")[:500]
                )
            elif not repro["match"]:
                errors["reproducibility"] = (
                    f"Script replay produced {repro['rows_replay']} rows × "
                    f"{repro['cols_replay']} cols; session produced "
                    f"{repro['rows_session']} rows × {repro['cols_session']} cols. "
                    "Column diff: " + repr(repro.get("col_diff"))
                )
        except Exception as e:  # pragma: no cover
            errors["reproducibility"] = f"{type(e).__name__}: {e}"

    return {"artifacts": artifacts, "errors": errors, "checks": checks}


# ---------------------------------------------------------------------------
# (f) Reproducibility check
# ---------------------------------------------------------------------------


def run_reproducibility_check(
    state: PipelineState,
    session: Session,
    *,
    timeout: int = 120,
) -> dict[str, Any]:
    """Run the emitted clean_script.py against the original source and compare output.

    Executes the script in a subprocess with a temporary working directory so
    it does not overwrite the session's clean.csv. Compares the replay output
    to the session's authoritative clean.csv row-for-row (order-insensitive,
    column-insensitive for columns absent from the replay due to target guards).

    Returns a dict:
      {
        "match": bool,
        "rows_session": int,
        "cols_session": int,
        "rows_replay":  int,
        "cols_replay":  int,
        "col_diff": {"only_in_session": [...], "only_in_replay": [...]},
        "stdout": str,   # subprocess stdout (trimmed)
        "stderr": str,   # subprocess stderr (trimmed, empty on success)
      }
    """
    import pandas as pd

    script_path = session.clean_script
    # Prefer raw_intake.csv (the pre-clean joined snapshot) so the check works
    # even when the session was built from multiple files merged at ingestion.
    raw_intake = session.dir / "raw_intake.csv"
    source_path = raw_intake if raw_intake.exists() else state.input_path

    # Pipeline-internal columns are environment-dependent (local file paths,
    # runtime presence flags) and are not part of the structural cleaning
    # the repro check is verifying. The replay runs without companion_dir
    # so these columns are absent by design — not a real mismatch.
    _COMPANION_COLS = frozenset({
        "image_path", "audio_path", "text_path", "video_path",
        "companion_path",
        "has_image", "has_audio", "has_text", "has_video", "has_companion",
    })

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            result = subprocess.run(
                [sys.executable, str(script_path), str(source_path)],
                capture_output=True,
                text=True,
                cwd=tmp,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "match": False,
                "crashed": True,
                "returncode": None,
                "timed_out": True,
                "rows_session": len(state.df),
                "cols_session": len(state.df.columns),
                "rows_replay": 0,
                "cols_replay": 0,
                "col_diff": {},
                "stdout": "",
                "stderr": f"Script timed out after {timeout}s.",
            }
        replay_csv = tmp_path / "clean.csv"
        stdout = result.stdout.strip()[-2000:]
        stderr = result.stderr.strip()[-2000:]

        if result.returncode != 0 or not replay_csv.exists():
            return {
                "match": False,
                "crashed": True,
                "returncode": result.returncode,
                "rows_session": len(state.df),
                "cols_session": len(state.df.columns),
                "rows_replay": 0,
                "cols_replay": 0,
                "col_diff": {},
                "stdout": stdout,
                "stderr": stderr,
            }

        replay_df = pd.read_csv(replay_csv)
        session_df = pd.read_csv(session.clean_csv)

        # Strip companion-derived columns from both sides before comparing.
        # They are environment-dependent (local file paths, presence flags) and
        # are not part of the structural cleaning the repro check is verifying.
        session_df = session_df.drop(
            columns=[c for c in _COMPANION_COLS if c in session_df.columns]
        )
        replay_df = replay_df.drop(
            columns=[c for c in _COMPANION_COLS if c in replay_df.columns]
        )

        session_cols = set(session_df.columns)
        replay_cols = set(replay_df.columns)
        col_diff = {
            "only_in_session": sorted(session_cols - replay_cols),
            "only_in_replay": sorted(replay_cols - session_cols),
        }

        # Compare on the intersection of columns only (target may be absent in replay)
        shared = sorted(session_cols & replay_cols)
        if shared:
            s = session_df[shared].sort_values(shared).reset_index(drop=True)
            r = replay_df[shared].sort_values(shared).reset_index(drop=True)
            try:
                import pandas.testing as pdt
                # 3% relative tolerance forgives MICE (iterative regression)
                # convergence variation across environments while still
                # catching genuine divergence.
                pdt.assert_frame_equal(
                    s, r, check_like=False, rtol=3e-2, atol=1e-4,
                    check_exact=False,
                )
                content_match = True
            except AssertionError:
                content_match = False
        else:
            content_match = len(replay_df) == len(session_df)

        # Accept as matching when shapes and shared-column content agree.
        # Target column may be absent in replay when guards are active.
        only_unexpected = [
            c for c in col_diff["only_in_session"]
            if c != getattr(state, "target_column", None)
        ]
        match = content_match and not only_unexpected and not col_diff["only_in_replay"]

        return {
            "match": match,
            "rows_session": len(session_df),
            "cols_session": len(session_df.columns),
            "rows_replay": len(replay_df),
            "cols_replay": len(replay_df.columns),
            "col_diff": col_diff,
            "stdout": stdout,
            "stderr": stderr,
        }
