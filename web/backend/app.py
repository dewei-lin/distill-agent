"""FastAPI backend for the Distill Agent web demo.

A thin HTTP layer over `sessions.WebSession`. The frontend orchestrates the
pipeline as a sequence of plain request/response calls:

    POST /api/upload                       -> create session, run profiling
    POST /api/sample                       -> create session from bundled data
    POST /api/session/{sid}/companion      -> companion matching stage (between profiling and format)
    POST /api/session/{sid}/companion_matching/resolve -> apply companion decision
    POST /api/session/{sid}/format         -> autonomous format resolution
    POST /api/session/{sid}/detect/{stage} -> decision cards for a stage (duplicate/missingness/outlier)
    POST /api/session/{sid}/resolve/{stage}-> apply the analyst's answers
    POST /api/session/{sid}/finalize       -> write the output artifacts
    GET  /api/session/{sid}/variable/{col} -> inspector data (histogram)
    GET  /api/session/{sid}/artifact/{f}   -> download one artifact
    GET  /api/session/{sid}/download_all   -> download every artifact (zip)
    POST /api/session/{sid}/chat           -> optional Anthropic-powered chat (non-streaming)
    POST /api/session/{sid}/chat/stream    -> SSE streaming chat (preferred)

Run it:
    pip install -e ".[web]"
    uvicorn web.backend.app:app --reload --port 8000
Then open http://localhost:8000
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import sys
import traceback
import uuid
import zipfile
from pathlib import Path

logger = logging.getLogger("distill.app")

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Make `distill` importable when the app is launched from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / "web" / ".env")
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

from distill.io import load  # noqa: E402
from distill.types import Stage  # noqa: E402

from .ingest import FILE_ROLES, ingest, ingest_confirmed, stage_ingest  # noqa: E402
from .llm import answer_question, chat_available, stream_answer  # noqa: E402
from .sessions import SessionStore  # noqa: E402

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

RUN_ARTIFACTS = _REPO_ROOT / os.environ.get("RUN_ARTIFACTS_DIR", "run_artifacts").lstrip("./")
UPLOADS_DIR = RUN_ARTIFACTS / "_uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR = _REPO_ROOT / "web" / "frontend"
SAMPLE_DATASET = _REPO_ROOT / "examples" / "adult_sample.csv"

_STAGE_BY_NAME = {
    "duplicate": Stage.DUPLICATE,
    "missingness": Stage.MISSINGNESS,
    "outlier": Stage.OUTLIER,
}

app = FastAPI(title="Distill Agent", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    ).split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

store = SessionStore(out_root=RUN_ARTIFACTS)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class Resolution(BaseModel):
    card_id: str
    action: str
    note: str | None = None


class ResolveBody(BaseModel):
    resolutions: list[Resolution] = []


class ChatBody(BaseModel):
    question: str


class TransformBody(BaseModel):
    col: str
    operation: str
    params: dict = {}


class CompanionModeBody(BaseModel):
    mode: str                       # "reference_only" | "bundle_all"
    match_template: str | None = None  # optional — triggers subject matching when set
    # Legacy alias — accepted but match_template takes precedence.
    match_key: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session(sid: str):
    try:
        return store.get(sid)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown session: {sid}")


def _stage(name: str) -> Stage:
    s = _STAGE_BY_NAME.get(name)
    if s is None:
        raise HTTPException(status_code=400, detail=f"Unknown stage: {name}")
    return s


def _artifact_dir(sid: str) -> Path:
    """Resolve a session's on-disk artifact directory.

    Uses `Path(sid).name` so the session id can never escape RUN_ARTIFACTS.
    Reads straight from disk — downloads keep working even after the
    server restarts or the in-memory session has been evicted.
    """
    return (RUN_ARTIFACTS / Path(sid).name).resolve()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "chat_available": chat_available(),
        "sample_available": SAMPLE_DATASET.is_file(),
    }


# In-memory store for pending ingestion proposals (sid → IngestionProposal).
# Kept separate from SessionStore because the session doesn't exist yet at
# the point the proposal is created.
_pending_proposals: dict[str, Any] = {}


@app.post("/api/upload")
async def upload(
    files: list[UploadFile] = File(...),
    docs: list[UploadFile] | None = File(None),
    note: str | None = Form(None),
) -> JSONResponse:
    """Upload data files + optional documentation.

    When only one tabular file is detected the session is created immediately
    and profiling runs.  When multiple tabular files are present the response
    includes an ``ingestion_proposal`` object that the frontend renders as a
    decision card; the analyst confirms or overrides file roles, then POSTs
    to ``/api/ingestion/confirm`` to proceed.
    """
    batch = UPLOADS_DIR / uuid.uuid4().hex[:10]
    batch.mkdir(parents=True, exist_ok=True)

    data_paths: list[Path] = []
    for f in files:
        if not f.filename:
            continue
        dest = batch / Path(f.filename).name
        dest.write_bytes(await f.read())
        data_paths.append(dest)
    if not data_paths:
        raise HTTPException(status_code=400, detail="No data file was uploaded.")

    doc_paths: list[Path] = []
    for f in docs or []:
        if not f.filename:
            continue
        dest = batch / Path(f.filename).name
        dest.write_bytes(await f.read())
        doc_paths.append(dest)

    try:
        ip, needs_confirmation = stage_ingest(
            data_paths, doc_paths, user_note=(note or "").strip()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load the data: {e}")

    if needs_confirmation:
        # Multiple files — return the proposal for the analyst to review.
        proposal_id = uuid.uuid4().hex[:12]
        _pending_proposals[proposal_id] = ip
        return JSONResponse({
            "status": "pending_confirmation",
            "proposal_id": proposal_id,
            "ingestion_proposal": ip.proposal,
        })

    # Single file — auto-confirm and go straight to profiling.
    result = ingest_confirmed(ip)
    hold_out_df = getattr(result, "hold_out_df", None)
    ws = store.create_from_df(
        result.primary_path,
        result.df,
        doc_text=result.doc_text,
        ingestion=result.report,
        companions=result.companions,
        hold_out_df=hold_out_df,
    )
    return JSONResponse(ws.run_profiling())


class IngestionConfirmBody(BaseModel):
    proposal_id: str
    files: list[dict]   # [{filename, role, join_key?}, ...]
    # join_key may be a single column name or a list for compound keys.
    # The LLM sometimes returns a list; we normalise to list[str] internally.
    join_key: str | list[str] | None = None
    key_map: dict | None = None
    na_values: list[str] | None = None
    target_column: str | None = None


def _normalise_join_keys(raw: str | list[str] | None) -> list[str]:
    """Always return a (possibly empty) list of join-key column names."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(k).strip() for k in raw if str(k).strip()]
    # comma-separated string fallback: "period_id, jurisdiction"
    return [k.strip() for k in str(raw).split(",") if k.strip()]


@app.post("/api/ingestion/confirm")
def confirm_ingestion(body: IngestionConfirmBody) -> JSONResponse:
    """Apply the analyst-confirmed ingestion plan and run profiling.

    Accepts the analyst's final file-role assignments (possibly modified from
    the LLM proposal), merges the data accordingly, creates the session, and
    returns the same profiling payload as a single-file upload would.
    """
    ip = _pending_proposals.pop(body.proposal_id, None)
    if ip is None:
        raise HTTPException(
            status_code=404,
            detail=f"No pending ingestion proposal '{body.proposal_id}'. "
                   "It may have already been confirmed or expired.",
        )

    # Validate all roles before doing anything irreversible.
    for f in body.files:
        if f.get("role") not in FILE_ROLES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown role {f.get('role')!r} for '{f.get('filename')}'. "
                       f"Valid: {sorted(FILE_ROLES)}",
            )

    # Build the confirmed plan, normalising join_key to a list.
    confirmed = dict(ip.proposal)
    confirmed["files"] = body.files
    confirmed["join_keys"] = _normalise_join_keys(body.join_key or confirmed.get("join_key"))
    if body.key_map is not None:
        confirmed["key_map"] = body.key_map
    if body.na_values is not None:
        confirmed["na_values"] = body.na_values
    if body.target_column is not None:
        confirmed["target_column"] = body.target_column

    try:
        result = ingest_confirmed(ip, confirmed_files=body.files, confirmed=confirmed)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    hold_out_df = getattr(result, "hold_out_df", None)
    ws = store.create_from_df(
        result.primary_path,
        result.df,
        doc_text=result.doc_text,
        ingestion=result.report,
        companions=result.companions,
        hold_out_df=hold_out_df,
    )
    return JSONResponse(ws.run_profiling())


@app.post("/api/sample")
def sample() -> JSONResponse:
    """Create a session from the bundled UCI Adult Income sample dataset.

    Lets a first-time visitor try the whole pipeline without having to
    find a dataset of their own.
    """
    if not SAMPLE_DATASET.is_file():
        raise HTTPException(status_code=404, detail="Bundled sample dataset not found.")
    try:
        df = load(SAMPLE_DATASET)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load sample: {e}")

    report = {
        "strategy": "single",
        "summary": (
            "Loaded the bundled UCI Adult Income sample — 6,000 rows of US "
            "census data. It is naturally messy (placeholder '?' values, "
            "numbers stored as text, hyphenated column names), so it "
            "exercises every cleaning stage. The outcome column is 'income'."
        ),
        "detail": [],
        "files_loaded": ["adult_sample.csv"],
        "na_values": [],
        "target_column": "income",
        "used_llm": False,
        "has_documentation": False,
        "is_sample": True,
        "companions": [],
        "skipped": [],
        "doc_files": [],
    }
    ws = store.create_from_df(SAMPLE_DATASET, df, "income", ingestion=report)
    return JSONResponse(ws.run_profiling())


@app.post("/api/session/{sid}/companion")
def run_companion_stage(sid: str, body: CompanionModeBody) -> dict:
    """Stage 1.5 — companion file handling + optional subject matching.

    *mode* controls packaging: 'reference_only' (record paths only) or
    'bundle_all' (copy files into the output ZIP).

    *match_template* is optional and independent of mode.  When provided it
    triggers row-level subject matching.  May be a plain column name
    ('subject_id') or a compound format string ('{jurisdiction}_{period_id}').
    """
    ws = _session(sid)
    # Resolve template: prefer match_template, fall back to legacy match_key.
    template = body.match_template or body.match_key or None
    try:
        return ws.run_companion_stage(body.mode, match_template=template)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/session/{sid}/companion/propose")
def propose_companion_match(sid: str) -> dict:
    """Ask the LLM to propose a companion match template from the session's documentation.

    Returns {available, template, rationale, confidence, column_refs}.
    When no API key is configured, returns available=False with a fallback message.
    """
    return _session(sid).propose_companion_match()


@app.post("/api/session/{sid}/reset")
def reset_session(sid: str) -> dict:
    """Start over: rewind the pipeline to stage 1, keeping the uploaded data.

    Discards all cleaning decisions, profiles, and stage cards, restores the
    pristine intake DataFrame, then re-runs profiling so the frontend can
    return to the post-upload state without a re-upload.
    """
    ws = _session(sid)
    try:
        ws.reset_pipeline()
        return ws.run_profiling()
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("reset failed: %s\n%s", e, tb)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/api/session/{sid}/format")
def run_format(sid: str) -> dict:
    return _session(sid).run_format()


@app.post("/api/session/{sid}/detect/{stage}")
def detect(sid: str, stage: str) -> dict:
    try:
        return _session(sid).detect_stage(_stage(stage))
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("detect/%s failed: %s\n%s", stage, e, tb)
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}",
        )


@app.post("/api/session/{sid}/resolve/{stage}")
def resolve(sid: str, stage: str, body: ResolveBody) -> dict:
    ws = _session(sid)
    try:
        return ws.resolve_stage(
            _stage(stage),
            [r.model_dump() for r in body.resolutions],
        )
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("resolve/%s failed: %s\n%s", stage, e, tb)
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}",
        )


@app.post("/api/session/{sid}/feature_engineering/detect")
def fe_detect(sid: str) -> dict:
    return _session(sid).detect_feature_engineering()


@app.post("/api/session/{sid}/feature_engineering/resolve")
def fe_resolve(sid: str, body: ResolveBody) -> dict:
    ws = _session(sid)
    return ws.resolve_feature_engineering([r.model_dump() for r in body.resolutions])


@app.post("/api/session/{sid}/feature_engineering/transform")
def fe_transform(sid: str, body: TransformBody) -> dict:
    ws = _session(sid)
    try:
        return ws.apply_on_demand_transform(body.col, body.operation, body.params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))



class CompanionMatchingResolveBody(BaseModel):
    action: str   # "keep_rows" | "drop_rows" | "flag_caveat"
    note: str | None = None


@app.post("/api/session/{sid}/companion_matching/resolve")
def resolve_companion_matching(sid: str, body: CompanionMatchingResolveBody) -> dict:
    ws = _session(sid)
    try:
        result = ws.resolve_companion_matching(body.action, note=body.note)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


class FinalizeRequest(BaseModel):
    output_formats: list[str] = ["csv"]


@app.post("/api/session/{sid}/finalize")
def finalize(sid: str, body: FinalizeRequest | None = Body(default=None)) -> dict:
    ws = _session(sid)
    formats = (body.output_formats if body else None) or ["csv"]
    try:
        result = ws.finalize(output_formats=formats)
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("finalize failed: %s\n%s", e, tb)
        # Structured detail: a friendly message for the user, with the
        # technical error and traceback kept in separate fields so the UI can
        # tuck them behind a "details" toggle instead of dumping a raw
        # traceback into the chat.
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Finalizing the session failed. Your uploaded data and "
                    "decisions are still intact — you can try again or reload."
                ),
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb,
            },
        )
    # bundle_all removed — reference_only is the only companion mode.
    # Ask the LLM for usage advice, but cap the wait to 15 s so a slow
    # or absent API key never blocks the response.
    s = result["summary"]
    has_comp = bool(ws.companions)
    advice_ctx = {
        "dataset": ws.state.input_path.name,
        "rows_in": s["rows_in"], "rows_out": s["rows_out"],
        "cols_in": s["cols_in"], "cols_out": s["cols_out"],
        "n_decisions": s["n_decisions"],
        "artifacts": list(result["artifacts"].keys()),
        "has_companion_files": has_comp,
        "n_companions": len(ws.companions),
    }
    # Persist the full chat context so /chat/stream keeps working after a server restart.
    try:
        import json as _json
        ctx_path = _artifact_dir(sid) / "chat_context.json"
        ctx_path.write_text(
            _json.dumps(_build_chat_context(ws), default=str), encoding="utf-8"
        )
    except Exception:
        pass

    try:
        import concurrent.futures as _cf
        _ex = _cf.ThreadPoolExecutor(max_workers=1)
        _fut = _ex.submit(
            answer_question,
            "The cleaning session is complete. Give concise, practical advice "
            "(3-5 bullet points) on how to use the output: "
            "when to use the Python script vs. the CSV, "
            "how to load companion files if any, "
            "and any next steps for modelling.",
            advice_ctx,
        )
        _ex.shutdown(wait=False)  # don't block on thread; we rely on _fut.result(timeout=)
        try:
            advice = _fut.result(timeout=15)
            result["usage_advice"] = advice.get("answer", "")
        except _cf.TimeoutError:
            result["usage_advice"] = ""
    except Exception:
        result["usage_advice"] = ""
    return result


@app.get("/api/session/{sid}/status")
def status(sid: str) -> dict:
    return _session(sid).status()


@app.get("/api/session/{sid}/repro_check")
def repro_check(sid: str) -> dict:
    return _session(sid).get_repro_status()


@app.get("/api/session/{sid}/variable/{name}")
def variable(sid: str, name: str) -> dict:
    ws = _session(sid)
    try:
        data = ws.variable_data(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No such variable: {name}")

    # Attach a human-readable description (doc-sourced or AI-inferred),
    # generated once and memoised on the session.
    desc, source = ws.describe_variable(name)
    data["description"] = desc
    data["description_source"] = source
    return data


@app.get("/api/session/{sid}/artifact/{name}")
def artifact(sid: str, name: str):
    """Download one output artifact straight from disk."""
    base = _artifact_dir(sid)
    path = (base / Path(name).name).resolve()
    if base not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail=f"No such artifact: {name}")
    return FileResponse(
        path,
        filename=path.name,
        media_type="application/octet-stream",
    )


@app.get("/api/session/{sid}/download_all")
def download_all(sid: str):
    """Download every output artifact for a session as a single zip."""
    base = _artifact_dir(sid)
    if not base.is_dir():
        raise HTTPException(status_code=404, detail="No artifacts for this session.")
    files = [f for f in sorted(base.iterdir()) if f.is_file()]
    if not files:
        raise HTTPException(status_code=404, detail="No artifacts for this session.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)
        # Include companion files (images etc.) when user chose "bundle_all".
        comp_dir = base / "companions"
        if comp_dir.is_dir():
            for cf in sorted(comp_dir.iterdir()):
                if cf.is_file():
                    zf.write(cf, arcname=f"companions/{cf.name}")
    buf.seek(0)
    fname = f"distill_{base.name}.zip"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


def _build_chat_context(ws) -> dict:
    """Assemble the LLM context dict from a WebSession."""
    import math as _math

    pending_cards = []
    for cards in ws.stage_cards.values():
        for c in cards:
            meta = dict(c.metadata)
            for list_key in ("flagged_indices", "implausible_indices"):
                if isinstance(meta.get(list_key), list):
                    meta[list_key] = meta[list_key][:10]
            if isinstance(meta.get("flagged_values"), list):
                meta["flagged_values"] = meta["flagged_values"][:15]
            pending_cards.append({
                "card_id": c.card_id,
                "stage": c.stage.value,
                "variable": c.variable,
                "issue": c.issue,
                "status": c.status.value,
                "default_action": c.default_action,
                "action_taken": c.action_taken,
                "metadata": meta,
            })

    def _compact_profile(p) -> dict:
        d = p.to_dict()
        d.pop("sample_values", None)
        if d.get("categorical"):
            d["categorical"]["top_values"] = d["categorical"]["top_values"][:5]
        return d

    def _safe_val(v):
        try:
            f = float(v)
            return None if (_math.isnan(f) or _math.isinf(f)) else f
        except (TypeError, ValueError):
            pass
        return None if (v is None) else str(v)

    sample_rows = ws.state.df.head(8).to_dict(orient="records")
    safe_sample = [{str(k): _safe_val(v) for k, v in row.items()} for row in sample_rows]

    return {
        "active_stage_cards": pending_cards,
        "data_sample": safe_sample,
        "documentation": (ws.doc_text or "")[:3000],
        "dataset": ws.state.input_path.name,
        "target_column": ws.state.target_column,
        "profile": [_compact_profile(p) for p in ws.profiles.values()],
        "decisions_log": [c.to_dict() for c in ws.state.log][-10:],
    }


@app.post("/api/session/{sid}/chat")
def chat(sid: str, body: ChatBody) -> dict:
    """Non-streaming fallback — kept for backward compat."""
    ws = _session(sid)
    context = _build_chat_context(ws)
    _ex = _cf.ThreadPoolExecutor(max_workers=1)
    _fut = _ex.submit(answer_question, body.question, context)
    _ex.shutdown(wait=False)
    try:
        return _fut.result(timeout=25)
    except _cf.TimeoutError:
        return {"available": False, "answer": "The chat request timed out. Try a shorter question, or ask again."}


@app.post("/api/session/{sid}/chat/stream")
def chat_stream(sid: str, body: ChatBody):
    """SSE streaming chat — tokens arrive progressively, no client-side timeout needed.

    Falls back to the on-disk chat_context.json snapshot when the live session
    has been evicted (e.g. after a server restart), so post-session chat keeps
    working even across reloads.
    """
    import json as _json

    # Try live session first; fall back to persisted snapshot.
    context = None
    try:
        ws = _session(sid)
        context = _build_chat_context(ws)
    except HTTPException:
        ctx_path = _artifact_dir(sid) / "chat_context.json"
        if ctx_path.exists():
            try:
                context = _json.loads(ctx_path.read_text(encoding="utf-8"))
            except Exception:
                pass

    if context is None:
        def _no_session():
            msg = (
                "Session not found and no cached context available. "
                "Please reload the page and upload your data again."
            )
            yield f"data: {_json.dumps({'token': msg})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(
            _no_session(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        stream_answer(body.question, context),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Frontend (served last so /api routes win)
# ---------------------------------------------------------------------------

if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
