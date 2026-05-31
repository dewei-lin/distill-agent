# Distill Agent — web demo

A browser UI over the Distill Agent pipeline: upload one or more data
files (plus an optional codebook / data dictionary), walk through the six
cleaning stages, accept or change each decision card, and download the
output artifacts. It is an optional showcase — the Claude Code skill is
the primary deliverable — but it lets anyone try the agent without
installing Claude Code.

First-time visitors can click **Try the sample dataset** to run the whole
pipeline on a bundled copy of the UCI Adult Income data — no upload needed.

## Uploading data

The data zone accepts every format the engine reads — CSV, TSV, Excel,
Parquet, JSON, SPSS (`.sav`), Stata (`.dta`), SAS — and you can drop more
than one file. The separate documentation zone accepts a codebook / data
dictionary / README as PDF, Markdown or text. When documentation is
provided, an LLM reads it to plan ingestion: which file is the main
table, how multiple files relate (single / stack / join), and which coded
values mean "missing" (e.g. `-99`, `Unknown`). With no documentation — or
no API key — a deterministic heuristic is used instead. Non-tabular files
(images, etc.) are catalogued as companion files and left untouched.

## Architecture

```
web/
├── backend/          FastAPI app
│   ├── app.py        HTTP routes (thin layer)
│   ├── sessions.py   WebSession / SessionStore — the pipeline core
│   └── llm.py        optional Anthropic-powered chat
└── frontend/         single-page UI (vanilla HTML/JS + Chart.js)
```

The backend reuses the exact same `distill` library as the Claude Code
path, so the web app and the agent produce identical cleaned output.

## Run it

```bash
# from the repo root
pip install -e ".[web]"

# (optional) enable the chat box
cp .env.example .env             # then add your ANTHROPIC_API_KEY to .env

uvicorn web.backend.app:app --port 8000
```

Open <http://localhost:8000>.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/upload` | upload data file(s) + optional documentation + target; runs profiling |
| POST | `/api/sample` | start a session from the bundled UCI Adult sample |
| POST | `/api/session/{sid}/format` | autonomous format resolution |
| POST | `/api/session/{sid}/detect/{stage}` | decision cards for a stage |
| POST | `/api/session/{sid}/resolve/{stage}` | apply the analyst's answers |
| POST | `/api/session/{sid}/finalize` | write the output artifacts |
| GET  | `/api/session/{sid}/variable/{col}` | inspector data (histogram + stats) |
| GET  | `/api/session/{sid}/artifact/{file}` | download one artifact |
| GET  | `/api/session/{sid}/download_all` | download every artifact as a zip |
| POST | `/api/session/{sid}/chat` | optional Anthropic-powered Q&A |

`{stage}` is one of `duplicate`, `missingness`, `outlier`.
Artifacts are served straight from disk, so downloads keep working even
after the server restarts.

## Cost

The pipeline and every decision card are produced by the `distill`
library — **no API key, no inference cost**. The only paid call is the
optional chat box, which uses Claude Haiku (a fraction of a cent per
question). Without `ANTHROPIC_API_KEY` the chat box reports itself
disabled and everything else works normally.
