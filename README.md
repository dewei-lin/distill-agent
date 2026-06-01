# Distill Agent — AI-Assisted Reproducible Data Cleaning

> **STAI-X Challenge 2026 — Award C submission (Statistical Skill / Agent Module)**

---

## What is **Distill Agent**?

Data is to data scientists what water is to life — and just as you can't survive on contaminated water, you can't build reliable models on dirty data. Yet real-world data is almost always messy, and cleaning it is consistently the most time-consuming step in any analysis pipeline. Worse, even well-executed cleaning is rarely reproducible: decisions go undocumented, analysts disagree on edge cases, and after three months nobody can reconstruct the choices they made themselves.


We built **Distill Agent** to be the distillation device for data. AI handles the mechanical, pattern-detection work **autonomously** — writing and executing the code — while the human analyst keeps **full control** over the judgment calls that require domain knowledge. And like a well-run water utility, nothing happens in the dark: every decision, automated or human, is **logged**, **justified**, and compiled into a **reproducible** set of outputs — including a standalone Python script that regenerates the cleaned data with no agent in the loop. Clean data you can drink, and a record of exactly how it was treated.

---

## Interface

Distill Agent runs as a no-code, chat-based web interface — no scripting required. The layout has four major panels working together:

![panel-illustration](assets/panel.png)

---

## How do I run this?

**Try it online** — no setup needed: [distill-agent.up.railway.app](https://distill-agent.up.railway.app/)

**Run it locally** — requires Python 3.10+:

```bash
git clone https://github.com/lindewei0423/distill-agent.git && cd distill-agent && pip install -e ".[web]"
uvicorn web.backend.app:app --port 8000
```

Open [http://localhost:8000](http://localhost:8000). Upload one or more data files (ZIP, CSV, TSV, Excel, Parquet, JSON, SPSS, Stata, SAS) plus an optional codebook / data dictionary — or click **Try the sample dataset** to start with no upload at all. For data with multiple files, we recommend uploading the entire ZIP file.


---

## What to expect from Distill Agent?

★ **Efficiency.** Analysts spend time on judgment calls that require expertise — not on boilerplate inspection and imputation code. At the end of every session, a self-contained Python script is generated so you can reproduce the clean dataset from your original upload with a single command.

★ **Human-in-the-loop.** The agent is a co-pilot, not an autopilot. It handles the mechanical work autonomously and surfaces every judgment call as a structured decision card — so you stay in the driver's seat, with a clear rationale attached to every choice.

★ **Accessibility.** No coding required. Any analyst can complete a full cleaning session through the chat interface and walk away with production-ready outputs, audit reports, and figures. Every decision card is written in plain language, so domain experts — not just data engineers — can meaningfully review and override recommendations.

★ **Reproducibility.** The entire session can be re-run from scratch, shared with collaborators, or attached as a methods appendix. The agent automatically verifies that the emitted script reproduces the result before closing the session. Every decision has a justification — no black-box transformations, no silent drops, no undocumented changes.

★ **Composability.** The JSON decision log is machine-readable, making Distill Agent a first-class component in larger multi-agent pipelines. Cleaning a similar dataset later? Upload the log and skip re-explaining the whole process.

---


## Pipeline Stages

Before stage 1, an **intake step** inspects everything that was uploaded and asks at most one clarifying message. It detects the shape of the input — a single table, a train/validation/test split, multiple tables to join, or non-tabular *companion* files (images, audio, text) — and, when a codebook or data dictionary is provided, reads it to load the data correctly.

### 1. Profiling
The agent profiles every variable: stored versus intended type, value distribution and cardinality, missingness rate, and type mismatches (e.g. numbers stored as strings, dates stored as free text). No data is changed.

### 2. Companion matching
When non-tabular companion files are present, the agent checks which rows have a matching file. If coverage is complete, it asks whether to store the file-path column in the cleaned table (kept out by default). If some rows are unmatched, it offers to keep all rows, add a presence indicator (`has_<type>`), drop the unmatched rows, or simply note the gap in the report. Companion files are never silently merged into the table.

### 3. Format resolution
The agent autonomously resolves unambiguous formatting issues and generates the corresponding code: type coercion, categorical label normalization (`"Yes"` / `"yes"` / `"Y"` → `"Yes"`), date normalization, whitespace and encoding cleanup, and column-name standardization.

### 4. (near-)Duplicate detection
Exact duplicate rows are flagged for removal; near-duplicates — records sharing an entity key but differing elsewhere — are surfaced for judgment. Duplicates are handled before the missingness and outlier passes so those passes measure rates and fit imputations on the genuine, de-duplicated sample.

### 5. Missingness decision
For each variable with missing data, the agent classifies the pattern (MCAR / MAR / MNAR), summarizes how much is missing, and recommends a strategy — drop the variable, drop the affected rows, or impute. 

### 6. Outlier and anomaly detection
Potential outliers are flagged using IQR, Z-score, and isolation forest, and presented for review. The agent distinguishes likely data-entry errors from genuine extreme values and explains its reasoning in each case.

### 7. Feature engineering
A combined, opt-in stage. The agent suggests binning only for skewed or wide-range columns, scaling only when columns are on dramatically different scales, and frequency encoding for high-cardinality categoricals; datetime decomposition is always analyst-initiated. If none of the criteria are met, the stage exits silently. Analysts can also request any transform on demand via chat or the selector panel.

---

## Outputs

Every cleaning session writes a self-contained artifact set to `run_artifacts/<session_id>/`. The core five:

### (a) Clean dataset
The cleaned data file, ready for analysis — exported as CSV plus the original format (and optionally Parquet or a HuggingFace Dataset; see below). Every transformation is reflected here, with no silent changes.

### (b) Human-readable audit report
A structured Cleaning Audit Report (`report.md` and `report.pdf`) for analysts, collaborators, and reviewers: a variable summary table showing each variable's original and final state side by side, before-and-after statistics, a plain-English narrative of what was found and why each decision was made, and a log of every analyst override with its stated reason.

### (c) Cleaning flowchart
A publication-ready figure in the style of a CONSORT participant-flow diagram (`flowchart.svg` and `flowchart.png`), visualizing every step from raw to clean — the row count entering each stage, what changed and why, and the row count exiting. It can be embedded directly in a paper, report, or presentation.

### (d) Machine-readable decision log
A structured JSON log (`decisions.json`) of every decision — variable, issue, recommendation, action taken, justification, and whether it was human-confirmed or agent-defaulted — designed to be passed directly to another LLM for downstream reasoning or composed into a larger analysis pipeline.

### (e) Reproducible cleaning script
A complete, standalone `clean_script.py` that replays every step from raw data to clean output with no dependency on the agent — only pandas / numpy (and scikit-learn when MICE is used). Running `python clean_script.py <input>` regenerates exactly the artifacts selected during the session. After it is written, the session **automatically replays it against the original source and verifies the output matches** before hand-off; the result is surfaced in the report.


---

## Architecture

The agent follows an Orchestrator–Worker pattern with human-in-the-loop checkpoints:

```
Upload (one or more files + optional codebook)
        ↓
Intake — detect structure (single / split / join / multimodal), read documentation
        ↓
Orchestrator
(coordinates stages, manages human-in-the-loop checkpoints)
        ↓
┌────────────────────────────────────────────────────────────┐
│  Profiling  →  Companion matching  →  Format                 │
│  Duplicates →  Missingness         →  Outliers               │
│  Feature engineering (binning · scaling · encoding)          │
└────────────────────────────────────────────────────────────┘
        ↓
Human-in-the-Loop Checkpoints
(judgment decisions surfaced as cards; confirmed, overridden, or defaulted)
        ↓
Output generation
(clean data, audit report, CONSORT flowchart, JSON log,
 reproducible script — then automatic reproducibility check)
```

Each worker is independent and reusable: the missingness component, for example, can be called standalone on any dataset without running the full pipeline, so individual pieces are adoptable on their own.

---

## Requirements

Core dependencies (installed automatically):

| Package | Min version | Used for |
|---|---|---|
| `pandas` | 2.1 | the DataFrame engine — all loading and transformation |
| `numpy` | 1.26 | numeric arrays underpinning pandas / scikit-learn |
| `scipy` | 1.11 | distribution statistics used in profiling and outlier detection |
| `scikit-learn` | 1.4 | MICE imputation, isolation-forest outliers, scalers |
| `openpyxl` | 3.1 | reading / writing `.xlsx` files |
| `pyreadstat` | 1.2 | reading SPSS (`.sav`), Stata (`.dta`), SAS (`.sas7bdat`) |
| `chardet` | 5.2 | text-encoding detection on load |
| `jinja2` | 3.1 | templating the Markdown audit report |
| `reportlab` | 4.1 | rendering the PDF audit report |
| `matplotlib` | 3.8 | plots embedded in the report and flowchart |

Optional `[web]` (only for the browser demo): `fastapi` ≥ 0.110, `uvicorn[standard]` ≥ 0.27, `python-multipart` ≥ 0.0.9, `sse-starlette` ≥ 2.1, `anthropic` ≥ 0.39, `python-dotenv` ≥ 1.0, `pydantic` ≥ 2.6, `pypdf` ≥ 4.0.

Optional `[dev]` (only for tests / linting): `pytest` ≥ 8.0, `pytest-cov` ≥ 4.1, `ruff` ≥ 0.4, `mypy` ≥ 1.9.

Every cleaning run also writes a `requirements.txt` and an `environment.yml` into its session folder, pinned to the exact versions used, so a cleaned dataset can always be reproduced in a matching environment.


## Repository Structure

```
.
├── distill/                 # Core library — one module per stage + outputs
│   ├── profiler.py  companions.py  format.py
│   ├── duplicates.py  missingness.py  outliers.py
│   ├── binning.py  scaling.py  encoding.py   # feature-engineering pass
│   ├── report.py  flowchart.py  script_gen.py  outputs.py
│   ├── pipeline.py  cli.py                    # orchestration + CLI
│   └── state.py  types.py  io.py  action_labels.py   # shared primitives
├── .claude/                 # Claude Code integration
│   ├── skills/distill/      #   the adoptable skill (SKILL.md)
│   ├── agents/              #   one subagent per stage
│   └── commands/clean.md    #   the /clean slash command
├── web/                     # Optional web demo (FastAPI + vanilla JS)
│   ├── backend/             #   app.py, sessions.py, llm.py
│   └── frontend/            #   single-page three-panel UI
├── examples/                # Bundled sample dataset + runnable demos
├── tests/                   # pytest suite
├── docs/                    # ARCHITECTURE.md, SKILL_USAGE.md, UNSTRUCTURED_DATA.md
├── CLAUDE.md                # Operating manual Claude Code reads on startup
└── README.md
```

---

## License

MIT
