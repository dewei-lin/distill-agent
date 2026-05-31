# Distill Agent — AI-Assisted Reproducible Data Cleaning

> **STAI-X Challenge 2026 — Award C submission (Statistical Skill / Agent Module)**

---

## What is **Distill Agent**?

Data is to data scientists what water is to life — and just as you can't survive on contaminated water, you can't build reliable models on dirty data. Yet real-world data is almost always messy, and cleaning it is consistently the most time-consuming step in any analysis pipeline. Worse, even well-executed cleaning is rarely reproducible: decisions go undocumented, analysts disagree on edge cases, and after three months nobody can reconstruct the choices they made themselves.

We built **Distill Agent** to be the distillation device for data. AI handles the mechanical, pattern-detection work **autonomously** — writing and executing the code — while the human analyst keeps **full control** over the judgment calls that require domain knowledge. And like a well-run water utility, nothing happens in the dark: every decision, automated or human, is **logged**, **justified**, and compiled into a **reproducible** set of outputs — including a standalone Python script that regenerates the cleaned data with no agent in the loop. Clean data you can drink, and a record of exactly how it was treated.
---

## Core Concept: Human-in-the-Loop Data Cleaning

The agent operates as an intelligent co-pilot, not an autopilot. It distinguishes between two types of decisions:

**Mechanical decisions** — format coercion, type detection, encoding fixes, whitespace removal, categorical label normalization, column name standardization, exact-duplicate flagging — are handled autonomously by the agent, which generates and runs the corresponding code without interrupting the analyst. Every such change is still logged.

**Judgment decisions** — should this variable be dropped or imputed? Is this outlier a data-entry error or a genuine extreme value? Should these near-duplicate records be merged? — are surfaced to the analyst as structured decision cards, each with a clear recommendation, the reasoning behind it, and the option to **confirm**, **override**, or **inspect** further before proceeding.

This keeps the analyst in the driver's seat while eliminating the tedious boilerplate that slows down every project. When the agent is run non-interactively, each card falls back to a sensible default action, and the report records exactly where defaults were used.

---

## Interface

Distill Agent runs as a no-code, chat-based web interface — no scripting required. The layout has three panels working together:

**Pipeline sidebar (left):** Tracks progress through each stage of the cleaning pipeline in real time, shows dataset metadata (row count, variable count, decisions logged), and provides one-click downloads for every output artifact once the session completes. A **Start Over** control rewinds the pipeline to the first stage while keeping the uploaded data in place — useful for trying a different set of decisions without re-uploading.

**Chat panel (center):** The primary interaction surface. The agent narrates what it finds at each stage and surfaces judgment decisions as decision cards — each card shows what was detected, what the agent recommends, and why. The analyst clicks Confirm, Override, or Inspect; overrides are written back to the chat as a natural conversation, and every decision is logged automatically as it is made.

**Variable inspector (right):** An interactive data-visualization panel. For any continuous variable, the inspector shows the distribution as a histogram, summary statistics, missingness rate, and flagged outliers; histogram bars are clickable, letting the analyst highlight a subset of rows and ask the agent to inspect them. Categorical variables show frequency breakdowns. Analysts can switch between variables at any time, or open one directly from a decision card.

The interface is designed so that an analyst with no programming background can complete a full cleaning session and walk away with production-ready outputs.

---

## Pipeline Stages

Before stage 1, an **intake step** inspects everything that was uploaded and asks at most one clarifying message. It detects the shape of the input — a single table, a train/validation/test split, multiple tables to join, or non-tabular *companion* files (images, audio, text) — and, when a codebook or data dictionary is provided, reads it to load the data correctly.

### 1. Profiling pass
The agent profiles every variable: stored versus intended type, value distribution and cardinality, missingness rate, and type mismatches (e.g. numbers stored as strings, dates stored as free text). No data is changed.

### 2. Companion matching (multimodal datasets)
When non-tabular companion files are present, the agent checks which rows have a matching file. If coverage is complete, it asks whether to store the file-path column in the cleaned table (kept out by default). If some rows are unmatched, it offers to keep all rows, add a presence indicator (`has_<type>`), drop the unmatched rows, or simply note the gap in the report. Companion files are never silently merged into the table.

### 3. Format resolution pass
The agent autonomously resolves unambiguous formatting issues and generates the corresponding code: type coercion, categorical label normalization (`"Yes"` / `"yes"` / `"Y"` → `"Yes"`), date normalization, whitespace and encoding cleanup, and column-name standardization.

### 4. Duplicate and near-duplicate detection
Exact duplicate rows are flagged for removal; near-duplicates — records sharing an entity key but differing elsewhere — are surfaced for judgment. Duplicates are handled before the missingness and outlier passes so those passes measure rates and fit imputations on the genuine, de-duplicated sample.

### 5. Missingness decision pass
For each variable with missing data, the agent classifies the pattern (MCAR / MAR / MNAR), summarizes how much is missing, and recommends a strategy — drop the variable, drop the affected rows, or impute (median, mode, or MICE iterative-regression imputation). Each recommendation is a decision card with a plain-English explanation. On split datasets, imputers are fit on the training split only and the fitted parameters are replayed on validation/test.

### 6. Outlier and anomaly detection pass
Potential outliers are flagged using IQR, Z-score, and isolation forest, and presented for review. The agent distinguishes likely data-entry errors from genuine extreme values and explains its reasoning in each case.

### 7. Feature engineering pass (binning · scaling · encoding)
A combined, opt-in stage. The agent suggests binning only for skewed or wide-range columns, scaling only when columns are on dramatically different scales, and frequency encoding for high-cardinality categoricals; datetime decomposition is always analyst-initiated. If none of the criteria are met, the stage exits silently. Analysts can also request any transform on demand via chat or the selector panel. On split datasets, bin edges, scaling parameters, and encoding tables are computed on the training split only and replayed downstream.

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

**Additional artifacts** are produced when the dataset warrants them: `fitted_params.json` and separate train/validation ML-ready datasets for split sessions, and `companion_checksums.json` plus a HuggingFace Dataset for multimodal sessions.

### Output format selection
After cleaning, the analyst chooses the output format(s) — **CSV** (universal), **Parquet** (typed, compressed), and/or **HuggingFace Dataset** (when companions are present, for direct use in PyTorch / HF Trainer). The generated script's entry point produces exactly the selected formats from a single command.

Together, these outputs ensure the cleaning process is auditable, shareable, re-runnable, and composable with other tools and agents downstream.

---

## Why This Matters

**Reproducibility.** The entire process can be re-run from scratch, shared with collaborators, or attached as a methods appendix — and the agent verifies the emitted script actually reproduces the result.

**Transparency.** Every decision has a justification. There are no black-box transformations, no silent drops, no undocumented changes.

**Efficiency.** Analysts spend time on the judgment calls that require expertise — not on writing boilerplate inspection and imputation code that looks the same on every project.

**Composability.** The JSON decision log lets downstream agents read and act on the session, making Distill Agent a first-class component in larger multi-agent analysis pipelines.

**Accessibility.** No coding required. Any analyst can complete a full cleaning session through the chat interface and walk away with production-ready outputs and figures.

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

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/lindewei0423/distill-agent.git
cd distill-agent
pip install -e .               # core library
pip install -e ".[dev]"        # add test + lint tooling (optional)
pip install -e ".[web]"        # add the web demo (optional)
```

`pip install -e .` pulls in everything the core pipeline needs — you do not have to install the packages below by hand; they are listed so you know exactly what the app depends on and the minimum version of each.

**Core dependencies** (installed automatically):

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

**Optional — `[web]`** (only for the browser demo): `fastapi` ≥ 0.110, `uvicorn[standard]` ≥ 0.27, `python-multipart` ≥ 0.0.9, `sse-starlette` ≥ 2.1, `anthropic` ≥ 0.39, `python-dotenv` ≥ 1.0, `pydantic` ≥ 2.6, `pypdf` ≥ 4.0.

**Optional — `[dev]`** (only for tests / linting): `pytest` ≥ 8.0, `pytest-cov` ≥ 4.1, `ruff` ≥ 0.4, `mypy` ≥ 1.9.

Every cleaning run also writes a `requirements.txt` and an `environment.yml` into its session folder, pinned to the exact versions used, so a cleaned dataset can always be reproduced in a matching environment.

## Quick start

Distill Agent is meant to be used **interactively** — you stay in the loop and answer one decision card at a time, so the judgment calls that shape your data are always yours. There are two ways to work that way.

**Inside Claude Code (recommended).** Open this repo with Claude Code and run the slash command:

```
/clean your_data.csv your_outcome_column
```

Claude narrates what it finds at each stage and surfaces every judgment call as a decision card — **Confirm**, **Override**, or **Inspect** — then produces the full artifact set once you're done. The cleaning skill lives in `.claude/skills/distill/`; drop that folder into any other Claude Code project to reuse it. See [`docs/SKILL_USAGE.md`](docs/SKILL_USAGE.md).

**In the browser.** A three-panel UI: a pipeline sidebar, a decision-card chat, and an interactive variable inspector.

```bash
pip install -e ".[web]"
uvicorn web.backend.app:app --port 8000      # then open http://localhost:8000
```

Upload one or more data files (CSV, TSV, Excel, Parquet, JSON, SPSS, Stata, SAS) plus an optional codebook / data dictionary (PDF, Markdown, or text) — the agent reads the documentation to load the data correctly and catalogue any non-tabular companion files — or click **Try the sample dataset** to start with no upload at all. The chat-based judgment uses an Anthropic API key (`.env`, see `.env.example`).

**Reuse a single stage as a library.** Every stage is independently importable — e.g. `from distill import detect_missingness` — so you can borrow one component without the whole pipeline.

## Tests

```bash
pytest
```

The suite covers the core library end to end.

---

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
