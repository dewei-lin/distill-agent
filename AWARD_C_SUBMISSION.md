# [Award C] Distill Agent

> **Team info**
> | Legal name | Affiliation | Institutional email | Kaggle username |
> |---|---|---|---|
> | Dewei Lin | UNC Chapel Hill Department of Biostatistics | lindewei@unc.edu | deweilin |
>
> **Registered team name:** distill

**GitHub repository:** https://github.com/dewei-lin/distill-agent

---

## What it does

Distill Agent produces data cleaning workflows that are reproducible, fully auditable, and defensible by design — no coding required. Through a point-and-click and chat interface backed by an LLM, it automates the mechanical work and answers analyst questions in plain language, while keeping the human in the driver's seat for every judgment call.

## Demo link

Web demo: [distill-agent.up.railway.app](https://distill-agent.up.railway.app/)

## Agent Design and Architecture

| Component | What it does |
|---|---|
| Brain / LLM | Claude Sonnet (claude-sonnet-4-6) — orchestrates the six-stage pipeline, interprets analyst intent, generates decision card recommendations with rationale, and answers free-form questions about the data in plain language |
| Memory | `decisions.json` (DecisionLog) persisted in `run_artifacts/<session_id>/` across every call — records every automated and human-confirmed action with timestamp, stage, variable, and `confirmed_by`; `fitted_params.json` stores train-derived transform parameters (imputer values, scaler stats, bin edges) for leak-free application to val/test |
| Planning | Orchestrator–worker decomposition: a central orchestrator sequences six worker stages (profiling → format resolution → duplicates → missingness → outliers → feature engineering), delegating each to its dedicated subagent; mechanical stages run autonomously, judgment stages pause and surface a decision card before proceeding |
| Action | Calls `distill` Python library functions — `detect_*` per stage, `apply_decision` to mutate the DataFrame, and `outputs.write_all_outputs` to emit artifacts; web UI calls FastAPI endpoints (`/upload`, `/detect/{stage}`, `/resolve/{stage}`, `/finalize`, `/chat/stream`) |
| Execution | Python process (pandas, scikit-learn for MICE imputation, isolation forest for outliers); FastAPI backend serves the web UI; fixed random seeds throughout for deterministic output |
| Observation | After format resolution, the DataFrame is re-profiled so downstream stages see updated types and missingness counts; after finalization, `clean_script.py` is re-executed against the original source in a temp directory and its output is diffed against `clean.csv` — a mismatch blocks hand-off |
| Response | Five output artifacts per session: `clean.csv` (cleaned dataset), `report.md` + `report.pdf` (human-readable audit report), `flowchart.svg` + `flowchart.png` (CONSORT-style cleaning flowchart), `decisions.json` (machine-readable decision log), `clean_script.py` (standalone Python script that reproduces `clean.csv` byte-for-byte from the original source) |

