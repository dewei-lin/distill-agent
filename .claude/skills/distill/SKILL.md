---
name: distill
description: Clean a tabular dataset reproducibly with a five-stage human-in-the-loop pipeline — profiling, format resolution, duplicate detection, missingness, and outliers. Use whenever the user wants to clean, tidy, prepare, audit, or quality-check a CSV / Excel / SPSS / Stata dataset before analysis or modelling. Produces clean data plus a full audit trail.
---

# Distill Agent — reusable data-cleaning skill

A reproducible, auditable data-cleaning pipeline. Mechanical fixes are
applied autonomously; judgment calls are surfaced to the analyst as
decision cards. Every decision is logged and re-emitted as a standalone
script.

## When to use this skill

Trigger when the user wants to clean / tidy / prepare / quality-check a
tabular dataset, or asks "what's wrong with this data", or needs an
analysis-ready dataset with an audit trail.

## Prerequisites

The `distill` package must be importable. From the repo root:

```bash
pip install -e .
```

## The pipeline (five stages)

1. **Profiling** — type / cardinality / missingness per variable (read-only).
2. **Format resolution** — autonomous: type coercion, whitespace cleanup,
   non-standard missing tokens (`?`, `NA`, ...) to NaN, column-name
   standardisation, categorical-label normalisation.
3. **Duplicates** — exact duplicates + entity-key collisions. Judgment.
4. **Missingness** — MCAR/MAR classification, then drop / impute per
   variable. Judgment — surface a decision card.
5. **Outliers** — IQR + Z-score; implausible vs extreme. Judgment.

## How to run it

### One-shot, unattended

```bash
python -m distill.cli clean <dataset.csv> --target <target-col>
```

Or from Python:

```python
from distill.pipeline import run_pipeline
result = run_pipeline("data.csv", target_column="income")
print(result.summary)
```

Both apply the agent's default for every judgment decision and write the
five artifacts to `run_artifacts/<session_id>/`.

### Interactive (human-in-the-loop)

Pass a `decide` callback to `run_pipeline` — it is called for every PENDING
`DecisionCard`; resolve the card (confirm or override) inside it. With no
callback, defaults are used. See `distill/pipeline.py`.

To drive it conversationally in Claude Code, use the `/clean` command,
which delegates each stage to the matching subagent in `.claude/agents/`.

## Outputs (always five)

Written to `run_artifacts/<session_id>/`:

| File | What it is |
|---|---|
| `clean.csv` | the cleaned dataset |
| `report.md` / `report.pdf` | human-readable Cleaning Audit Report |
| `flowchart.svg` / `flowchart.png` | CONSORT-style cleaning flow diagram |
| `decisions.json` | machine-readable decision log (LLM-composable) |
| `clean_script.py` | standalone script that reproduces `clean.csv` byte-for-byte |

## Adopting individual stages

Each stage is independently importable — you do not need the whole
pipeline. For example, just the missingness analysis:

```python
from distill import profile, detect_missingness
cards = detect_missingness(df, profile(df))
```

The same is true of `detect_outliers`, `detect_duplicates`,
and `resolve_formats`.

## Reference files

- `CLAUDE.md` — full operating manual (decision rules, output contract).
- `.claude/agents/` — one subagent per stage, plus `orchestrator`.
- `docs/ARCHITECTURE.md` — design notes.
