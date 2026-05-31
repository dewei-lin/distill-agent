# Adopting the Distill skill

Distill Agent is built so other participants can **reuse it** — that is
the point of Award C. There are three levels of adoption, from lightest
to fullest.

## 1. Drop the skill into another Claude Code project

The cleaning capability is packaged as a self-contained Claude Code skill
at `.claude/skills/distill/`.

```bash
# from your other project's root
cp -r /path/to/distill-agent/.claude/skills/distill .claude/skills/
pip install distill-agent          # or: pip install -e /path/to/distill-agent
```

Now, in that project, Claude Code will use the skill whenever you ask it
to clean / tidy / quality-check a dataset. The skill's `SKILL.md` tells
the agent how to run the pipeline and where the outputs go.

For the full human-in-the-loop experience, also copy the subagents and the
slash command:

```bash
cp /path/to/distill-agent/.claude/agents/{profiler,format,missingness,outlier,duplicates,orchestrator}.md .claude/agents/
cp /path/to/distill-agent/.claude/commands/clean.md .claude/commands/
```

Then `/clean <dataset> [target]` works in your project.

## 2. Use the Python library

`pip install distill-agent`, then call the orchestrator:

```python
from distill.pipeline import run_pipeline

result = run_pipeline("data.csv", target_column="outcome")
print(result.summary)
# artifacts are in result.session.dir
```

or the CLI:

```bash
distill clean data.csv --target outcome
distill profile data.csv
```

## 3. Adopt a single stage

Every stage is independently importable — you do **not** need the whole
pipeline. Each `detect_*` function takes a DataFrame (and its profile) and
returns `DecisionCard` objects you can inspect or act on.

```python
import pandas as pd
from distill import profile, detect_missingness, detect_outliers

df = pd.read_csv("data.csv")
profiles = profile(df)

# Just the missingness analysis:
for card in detect_missingness(df, profiles):
    print(card.variable, card.metadata["pattern"], "→", card.recommendation)

# Just outlier flagging:
for card in detect_outliers(df, profiles):
    print(card.variable, card.metadata["severity"])
```

The same pattern works for `resolve_formats` and `detect_duplicates`. This
is the lightest form of reuse: borrow the one piece you need.

## What you get out

Every full run writes five artifacts to `run_artifacts/<session_id>/`:

- `clean.csv` — the cleaned dataset
- `report.md` / `report.pdf` — the Cleaning Audit Report
- `flowchart.svg` / `flowchart.png` — the CONSORT-style flow diagram
- `decisions.json` — the machine-readable decision log
- `clean_script.py` — a standalone script that reproduces `clean.csv`

The `decisions.json` log is deliberately LLM-friendly: another agent can
read it to understand exactly what cleaning was done and compose further
steps on top.

## Requirements

- Python 3.10+
- Core dependencies: pandas, numpy, scipy, scikit-learn, matplotlib,
  reportlab, jinja2 (installed automatically by pip).
- The web demo additionally needs the `[web]` extra (FastAPI, uvicorn,
  anthropic).
