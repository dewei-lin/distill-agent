# Distill Agent — Architecture

## Design goal

One cleaning engine, reachable three ways — a Claude Code skill, a Python
API/CLI, and a web app — all producing **identical** output. The way that
is achieved is the central architectural decision: a single deterministic
core library with thin interface layers on top.

```
                    ┌──────────────────────────┐
                    │   distill/  (core)       │
                    │  deterministic library   │
                    │  6 stages + 5 outputs    │
                    └────────────┬─────────────┘
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
   .claude/ (Claude Code)   distill.cli        web/backend (FastAPI)
   skill + subagents +      argparse CLI       sessions.py + app.py
   /clean command                              + web/frontend
```

Because all three call the same `distill` functions, the web demo, the
CLI, and the Claude Code agent clean a given dataset the same way — and the
reproducible script they emit reproduces it again.

## Orchestrator–worker pattern

The pipeline is five worker stages coordinated by an orchestrator:

```
load → profiling → format → duplicate → missingness → outlier
                                                          ↓
                  write 5 artifacts ← finalize ←──────────┘
```

- **Workers** — `profiler`, `format`, `missingness`, `outliers`,
  `duplicates`. Each is one Python module with a `detect_*`
  function and (for judgment stages) an `apply_decision` function. Each is
  independently importable: a worker can be used standalone on any
  DataFrame without the rest of the pipeline.
- **Orchestrator** — `pipeline.run_pipeline`. Sequences the stages,
  refreshes the variable profile after format changes, records per-stage
  row/column counts, resolves decision cards, and triggers the output
  writers. In Claude Code the `orchestrator` subagent plays this role,
  delegating each stage to its worker subagent.

Mechanical vs. judgment is the load-bearing distinction:

- **Mechanical** stages (profiling, format resolution) run autonomously.
  Profiling changes nothing; format resolution applies only unambiguous
  fixes (type coercion, whitespace, sentinel-to-NaN, name standardisation,
  label normalisation).
- **Judgment** stages (duplicates, missingness, outliers) never
  change data without a resolved decision card.

## The decision-card protocol

Every action — mechanical or judgment — is represented by one
`DecisionCard` (`distill/types.py`). This uniform representation is what
makes the audit trail complete and queryable.

```
DecisionCard
  card_id, stage, variable, issue, recommendation, rationale,
  alternatives, default_action, metadata
  ── resolution ──
  status, action_taken, confirmed_by, resolved_at, user_note
```

A card's lifecycle:

1. A `detect_*` function emits the card. Mechanical cards arrive already
   resolved (`status = AGENT_AUTO`). Judgment cards arrive `PENDING`.
2. The orchestrator resolves each `PENDING` card — via a `decide` callback
   (interactive) or by applying `default_action` (`status = AGENT_DEFAULT`).
3. The stage's `apply_decision` executes the resolved card against the
   DataFrame and records what happened in `metadata["applied_info"]`.
4. The card is appended to the `DecisionLog`.

One callback seam — `run_pipeline(..., decide=...)` — drives every mode:
no callback for the CLI and unattended runs, a UI-bound callback for the
web app, and conversational resolution in Claude Code.

## Determinism

Same input → same output, every time. Stochastic steps (MICE iterative
imputation) use fixed random seeds. This is what lets the **reproducible
script** (`clean_script.py`) regenerate `clean.csv` byte-for-byte, and
what keeps the test suite a stable regression check.

## The five output artifacts

Produced by `outputs.write_all_outputs` at the end of every session:

| Artifact | Module | Purpose |
|---|---|---|
| `clean.csv` | `io.save_clean` | the cleaned dataset |
| `report.md` / `report.pdf` | `report.py` | human-readable audit report |
| `flowchart.svg` / `.png` | `flowchart.py` | CONSORT-style flow diagram |
| `decisions.json` | `state.DecisionLog` | machine-readable decision log |
| `clean_script.py` | `script_gen.py` | standalone reproduction script |

Each writer degrades gracefully: a missing optional dependency skips one
artifact rather than failing the run.

## Module map

```
distill/
  types.py        DecisionCard, VariableProfile, enums — the data contract
  state.py        PipelineState, DecisionLog, Session
  io.py           dataset loaders + clean-data writer
  profiler.py     stage 1 — diagnostic profiling
  format.py       stage 2 — autonomous format resolution
  duplicates.py   stage 3 — exact + entity-key collisions
  missingness.py  stage 4 — MCAR/MAR + impute/drop
  outliers.py     stage 5 — IQR / Z-score / implausibility
  report.py       artifact (b)
  flowchart.py    artifact (c)
  script_gen.py   artifact (e)
  outputs.py      artifact orchestration
  pipeline.py     run_pipeline — the orchestrator
  cli.py          command-line interface
```
