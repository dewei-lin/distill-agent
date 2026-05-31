# [Award C] Distill Agent — AI-Assisted Reproducible Data Cleaning

**Team info**

- **Legal name:** Dewei Lin
- **Affiliation:** _[fill in — your institution / program]_
- **Institutional email:** _[fill in — or lindewei0423@gmail.com]_
- **Kaggle username:** _[fill in]_

**Registered team name:** _[fill in]_

**GitHub repository:** _[fill in — e.g. https://github.com/lindewei0423/distill-agent]_
**Demo:** see "Demo" below
**Submission commit tag:** `award-c-submission`

---

## What it does (plain language)

Data cleaning is the most time-consuming step in any statistical analysis —
commonly 60–80% of project time — and the least reproducible: two analysts
cleaning the same file make different undocumented judgment calls and get
different results.

**Distill Agent** is a reusable agent module that fixes this. It runs a
five-stage human-in-the-loop cleaning pipeline. The mechanical, pattern-
detection work (type coercion, encoding, whitespace, column naming,
non-standard missing tokens) is done autonomously. The work that needs
domain judgment — drop or impute this variable? is this outlier an error
or a real extreme? should these near-duplicates be merged? — is surfaced to the
analyst as a **decision card** with a recommendation, the reasoning, and
alternatives. The analyst confirms, overrides, or inspects.

Every decision, automated or human, is logged and justified. The session
produces five artifacts: the clean dataset, a human-readable audit report
(Markdown + PDF), a CONSORT-style cleaning flowchart, a machine-readable
JSON decision log, and a standalone Python script that reproduces the
clean dataset byte-for-byte.

It is built for analysts, biostatisticians, and applied researchers who
need a defensible, auditable cleaning process — not a black box.

## The six-stage pipeline

1. **Profiling** — per-variable type, cardinality, missingness, and type-
   mismatch detection (read-only).
2. **Format resolution** — autonomous, mechanical fixes: type coercion,
   whitespace/encoding cleanup, non-standard missing tokens (`?`, `NA`,
   `null`) to `NaN`, `snake_case` column names, categorical-label
   normalisation.
3. **Duplicates** — exact-duplicate rows, plus entity-key collisions when
   an id column exists. Handled first, so the later passes measure
   missingness and outliers on the de-duplicated sample. Judgment.
4. **Missingness** — classifies the pattern (MCAR / MAR) and recommends
   drop / impute (median, mode, MICE iterative regression). Judgment.
5. **Outliers** — IQR + Z-score detection, distinguishing physically
   *implausible* values (likely data-entry errors) from *extreme but
   plausible* tails. Judgment.

## Why this is a statistical skill, not just a code tool

- **Uncertainty-aware missingness.** The agent classifies the missingness
  *mechanism* (MCAR vs. MAR) from the data and chooses the imputation
  method to match — model-based imputation when missingness depends on
  observed variables, marginal imputation when it does not — rather than
  blindly mean-filling.
- **Robustness by construction.** Outlier thresholds are quartile-based so
  the outliers themselves do not distort the fences; "implausible" is
  reserved for the one signal that is reliable without domain knowledge
  (a negative value in a positive-domain column) so the agent never
  silently deletes a legitimate heavy tail.
- **Reproducibility is the headline feature.** The pipeline is fully
  deterministic (fixed seeds for stochastic imputers), and every run emits
  a standalone script that regenerates the cleaned file exactly — verified
  byte-for-byte in the test suite.
- **Auditability.** Nothing is a black box: every transformation is a
  logged decision card with a justification, and the audit report shows
  every variable's before/after state and every analyst override.

## Demo

The agent is usable three ways, all driving the **same** cleaning engine:

- **Claude Code** — open the repo and run `/clean <dataset> <target>`. The
  agent walks you through each decision card conversationally.
- **Web app** — a three-panel browser UI (pipeline sidebar, decision-card
  chat, interactive variable inspector): `uvicorn web.backend.app:app`.
- **CLI / Python** — `distill clean data.csv --target outcome`, or
  `run_pipeline(...)` from Python.

A representative end-to-end run on the UCI Adult dataset (32k rows): the
agent detects nine numeric columns stored as strings, converts 4,262 `?`
missing-sentinels to `NaN`, classifies `workclass`/`occupation`/
`native-country` missingness as MAR, flags `capital-gain`/`capital-loss`
as heavy-tailed extremes, removes exact-duplicate rows, and emits all five
artifacts — with a full decision log.

## Reproducibility

- **Full source** in the GitHub repository above (MIT-licensed).
- **Quick start** in the `README`: `pip install -e ".[dev]"`, then
  `python examples/quick_demo.py`.
- **Bundled data:** a 6,000-row sample of the UCI Adult dataset
  (`examples/`), plus `fetch_adult.py` documenting provenance.
- **Test suite:** `pytest` over the core library.
- **Submission snapshot:** commit tagged `award-c-submission`.

## Agent design and architecture

**Pattern:** orchestrator–worker. Five independent worker stages are
sequenced by an orchestrator that manages the human-in-the-loop
checkpoints and triggers the output writers.

**Shared core, three interfaces.** A single deterministic library
(`distill/`) is the cleaning engine; the Claude Code skill, the CLI, and
the web app are thin layers over it — so all three produce identical
output.

**The decision-card protocol.** Every action — mechanical or judgment — is
one uniform `DecisionCard` object. A single `decide` callback seam lets the
same orchestration run silently (CLI / unattended), interactively (web),
or conversationally (Claude Code).

Full design notes: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Why it's reusable

- **Drop-in Claude Code skill.** `.claude/skills/distill/` is
  self-contained — copy it into any Claude Code project to add the
  cleaning capability. See [`docs/SKILL_USAGE.md`](docs/SKILL_USAGE.md).
- **Composable stages.** Every stage is independently importable —
  `from distill import detect_missingness` — so adopters can borrow a
  single component without the whole pipeline.
- **LLM-composable output.** The `decisions.json` log is designed to be
  read by a downstream agent, making Distill Agent a first-class
  component in a larger multi-agent analysis pipeline.

## Relevance to STAI-X 2026

Distill Agent targets the step every statistical analysis depends on and
that most undermines reproducibility. It embodies the STAI-X goal of
trustworthy statistics-plus-AI: an agent that does the tedious work
autonomously, keeps the human in control of every judgment call, and
leaves behind an audit trail and a reproducible script that any reviewer
can re-run. The same pipeline prepares the kind of public-health panel
data this competition targets — surfacing missingness mechanisms and
data-entry errors before any model is fit.

---

*Feedback and contributions welcome. Open-source under MIT.*
