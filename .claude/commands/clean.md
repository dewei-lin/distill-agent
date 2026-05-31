---
description: Clean a dataset with the Distill Agent five-stage human-in-the-loop pipeline.
argument-hint: <dataset-path> [target-column]
---

# /clean — run the Distill Agent pipeline

The user wants to clean a dataset with Distill Agent.

- Dataset: `$1`
- Target / outcome column (optional): `$2`

First read `CLAUDE.md` (the operating manual) if you have not already this
session. Then proceed.

## Decide the mode

- **Interactive** — a human is in the conversation and can answer questions.
  Surface every judgment decision and wait for them. This is the default
  when a person is clearly present.
- **Unattended** — you were given a single instruction with nobody to reply
  (e.g. an automated `Do the data analysis` run). Use agent defaults for
  every judgment decision; do not ask questions.

## Intake (before Stage 1)

Before profiling, ask the analyst two things — together, in one message:

1. **Data context** (optional): "Do you have any notes about this dataset —
   what it represents, known quirks, or a codebook? I'll include them in the
   report."
2. **Target variable** (optional): "Is there an outcome or label column you're
   trying to predict? (Recorded for downstream modelling and reporting.)"

If the analyst provides notes, store them in `PipelineState.intake_metadata`.
If multiple files were uploaded, check whether they need combining — see the
multi-file joining rules below. Then proceed.

### Multi-file combining

When more than one tabular file is uploaded:
- Delegate to `ingest()` in `web/backend/ingest.py` (or use `plan_ingestion`
  directly). The LLM planner will detect the right strategy: stack / join.
- **Key column name mismatch**: if files use different column names for the
  same entity (e.g. "patient_id" vs "pid"), the planner returns a `key_map`.
  Show it to the analyst: "I'll join on 'patient_id', renaming 'pid' in
  lab_results.csv. Does that look right?"
- **Image or unstructured companions**: register them as `DataCompanion`
  objects (name, mime type). They are never merged into the DataFrame. Tell
  the analyst: "I've registered 42 image files as companion files — they
  won't be cleaned but will be listed in the report."

## Interactive run

Work stage by stage. Stages 1-2 are autonomous; stages 3-7 may need the
analyst (stages 6-7 only surface cards when warranted).

1. **Profiling** — delegate to the `profiler` subagent. Report the variable
   count, type mismatches, and missingness overview.
2. **Format resolution** — delegate to the `format` subagent. Report the
   mechanical fixes applied. These need no confirmation.
3. **Duplicates**, 4. **Missingness**, 5. **Outliers** —
   delegate to the matching subagent. For each, surface every PENDING
   decision card as a short, plain-English message: what was found, the
   recommendation, the reasoning, and the alternatives. Ask the analyst to
   **Confirm**, **Override** (naming an alternative), or **Inspect**. Apply
   their answer before moving on. Never modify data before they answer.
6. **Binning** — delegate to the `binning` subagent. Only surfaces cards
   when a column is skewed or has a wide range relative to its IQR. If
   nothing is flagged, the stage logs "no action needed" and moves on.
7. **Scaling** — delegate to the `scaling` subagent. Only surfaces cards
   when numeric columns are on dramatically different scales or are heavily
   skewed. If nothing is flagged, the stage logs "no action needed" and
   moves on.

**At any point** the analyst can type a transform request in the chat —
"bin age into 5 groups", "standardise income" — and the relevant subagent
will handle it immediately as a confirmed on-demand card.

Maintain the working DataFrame across stages by keeping it in the session
directory; re-load it at the start of each stage.

When all seven stages are done, produce the five output artifacts (clean
data, audit report, flowchart, decision log, reproducible script) by
calling `distill.outputs.write_all_outputs`.

## Unattended run

Run the whole pipeline non-interactively with agent defaults:

```bash
python -m distill.cli clean "$1"          # add: --target "<col>" if a target was given
```

This applies the default action for every judgment decision and writes all
five artifacts to `run_artifacts/<session_id>/`. Then read `report.md`
from that folder and summarise it for the user.

## Always finish by

Telling the user, in a few lines:
- rows in → out, columns in → out;
- how many decisions were automated vs. judgment calls vs. overridden;
- the path to each of the five artifacts.
