"""Optional Anthropic-powered chat for the web demo.

The core cleaning pipeline and every decision card are produced by the
`distill` library with no LLM involved — the web app is fully functional
without an API key. This module adds ONE optional extra: a free-text chat
box where the analyst can ask questions about their data ("why is bmi
flagged MAR?") and get a conversational answer.

That is the only place the web app spends Anthropic API credits, and it
uses the cheapest model (Haiku) — so a demo session costs a fraction of a
cent. If `ANTHROPIC_API_KEY` is not set, the chat endpoint returns a clear
"disabled" message and everything else keeps working.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterator

# Default model — cheap and fast; overridable via env.
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

_APP_KNOWLEDGE = """
=== Distill Agent — App Guide (answer UI questions from this) ===

WHAT IT DOES
Distill Agent is a human-in-the-loop data-cleaning tool. You upload a CSV \
(or Excel/SPSS/Stata/ZIP), the agent runs a 7-stage pipeline and surfaces \
decision cards at every judgment call. You confirm or override each card, then \
download five output artifacts.

UI LAYOUT (three panels)
- Left sidebar: pipeline progress (7 dots), session stats, output download links.
- Centre chat panel: all agent messages, decision cards, and this chat box.
- Right panel: Variable Inspector (histogram + stats for every column) with a \
  tab strip at the top to switch columns.

PIPELINE STAGES (left sidebar dots)
1. Profiling — reads every column, detects types, missingness, distribution. No changes.
2. Format resolution — autonomous fixes: whitespace, encoding, type coercion, \
   label normalisation. Logged but no card needed.
3. Duplicates — exact duplicate rows and entity-key collisions surfaced as cards.
4. Missingness — each column with missing values gets a card (MCAR/MAR pattern + \
   imputation recommendation).
5. Outliers — IQR + Z-score flags; distinguishes "implausible" (data-entry error) \
   from "extreme but plausible".
6. Feature Engineering — optional binning, scaling, frequency encoding. Only \
   suggested when data warrants it.

DECISION CARDS
Each card shows: the issue, the recommended action (green button), and alternative \
actions (grey buttons).
- **Accept / green button** — apply the recommended action.
- **Override** — click a grey alternative button to choose a different action.
- **Inspect** — click "Inspect" or "Show distribution" to see the data before deciding.
- For numeric missingness cards, model-based imputation uses MICE \
  (iterative regression, BayesianRidge, 10 rounds). Categorical columns use \
  mode imputation. Click ⓘ on a card to see the method details.
- Decisions cannot be undone within a session. To retry, use "Start Over" \
  (top of the left sidebar) to reload the page with a fresh session.

OUTPUTS (bottom of left sidebar, available after Finalize)
- clean.csv — the cleaned dataset (also in original format if Excel/SPSS).
- report.md / report.pdf — audit report with every decision explained.
- flowchart.svg / flowchart.png — CONSORT-style data-flow diagram.
- decisions.json — machine-readable log of every decision.
- clean_script.py — standalone Python script that reproduces the entire \
  cleaning pipeline. Run it: python clean_script.py <your-raw-file>.
  The output clean.csv is written to the current working directory.
- "Download all (zip)" link bundles everything.

REPRODUCIBILITY CHECK
After Finalize, the agent automatically reruns clean_script.py against the \
original data in a temporary folder and compares the result to the session \
output. A green tick means the script reproduces the session exactly. A \
warning means a mismatch — the script is still valid but may have minor \
numerical differences (e.g. MICE iterative-regression convergence).

START OVER
The "Start Over" button is at the top of the left sidebar, above the Pipeline \
section. Clicking it reloads the page so you can upload a new dataset. \
Downloaded artifacts from the previous session remain on disk.

VARIABLE INSPECTOR (right panel)
Click any column tab to see: AI-inferred description, missingness bar, \
distribution histogram (numeric) or top-values bar chart (categorical), \
and summary statistics (mean, median, std). The inspector updates live \
after each cleaning stage.

COMPANION FILES
If your dataset references external files (images, audio, documents), upload \
them as a ZIP together with the CSV. The agent will detect and register them \
as companion objects.

SESSION EXPIRY
Sessions live in server memory. If the server restarts (e.g. during development \
with hot-reload), the live session is lost — but the chat can still answer \
questions using the saved context snapshot, and all downloaded artifacts remain \
on disk. Use "Start Over" to begin a fresh session.
"""

_SYSTEM_PROMPT = """You are the assistant inside Distill Agent, a \
human-in-the-loop data-cleaning tool. Answer questions about the data \
being cleaned AND about how to use the app itself.

APP KNOWLEDGE
""" + _APP_KNOWLEDGE + """

SESSION CONTEXT (supplied per request)
You are given up to five sources — use ALL of them:

1. active_stage_cards — current decision cards with full metadata:
   - flagged_values: actual data values that triggered the flag
   - implausible_indices / flagged_indices: row positions
   - bounds: IQR fences (iqr_low / iqr_high), min, max, mean, std
   - severity: "implausible" | "extreme" | "mild"
   ALWAYS read flagged_values from the relevant card before answering.

2. data_sample — 8 actual rows. Use for "show me some data" questions.

3. documentation — analyst's codebook / data dictionary (if provided).

4. profile — per-column statistics (type, missing rate, min/max/mean/std).

5. decisions_log — transformations already applied.

ANSWERING DATA QUESTIONS
When answering about outliers or implausible values:
1. Read flagged_values from the card and list the specific values.
2. Explain WHY each is problematic in domain context.
3. State the IQR fence bounds.
4. Recommend treat-as-missing, winsorize, or keep — one sentence.

ANSWERING APP / UI QUESTIONS
Use the App Knowledge above. Be specific about button names and locations. \
If asked "where is X", say which panel and what label to look for. \
If asked "can I undo", explain the Start Over button.

Answer concisely. Keep answers under 200 words unless the user asks for more. \
Never invent numbers not present in the context.

POST-SESSION USAGE QUESTIONS
If the session is complete and the user asks how to load or use the clean data \
in a specific environment, provide ready-to-run code. Use the EXACT dataset \
filename from the "dataset" field in context — never write "your_raw_data.csv". \
Cover Python/Jupyter, Google Colab, terminal script, and R. \
Format all code in fenced code blocks (```python```, ```bash```, ```r```)."""


def _placeholder_key(key: str | None) -> bool:
    return not key or key.strip() in {
        "", "your_anthropic_api_key_here", "sk-ant-xxx",
    }


def chat_available() -> bool:
    """True if an Anthropic API key is configured."""
    return not _placeholder_key(os.environ.get("ANTHROPIC_API_KEY"))


def propose_companion_match(
    doc_text: str,
    columns: list[str],
    filename_stems: list[str],
    *,
    model: str | None = None,
) -> dict:
    """Use the LLM to propose a companion match template from documentation.

    Returns::

        {
          "available": bool,
          "template": str,          # e.g. "{jurisdiction}_{period_id}"
          "rationale": str,
          "confidence": "high"|"medium"|"low",
          "column_refs": list[str],
        }

    Falls back gracefully when no API key is set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if _placeholder_key(api_key):
        return {"available": False, "template": None, "rationale": "LLM unavailable.",
                "confidence": "low", "column_refs": []}

    try:
        import anthropic
    except ImportError:
        return {"available": False, "template": None, "rationale": "anthropic package not installed.",
                "confidence": "low", "column_refs": []}

    system = (
        "You are a data integration assistant. "
        "Given dataset documentation, a list of CSV column names, and a sample of "
        "companion filenames, propose how to derive the companion filename stem "
        "from one CSV row using a Python str.format() template.\n\n"
        "Rules:\n"
        "- Use only column names from the provided list.\n"
        "- A plain column name (e.g. 'subject_id') is valid when one column "
        "  directly matches the filename stem.\n"
        "- A compound template (e.g. '{jurisdiction}_{period_id}') is needed when "
        "  filenames are built from multiple columns.\n"
        "- Respond with JSON only — no markdown, no explanation outside the JSON.\n\n"
        "Response schema:\n"
        '{"template":"<str>","rationale":"<one sentence>","confidence":"high|medium|low",'
        '"column_refs":["col1",...]}'
    )

    stems_sample = json.dumps(filename_stems[:12])
    user_msg = (
        f"CSV columns: {json.dumps(columns)}\n\n"
        f"Sample companion filename stems: {stems_sample}\n\n"
        f"Documentation:\n{doc_text[:4000]}"
    )

    model = model or os.environ.get("ANTHROPIC_MODEL_FAST", _DEFAULT_MODEL)
    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = "".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        ).strip()
        # Strip accidental markdown fences.
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
        proposal = json.loads(text)
        proposal["available"] = True
        proposal.setdefault("column_refs", [])
        return proposal
    except Exception as e:
        return {"available": False, "template": None,
                "rationale": f"LLM call failed: {e}", "confidence": "low", "column_refs": []}


def stream_answer(
    question: str,
    context: dict[str, Any],
    *,
    model: str | None = None,
) -> Iterator[str]:
    """Yield SSE-formatted chunks for the chat streaming endpoint.

    Each chunk is a ``data: {...}\\n\\n`` line.  The final chunk is always
    ``data: [DONE]\\n\\n``.  Never raises — errors are surfaced as an
    ``{"error": "..."}`` event followed by ``[DONE]``.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if _placeholder_key(api_key):
        msg = (
            "Chat is disabled. Copy .env.example to .env and set "
            "ANTHROPIC_API_KEY to enable conversational questions."
        )
        yield f"data: {json.dumps({'token': msg})}\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        import anthropic
    except ImportError:
        yield f"data: {json.dumps({'error': 'anthropic package not installed'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    model = model or os.environ.get("ANTHROPIC_MODEL_FAST", _DEFAULT_MODEL)
    client = anthropic.Anthropic(api_key=api_key)

    context_json = json.dumps(context, default=str)[:20000]
    user_msg = (
        f"Session context (JSON):\n{context_json}\n\n"
        f"Question: {question}"
    )

    try:
        with client.messages.stream(
            model=model,
            max_tokens=400,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'token': text})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': f'{type(e).__name__}: {e}'})}\n\n"
    finally:
        yield "data: [DONE]\n\n"


def answer_question(
    question: str,
    context: dict[str, Any],
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Answer a free-text question about the current cleaning session.

    `context` is a JSON-serialisable dict — typically the variable profile
    and recent decision cards. Returns {available, answer}.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if _placeholder_key(api_key):
        return {
            "available": False,
            "answer": (
                "Chat is disabled. Copy .env.example to .env and set "
                "ANTHROPIC_API_KEY to enable conversational questions. "
                "The cleaning pipeline itself does not need an API key."
            ),
        }

    try:
        import anthropic
    except ImportError:
        return {
            "available": False,
            "answer": "The 'anthropic' package is not installed "
                      "(pip install -e \".[web]\").",
        }

    model = model or os.environ.get("ANTHROPIC_MODEL_FAST", _DEFAULT_MODEL)
    client = anthropic.Anthropic(api_key=api_key)

    context_json = json.dumps(context, default=str)[:20000]  # keep prompt small
    user_msg = (
        f"Session context (JSON):\n{context_json}\n\n"
        f"Question: {question}"
    )

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=400,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        return {"available": True, "answer": text.strip(), "model": model}
    except Exception as e:  # network / auth / rate-limit
        return {
            "available": False,
            "answer": f"Chat request failed: {type(e).__name__}: {e}",
        }
