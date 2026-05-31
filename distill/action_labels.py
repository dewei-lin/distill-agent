"""Plain-language labels for decision actions.

Single source of truth that turns the internal snake_case action
identifiers used across the cleaning stages (e.g. ``treat_as_missing``,
``median_impute``) into human-readable labels and one-line descriptions.

No code-format identifier should ever reach the user interface: the web
app renders every decision option through :func:`options_for`, and the
decision-log JSON carries the same labels so downstream readers never see
a raw identifier either.
"""

from __future__ import annotations

# Generic label + description per action, used when a stage does not
# override it in ``_BY_STAGE`` below. Each value is (label, description).
_GENERIC: dict[str, tuple[str, str]] = {
    "drop_variable": (
        "Remove this column",
        "Delete the whole column from the dataset.",
    ),
    "drop_rows": (
        "Delete the affected rows",
        "Remove every row involved in this issue.",
    ),
    "leave_as_is": (
        "Leave it unchanged",
        "Make no change to the data.",
    ),
    "mean_impute": (
        "Fill the gaps with the average",
        "Replace the missing values with the column's average (mean).",
    ),
    "median_impute": (
        "Fill the gaps with the median",
        "Replace the missing values with the column's middle (median) value.",
    ),
    "mode_impute": (
        "Fill the gaps with the most common value",
        "Replace the missing values with the most frequent value in the column.",
    ),
    "regression_impute": (
        "Fill the gaps using a model (MICE)",
        "Predict each missing value from the other columns in the row using "
        "iterative regression (MICE).",
    ),
    "flag_missing_indicator": (
        "Keep blank, add a 'was missing' marker",
        "Leave the values blank but add a yes/no column recording which "
        "rows were missing.",
    ),
    "treat_as_missing": (
        "Replace with N/A (missing)",
        "Blank out the flagged values so they are treated as missing data.",
    ),
    "winsorize": (
        "Cap them at the normal range",
        "Pull the flagged values in to the edge of the column's typical range.",
    ),
    "inspect": (
        "Leave for now — needs a closer look",
        "Make no change yet; review the variable before deciding.",
    ),
    "remove_all_duplicates": (
        "Remove the duplicate copies",
        "Keep one row from each duplicate group and drop the extra copies.",
    ),
    "keep_all_duplicates": (
        "Keep every row",
        "Make no change — leave the duplicate rows in place.",
    ),
    "keep_all": (
        "Keep every row",
        "Make no change — leave the rows in place.",
    ),
    "drop_all_duplicate_rows": (
        "Delete every duplicated row",
        "Remove all rows involved in a duplicate, including the first copy.",
    ),
    "merge_keep_first": (
        "Keep the first record",
        "When the same entity appears more than once, keep the first row "
        "and drop the rest.",
    ),
    "merge_keep_last": (
        "Keep the most recent record",
        "When the same entity appears more than once, keep the last row "
        "and drop the rest.",
    ),
    "flag_caveat": (
        "Keep it, but add a warning to the report",
        "Keep the column and record a caveat in the report.",
    ),
    "proceed": ("Continue", "No action needed — move on."),
    "skip": ("Skip this check", "No action needed — move on."),
    # Binning actions (Stage 7)
    "bin_equal_width": (
        "Bin into equal-width intervals",
        "Divide the value range into N equally-spaced bins.",
    ),
    "bin_quantile": (
        "Bin into quantile groups",
        "Divide into N groups of roughly equal size (one group per quantile).",
    ),
    "bin_natural_breaks": (
        "Bin at natural breaks",
        "Find natural cluster boundaries in the distribution (Jenks/k-means).",
    ),
    # Scaling actions (Stage 8)
    "scale_standard": (
        "Standardise (z-score)",
        "Subtract the mean and divide by the standard deviation → mean 0, std 1.",
    ),
    "scale_minmax": (
        "Scale to [0, 1]",
        "Shift and rescale so all values fall between 0 and 1.",
    ),
    "scale_robust": (
        "Robust-scale (median / IQR)",
        "Centre on the median and scale by the interquartile range — "
        "unaffected by extreme values.",
    ),
    "scale_log": (
        "Log-transform (log1p)",
        "Apply log(1 + x) to compress a right-skewed distribution.",
    ),
    # Encoding actions (Stage 9 / Feature Engineering)
    "frequency_encode": (
        "Frequency encode",
        "Replace each category with its relative frequency in the dataset.",
    ),
    "datetime_decompose": (
        "Decompose datetime",
        "Extract year, month, day-of-week, hour, and is_weekend into new columns.",
    ),
    # Format-stage automatic actions. Never shown as buttons, but included
    # so the decision-log JSON stays free of raw identifiers too.
    "renamed": ("Renamed", "Column names standardised automatically."),
    "stripped": ("Trimmed", "Stray spaces removed automatically."),
    "converted": (
        "Converted",
        "Placeholder values converted to missing automatically.",
    ),
    "coerced": (
        "Re-typed",
        "Column converted to its proper data type automatically.",
    ),
    "normalized": (
        "Tidied",
        "Label spellings made consistent automatically.",
    ),
}

# Stage-specific overrides, keyed by (stage value, action). These exist
# because the same action means something slightly different per stage —
# e.g. "drop_rows" removes missing rows in the missingness stage but
# flagged-outlier rows in the outlier stage.
_BY_STAGE: dict[tuple[str, str], tuple[str, str]] = {
    ("missingness", "drop_rows"): (
        "Delete the rows with a missing value",
        "Remove every row that has no value in this column.",
    ),
    ("missingness", "leave_as_is"): (
        "Leave the values as N/A (missing)",
        "Make no change — keep the missing values in place.",
    ),
    ("outlier", "drop_rows"): (
        "Delete the rows with a flagged value",
        "Remove every row whose value here was flagged as an outlier.",
    ),
    ("outlier", "leave_as_is"): (
        "Keep these values — they are valid",
        "Treat the flagged values as genuine and make no change.",
    ),
    ("duplicate", "inspect"): (
        "Leave them — review manually",
        "Make no change; these records need a human look before deciding.",
    ),
    ("binning", "leave_as_is"): (
        "Keep as continuous",
        "Make no change — leave the column as a numeric variable.",
    ),
    ("scaling", "leave_as_is"): (
        "Keep unscaled",
        "Make no change — leave the column on its current scale.",
    ),
    ("encoding", "leave_as_is"): (
        "Leave as categorical",
        "Make no change — keep the column as-is.",
    ),
    # Companion-matching cards (companion stage)
    ("companion", "keep_rows"): (
        "Keep all subjects",
        "Include subjects without a companion; they will have has_companion = False.",
    ),
    ("companion", "drop_rows"): (
        "Remove subjects with no companion",
        "Drop every row whose subject ID has no matching companion file.",
    ),
    ("companion", "flag_caveat"): (
        "Keep all, add a note to the report",
        "Keep every row and record the missing-companion count as a caveat in the audit report.",
    ),
    ("companion", "add_indicator"): (
        "Keep all, add a presence indicator",
        "Keep every row and add a has_<type> column flagging which rows have a companion file.",
    ),
    ("companion", "exclude_path"): (
        "Keep the path column out of the data",
        "Reconstruct companion file paths on demand in the script; do not store them in the cleaned table.",
    ),
    ("companion", "include_path"): (
        "Include the companion path column",
        "Store the resolved companion file-path column (e.g. image_path) in the cleaned table.",
    ),
}


def describe(stage: str, action: str) -> dict[str, str]:
    """Return ``{'action', 'label', 'description'}`` for one action.

    ``stage`` is the stage's string value (e.g. ``"missingness"``).
    Unknown actions fall back to a humanised version of the identifier so
    the UI never crashes on an unexpected value.
    """
    key = (str(stage), str(action))
    pair = _BY_STAGE.get(key) or _GENERIC.get(str(action))
    if pair is None:
        pair = (_humanize(action), "")
    return {"action": str(action), "label": pair[0], "description": pair[1]}


def options_for(
    stage: str,
    default_action: str,
    alternatives: list[str],
) -> list[dict[str, object]]:
    """Build the ordered, de-duplicated option list for a decision card.

    The recommended (default) action is always first and carries
    ``recommended: True``. Every entry has a plain-language label and
    description — never a raw identifier.
    """
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for action in [default_action, *(alternatives or [])]:
        if action is None or action in seen:
            continue
        seen.add(action)
        entry: dict[str, object] = dict(describe(stage, action))
        entry["recommended"] = action == default_action
        out.append(entry)
    return out


def label_for(stage: str, action: str) -> str:
    """Just the plain-language label for an action."""
    return describe(stage, action)["label"]


def _humanize(action: str) -> str:
    """Fallback: turn an unknown identifier into a readable phrase."""
    return str(action).replace("_", " ").strip().capitalize() or "Action"
