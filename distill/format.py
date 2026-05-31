"""Stage 2 — Format resolution.

Autonomous, mechanical fixes that should never need human judgment:

  * Column name standardization (snake_case, no spaces)
  * Whitespace and case cleanup on string columns
  * Type coercion based on profiler flags (numeric/datetime/boolean stored
    as strings)
  * Categorical label normalization (Yes / yes / YES / " yes " -> Yes;
    yes/no/y/n/t/f/true/false canonicalized to Yes / No for boolean-like
    columns)

Every transformation appends a `DecisionCard` with `status=AGENT_AUTO` so
the audit trail is uniform with the human-in-the-loop stages. The card's
metadata records both "before" and "after" exemplars so the report can
show what changed.

This stage is deterministic and idempotent: running it twice on the same
input produces identical output (and the second run produces no cards).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ._pdcompat import is_text_dtype
from .types import (
    DecisionCard,
    DecisionStatus,
    Stage,
    VariableProfile,
    VariableType,
)

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Canonical mappings
# ---------------------------------------------------------------------------

# Boolean-like text values -> canonical {Yes, No}
_BOOL_TRUE = frozenset({"true", "t", "yes", "y", "1"})
_BOOL_FALSE = frozenset({"false", "f", "no", "n", "0"})

# Unambiguous non-standard missing-value sentinels. Compared case-insensitively
# AFTER whitespace stripping. Deliberately conservative: tokens like "none",
# "missing", "-", "." are excluded because they can be legitimate values in
# some columns. Everything here is something essentially nobody uses as a
# real data value.
_MISSING_TOKENS = frozenset(
    {"?", "na", "n/a", "n.a.", "nan", "null", "nil", "#n/a", "<na>", "n\\a"}
)


# Snake-case conversion: replace non-alnum with "_", collapse repeats,
# lowercase, strip leading/trailing underscores.
_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")
_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve_formats(
    df: "pd.DataFrame",
    profiles: dict[str, VariableProfile],
) -> tuple["pd.DataFrame", list[DecisionCard]]:
    """Run the full format-resolution pass.

    Returns the modified DataFrame and a list of AGENT_AUTO decision cards
    describing every change made. Does not mutate the input DataFrame.
    """
    cards: list[DecisionCard] = []
    out = df.copy()

    out, rename_card = _normalize_column_names(out)
    if rename_card is not None:
        cards.append(rename_card)
        # Re-key profiles so subsequent steps can still look them up.
        profiles = _rekey_profiles(profiles, rename_card.metadata["renames"])

    out, strip_cards = _strip_string_whitespace(out)
    cards.extend(strip_cards)

    out, token_cards = _normalize_missing_tokens(out)
    cards.extend(token_cards)
    if token_cards:
        # Missingness counts in the profiles are now stale — refresh so the
        # downstream coercion + missingness stages see the new NaNs.
        from .profiler import profile as _profile

        profiles = _profile(out)

    out, coerce_cards = _coerce_types(out, profiles)
    cards.extend(coerce_cards)

    out, normalize_cards = _normalize_categorical_labels(out, profiles)
    cards.extend(normalize_cards)

    return out, cards


# ---------------------------------------------------------------------------
# Column name standardization
# ---------------------------------------------------------------------------


def to_snake_case(name: str) -> str:
    """Convert an arbitrary column name to snake_case.

    Handles spaces, hyphens, camelCase, PascalCase, mixed punctuation, and
    leading/trailing whitespace. Preserves leading digits by prefixing "_"
    if the original started with a digit (Python identifier safety).
    """
    s = str(name).strip()
    s = _CAMEL_BOUNDARY.sub(r"\1_\2", s)
    s = _NON_ALNUM.sub("_", s).strip("_").lower()
    if s and s[0].isdigit():
        s = "_" + s
    return s or "_"


def _normalize_column_names(df: "pd.DataFrame") -> tuple["pd.DataFrame", DecisionCard | None]:
    new_names = []
    seen: dict[str, int] = {}
    renames: dict[str, str] = {}
    for original in df.columns:
        snake = to_snake_case(original)
        # Disambiguate accidental collisions ("Age (yrs)" and "Age yrs" both
        # snake to "age_yrs") by suffixing _1, _2, ...
        candidate = snake
        if candidate in seen:
            seen[candidate] += 1
            candidate = f"{snake}_{seen[snake]}"
        seen[candidate] = seen.get(candidate, 0)
        new_names.append(candidate)
        if str(original) != candidate:
            renames[str(original)] = candidate

    if not renames:
        return df, None

    out = df.copy()
    out.columns = new_names

    card = DecisionCard(
        card_id="format_column_names",
        stage=Stage.FORMAT,
        issue=f"{len(renames)} column name(s) needed standardization",
        recommendation="rename to snake_case",
        rationale=(
            "Snake_case column names are valid Python identifiers and "
            "prevent quoting issues in downstream code."
        ),
        default_action="renamed",
        metadata={"renames": renames},
        status=DecisionStatus.AGENT_AUTO,
        action_taken="renamed",
        confirmed_by="agent_auto",
    )
    return out, card


# ---------------------------------------------------------------------------
# Whitespace cleanup
# ---------------------------------------------------------------------------


def _strip_string_whitespace(df: "pd.DataFrame") -> tuple["pd.DataFrame", list[DecisionCard]]:
    import pandas as pd

    cards: list[DecisionCard] = []
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if not is_text_dtype(s):
            continue
        # Skip if no string-valued cells (objects can also hold lists/dicts).
        if not s.dropna().map(lambda v: isinstance(v, str)).any():
            continue
        stripped = s.where(s.isna(), s.astype(str).str.strip())
        n_changed = int((stripped != s).fillna(False).sum())
        if n_changed == 0:
            continue
        out[col] = stripped
        cards.append(
            DecisionCard(
                card_id=f"format_strip_ws__{col}",
                stage=Stage.FORMAT,
                variable=col,
                issue=f"{n_changed} value(s) had leading/trailing whitespace",
                recommendation="strip whitespace",
                rationale="Whitespace is invisible and breaks equality checks.",
                default_action="stripped",
                metadata={"n_changed": n_changed},
                status=DecisionStatus.AGENT_AUTO,
                action_taken="stripped",
                confirmed_by="agent_auto",
            )
        )
    return out, cards


# ---------------------------------------------------------------------------
# Non-standard missing-token normalization
# ---------------------------------------------------------------------------


def _normalize_missing_tokens(
    df: "pd.DataFrame",
) -> tuple["pd.DataFrame", list[DecisionCard]]:
    """Convert non-standard missing sentinels ('?', 'NA', 'null', ...) to
    actual NaN in object columns.

    This is mechanical and auto-applied: a value of '?' is essentially never
    a legitimate data point. Converting it lets the missingness stage see
    the true missing rate instead of treating '?' as a category.
    """
    import pandas as pd

    cards: list[DecisionCard] = []
    out = df.copy()

    for col in out.columns:
        s = out[col]
        if not is_text_dtype(s):
            continue

        def is_token(v: object) -> bool:
            return isinstance(v, str) and v.strip().lower() in _MISSING_TOKENS

        token_mask = s.map(is_token)
        n_tokens = int(token_mask.sum())
        if n_tokens == 0:
            continue

        # Record which surface tokens appeared, for the audit trail.
        found = sorted({str(v).strip() for v in s[token_mask].tolist()})
        out.loc[token_mask, col] = pd.NA

        cards.append(
            DecisionCard(
                card_id=f"format_missing_tokens__{col}",
                stage=Stage.FORMAT,
                variable=col,
                issue=(
                    f"{n_tokens} cell(s) held a missing-value sentinel "
                    f"({', '.join(repr(t) for t in found)})"
                ),
                recommendation="convert sentinel tokens to NaN",
                rationale=(
                    "Tokens like '?' / 'NA' / 'null' are placeholders for "
                    "missing data, not real values. Converting them to NaN "
                    "lets the missingness stage measure and handle them."
                ),
                default_action="converted",
                metadata={"n_converted": n_tokens, "tokens_found": found},
                status=DecisionStatus.AGENT_AUTO,
                action_taken="converted",
                confirmed_by="agent_auto",
            )
        )

    return out, cards


# ---------------------------------------------------------------------------
# Type coercion (driven by profile flags)
# ---------------------------------------------------------------------------


def _coerce_types(
    df: "pd.DataFrame",
    profiles: dict[str, VariableProfile],
) -> tuple["pd.DataFrame", list[DecisionCard]]:
    import pandas as pd

    cards: list[DecisionCard] = []
    out = df.copy()

    for col in out.columns:
        p = profiles.get(col)
        if p is None:
            continue
        flags = p.flags

        # Boolean-flag check FIRST — it can override an existing integer dtype
        # (the 0/1-encoded-as-int case). We coerce later in the function; here
        # we just make sure we don't early-continue past it.
        bool_flagged = any(
            "boolean stored as string" in f or "0/1 encoded" in f for f in flags
        )

        # Already in the desired type AND not flagged for boolean override — skip.
        if not bool_flagged:
            if p.detected_type == VariableType.NUMERIC and pd.api.types.is_numeric_dtype(out[col]):
                continue
            if p.detected_type == VariableType.INTEGER and pd.api.types.is_integer_dtype(out[col]):
                continue
            if p.detected_type == VariableType.DATETIME and pd.api.types.is_datetime64_any_dtype(out[col]):
                continue
            if p.detected_type == VariableType.BOOLEAN and pd.api.types.is_bool_dtype(out[col]):
                continue

        # Numeric coercion
        if any("numeric stored as string" in f for f in flags):
            before_dtype = str(out[col].dtype)
            coerced = pd.to_numeric(out[col], errors="coerce")
            n_lost = int(coerced.isna().sum() - out[col].isna().sum())
            out[col] = coerced
            cards.append(
                DecisionCard(
                    card_id=f"format_coerce_numeric__{col}",
                    stage=Stage.FORMAT,
                    variable=col,
                    issue=f"stored as {before_dtype} but values are numeric",
                    recommendation=f"coerce to numeric ({out[col].dtype})",
                    rationale="Profiler detected >=85% numeric-parseable sample.",
                    default_action="coerced",
                    metadata={
                        "before_dtype": before_dtype,
                        "after_dtype": str(out[col].dtype),
                        "n_unparseable_to_nan": n_lost,
                    },
                    status=DecisionStatus.AGENT_AUTO,
                    action_taken="coerced",
                    confirmed_by="agent_auto",
                )
            )
            continue

        # Datetime coercion
        if any("datetime stored as string" in f for f in flags):
            before_dtype = str(out[col].dtype)
            coerced = pd.to_datetime(out[col], errors="coerce", utc=False)
            n_lost = int(coerced.isna().sum() - out[col].isna().sum())
            out[col] = coerced
            cards.append(
                DecisionCard(
                    card_id=f"format_coerce_datetime__{col}",
                    stage=Stage.FORMAT,
                    variable=col,
                    issue=f"stored as {before_dtype} but values look like dates",
                    recommendation="parse to datetime64",
                    rationale="Profiler detected >=85% date-pattern matches.",
                    default_action="coerced",
                    metadata={
                        "before_dtype": before_dtype,
                        "after_dtype": str(out[col].dtype),
                        "n_unparseable_to_nat": n_lost,
                    },
                    status=DecisionStatus.AGENT_AUTO,
                    action_taken="coerced",
                    confirmed_by="agent_auto",
                )
            )
            continue

        # Boolean coercion (string yes/no/etc OR 0/1 integer)
        if bool_flagged:
            before_dtype = str(out[col].dtype)
            coerced = _coerce_boolean(out[col])
            n_lost = int(coerced.isna().sum() - out[col].isna().sum())
            out[col] = coerced
            cards.append(
                DecisionCard(
                    card_id=f"format_coerce_boolean__{col}",
                    stage=Stage.FORMAT,
                    variable=col,
                    issue=f"stored as {before_dtype} but values are boolean-like",
                    recommendation="coerce to boolean (Yes/No / True/False)",
                    rationale="Profiler detected >=85% yes/no/y/n/t/f/0/1 sample.",
                    default_action="coerced",
                    metadata={
                        "before_dtype": before_dtype,
                        "after_dtype": str(out[col].dtype),
                        "n_unrecognized_to_nan": n_lost,
                    },
                    status=DecisionStatus.AGENT_AUTO,
                    action_taken="coerced",
                    confirmed_by="agent_auto",
                )
            )
            continue

    return out, cards


def _coerce_boolean(s: "pd.Series") -> "pd.Series":
    """Map a series of yes/no/true/false/0/1 to pandas BooleanDtype (nullable)."""
    import pandas as pd

    def map_one(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return pd.NA
        if isinstance(v, bool):
            return v
        sv = str(v).strip().lower()
        if sv in _BOOL_TRUE:
            return True
        if sv in _BOOL_FALSE:
            return False
        return pd.NA

    mapped = s.map(map_one)
    return mapped.astype("boolean")


# ---------------------------------------------------------------------------
# Categorical label normalization
# ---------------------------------------------------------------------------


def _normalize_categorical_labels(
    df: "pd.DataFrame",
    profiles: dict[str, VariableProfile],
) -> tuple["pd.DataFrame", list[DecisionCard]]:
    """Collapse case- and whitespace-equivalent categorical labels.

    Applies to any object-dtype column (not just CATEGORICAL) when
    collapsing case-variants would meaningfully reduce cardinality. This
    catches the common bug where a column has labels like "M"/"m"/"F"/"f"
    that the cardinality classifier puts just over the CATEGORICAL
    threshold but that obviously should be normalized.

    Picks the most-common surface form as canonical per lowercase key.
    Only emits a card if at least one value actually changes.
    """
    import pandas as pd

    cards: list[DecisionCard] = []
    out = df.copy()

    for col in out.columns:
        p = profiles.get(col)
        if p is None:
            continue
        # Apply to CATEGORICAL and STRING columns. Skip non-object dtypes
        # (numeric/datetime/bool have no case to normalize).
        if p.detected_type not in (VariableType.CATEGORICAL, VariableType.STRING):
            continue
        s = out[col]
        if not is_text_dtype(s):
            continue
        nonnull_strs = s.dropna().astype(str).str.strip()
        if len(nonnull_strs) == 0:
            continue

        # For STRING columns, only act if case-collapse actually reduces
        # cardinality (otherwise we'd touch free-text fields uselessly).
        if p.detected_type == VariableType.STRING:
            n_raw = nonnull_strs.nunique()
            n_lc = nonnull_strs.str.lower().nunique()
            if n_lc >= n_raw:
                continue

        # Group by lowercase key, pick most-common surface form as canonical.
        grouped: dict[str, dict[str, int]] = {}
        for v in nonnull_strs:
            key = v.lower()
            grouped.setdefault(key, {})
            grouped[key][v] = grouped[key].get(v, 0) + 1

        mapping: dict[str, str] = {}
        for variants in grouped.values():
            if len(variants) <= 1:
                continue
            # Pick the most-common surface form as canonical. On a tie,
            # prefer the lexicographically smaller form — in ASCII this
            # means uppercase wins ("A" beats "a", "Yes" beats "yes"),
            # which matches user expectations for category labels.
            canonical = min(variants.keys(), key=lambda s: (-variants[s], s))
            for surface in variants:
                if surface != canonical:
                    mapping[surface] = canonical

        if not mapping:
            continue

        # Apply mapping. Don't touch NaN; preserve original dtype.
        applied = s.map(lambda v: mapping.get(v, v) if isinstance(v, str) else v)
        n_changed = int((applied != s).fillna(False).sum())
        if n_changed == 0:
            continue
        out[col] = applied

        cards.append(
            DecisionCard(
                card_id=f"format_normalize_labels__{col}",
                stage=Stage.FORMAT,
                variable=col,
                issue=(
                    f"{len(mapping)} label variant(s) collapsed "
                    f"({n_changed} value(s) changed)"
                ),
                recommendation="normalize to most-common surface form",
                rationale=(
                    "Categorical labels that differ only in case or "
                    "whitespace are treated as distinct categories by "
                    "most downstream tools, which silently inflates "
                    "cardinality."
                ),
                default_action="normalized",
                metadata={"label_mapping": mapping, "n_values_changed": n_changed},
                status=DecisionStatus.AGENT_AUTO,
                action_taken="normalized",
                confirmed_by="agent_auto",
            )
        )

    return out, cards


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def apply_renames_to_target(
    target_column: str | None,
    cards: list[DecisionCard],
) -> str | None:
    """If column-rename happened during format resolution, return the new
    name of `target_column`. Otherwise return it unchanged.

    Pass the cards returned by `resolve_formats`. Looks for the
    `format_column_names` card and consults its `renames` map.
    """
    if target_column is None:
        return None
    for c in cards:
        if c.card_id == "format_column_names":
            return c.metadata.get("renames", {}).get(target_column, target_column)
    return target_column


def _rekey_profiles(
    profiles: dict[str, VariableProfile],
    renames: dict[str, str],
) -> dict[str, VariableProfile]:
    """Return a new profiles dict keyed by post-rename column names."""
    out: dict[str, VariableProfile] = {}
    for old_name, p in profiles.items():
        new_name = renames.get(old_name, old_name)
        # Rebuild with updated name field so JSON dumps stay consistent.
        out[new_name] = VariableProfile(
            name=new_name,
            detected_type=p.detected_type,
            stored_dtype=p.stored_dtype,
            n_total=p.n_total,
            n_missing=p.n_missing,
            n_unique=p.n_unique,
            sample_values=list(p.sample_values),
            numeric=p.numeric,
            categorical=p.categorical,
            flags=list(p.flags),
        )
    return out
