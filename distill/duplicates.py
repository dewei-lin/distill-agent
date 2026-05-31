"""Stage 5 — Duplicate and near-duplicate detection.

Two kinds of duplicates, handled differently:

  * **Exact duplicates** — rows identical across all non-identifier
    columns. Surfaced as a single summary DecisionCard with the count and
    a blanket confirm/override action. Default: remove (keep first).

  * **Near-duplicates** — defined relative to an *entity key*. If the
    dataset has an id-like column, two rows that share the same key value
    but differ in other fields are a near-duplicate: the same entity
    recorded twice with conflicting data. Each colliding key value gets
    its own card.

Why entity-key-based rather than "rows differing in <= N fields"? On real
data, a field-difference rule floods: thousands of *distinct* people
legitimately share most demographic attributes, so "rows differing in 2
fields" flags enormous numbers of non-duplicates. A near-duplicate is only
meaningful when you can say "this is the SAME entity twice" — and that
requires a key. With no key column, near-duplicate detection is skipped
with an explicit logged note.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .types import (
    DecisionCard,
    DecisionStatus,
    Stage,
    VariableProfile,
)

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Maximum number of near-duplicate (key-collision) clusters to surface
# individually; beyond this, a single overflow card summarizes the rest.
MAX_CLUSTERS_TO_SURFACE = 50

# Name tokens that mark a column as an entity key. Deliberately tight:
# "num"/"no" were excluded because they false-match columns like
# "education_num" or "num_visits" that are ordinary measurements.
_ID_NAME_TOKENS = frozenset(
    {"id", "key", "uuid", "guid", "identifier", "idx"}
)

EXACT_DUPLICATE_ACTIONS = [
    "remove_all_duplicates",     # keep one per group (recommended)
    "keep_all_duplicates",       # leave as-is
    "drop_all_duplicate_rows",   # remove EVERY copy, including the first
]

NEAR_DUPLICATE_ACTIONS = [
    "inspect",           # surface but don't transform (recommended)
    "merge_keep_first",  # keep the first member of each cluster
    "merge_keep_last",   # keep the last member
    "keep_all",          # leave as-is
]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def detect_duplicates(
    df: "pd.DataFrame",
    profiles: dict[str, VariableProfile] | None = None,
) -> list[DecisionCard]:
    """Return decision cards for exact duplicates + entity-key collisions.

    Always returns a card for exact duplicates if any are found. Near-
    duplicate cards are produced only when an entity-key column exists; if
    none does, a single AGENT_AUTO card records that the near-duplicate
    check was skipped.
    """
    cards: list[DecisionCard] = []

    # Identifier-looking columns are excluded from the EXACT-duplicate key
    # (otherwise no two rows ever match on a unique id).
    excluded = _all_unique_identifier_columns(df, profiles)
    key_cols = [c for c in df.columns if c not in excluded]

    # ----- exact duplicates (needs at least one non-identifier column) -----
    if key_cols:
        df_key = df[key_cols]
        duped_mask = df_key.duplicated(keep=False)
        n_exact_dupe_rows = int(duped_mask.sum())
        if n_exact_dupe_rows > 0:
            n_unique_groups = int(df_key[duped_mask].drop_duplicates().shape[0])
            n_to_remove = n_exact_dupe_rows - n_unique_groups
            cards.append(
                DecisionCard(
                    card_id="duplicates_exact",
                    stage=Stage.DUPLICATE,
                    variable=None,
                    issue=(
                        f"{n_exact_dupe_rows} row(s) form {n_unique_groups} "
                        f"exact-duplicate group(s); removing copies drops "
                        f"{n_to_remove} row(s)."
                    ),
                    recommendation="remove duplicate copies, keep one row per group",
                    rationale=(
                        "Exact duplicates almost always indicate a data-load "
                        "error. Keeping one representative per group preserves "
                        "the unique observations."
                    ),
                    default_action="remove_all_duplicates",
                    alternatives=[a for a in EXACT_DUPLICATE_ACTIONS if a != "remove_all_duplicates"],
                    metadata={
                        "key_columns": key_cols,
                        "excluded_columns": excluded,
                        "n_duplicate_rows": n_exact_dupe_rows,
                        "n_groups": n_unique_groups,
                        "n_to_remove": n_to_remove,
                    },
                )
            )

    # ----- near duplicates: entity-key collisions -----
    entity_keys = _entity_key_columns(df)
    if not entity_keys:
        cards.append(
            DecisionCard(
                card_id="duplicates_near_skipped",
                stage=Stage.DUPLICATE,
                variable=None,
                issue="No entity-key column found",
                recommendation="skip near-duplicate detection",
                rationale=(
                    "Near-duplicate detection requires an id-like key "
                    "column to define what counts as 'the same entity "
                    "twice'. None was found, so only exact-duplicate "
                    "detection ran. (A field-difference rule would flood "
                    "with false positives on data where distinct rows "
                    "legitimately share most attributes.)"
                ),
                default_action="skip",
                status=DecisionStatus.AGENT_AUTO,
                action_taken="skip",
                confirmed_by="agent_auto",
                metadata={"reason": "no entity-key column"},
            )
        )
        return cards

    # Use the first entity-key column; flag every value that repeats.
    key = entity_keys[0]
    key_series = df[key]
    collide_mask = key_series.duplicated(keep=False) & key_series.notna()
    if collide_mask.sum() == 0:
        cards.append(
            DecisionCard(
                card_id="duplicates_near_clear",
                stage=Stage.DUPLICATE,
                variable=key,
                issue=f"Entity key '{key}' has no repeated values",
                recommendation="proceed",
                rationale=(
                    f"Every value of the entity key '{key}' is unique, so "
                    f"there are no same-entity near-duplicate records."
                ),
                default_action="proceed",
                status=DecisionStatus.AGENT_AUTO,
                action_taken="proceed",
                confirmed_by="agent_auto",
                metadata={"entity_key": key},
            )
        )
        return cards

    other_cols = [c for c in df.columns if c != key]
    groups = list(df[collide_mask].groupby(key, observed=True))

    # If the majority of key values collide, this is almost certainly a
    # panel/group dimension (time period, category) shared across many rows,
    # not a unique entity key.  Surfacing one card per value would flood the
    # UI and the audit trail with meaningless decisions.
    n_unique_key = int(key_series.nunique())
    collision_rate = len(groups) / max(n_unique_key, 1)
    if collision_rate > 0.5:
        cards.append(
            DecisionCard(
                card_id="duplicates_near_panel_key",
                stage=Stage.DUPLICATE,
                variable=key,
                issue=(
                    f"'{key}' repeats on {len(groups):,} of {n_unique_key:,} "
                    f"unique values ({collision_rate:.0%}). This looks like a "
                    f"panel or group identifier (e.g. time period shared "
                    f"across jurisdictions/categories), not a unique entity key."
                ),
                recommendation=(
                    "Skip near-duplicate detection — the repeats are "
                    "structural, not data errors."
                ),
                rationale=(
                    "When most key values collide, the column is a panel "
                    "dimension rather than a unique record identifier. "
                    "Near-duplicate detection is designed for unique-key "
                    "violations, not repeated panel indices."
                ),
                default_action="skip",
                status=DecisionStatus.AGENT_AUTO,
                action_taken="skip",
                confirmed_by="agent_auto",
                metadata={
                    "entity_key": key,
                    "collision_rate": round(collision_rate, 3),
                    "n_colliding": len(groups),
                    "n_unique": n_unique_key,
                },
            )
        )
        return cards

    for i, (key_val, grp) in enumerate(groups[:MAX_CLUSTERS_TO_SURFACE]):
        sub = grp[other_cols].astype(str)
        diff_cols = [c for c in other_cols if sub[c].nunique() > 1]
        cards.append(
            DecisionCard(
                card_id=f"duplicates_near_{i + 1:03d}",
                stage=Stage.DUPLICATE,
                variable=key,
                issue=(
                    f"Entity key '{key}'={key_val!r} appears in "
                    f"{len(grp)} rows"
                    + (f", differing in: {', '.join(diff_cols)}" if diff_cols
                       else " with identical other fields")
                ),
                recommendation="inspect the conflicting records before deciding",
                rationale=(
                    "The same entity key appears more than once. This is "
                    "either a duplicate entry or a key-assignment error; "
                    "domain knowledge is needed to pick the right record."
                ),
                default_action="inspect",
                alternatives=[a for a in NEAR_DUPLICATE_ACTIONS if a != "inspect"],
                metadata={
                    "entity_key": key,
                    "key_value": str(key_val),
                    "row_indices": [int(ix) for ix in grp.index.tolist()],
                    "differing_columns": diff_cols,
                    "cluster_size": len(grp),
                },
            )
        )

    if len(groups) > MAX_CLUSTERS_TO_SURFACE:
        cards.append(
            DecisionCard(
                card_id="duplicates_near_overflow",
                stage=Stage.DUPLICATE,
                variable=key,
                issue=(
                    f"{len(groups) - MAX_CLUSTERS_TO_SURFACE} more entity-key "
                    f"collision(s) not surfaced individually."
                ),
                recommendation="review the surfaced cards or handle in bulk",
                rationale=(
                    f"'{key}' has {len(groups)} colliding values; surfacing "
                    f"all of them would flood the audit trail. The first "
                    f"{MAX_CLUSTERS_TO_SURFACE} are shown individually."
                ),
                default_action="inspect",
                alternatives=["keep_all"],
                metadata={
                    "entity_key": key,
                    "n_unsurfaced": len(groups) - MAX_CLUSTERS_TO_SURFACE,
                },
            )
        )

    return cards


def _all_unique_identifier_columns(
    df: "pd.DataFrame",
    profiles: dict[str, VariableProfile] | None,
) -> list[str]:
    """Columns the profiler flagged as all-unique identifiers — excluded
    from the EXACT-duplicate key (a unique id makes every row distinct)."""
    if profiles is None:
        return []
    return [
        col
        for col, p in profiles.items()
        if any("all-unique (possible identifier)" in f for f in p.flags)
    ]


def _entity_key_columns(df: "pd.DataFrame") -> list[str]:
    """Columns whose NAME marks them as an entity key (id / key / code /
    ...). Name-based only — deliberately conservative so we never treat a
    high-cardinality measurement as a key."""
    keys: list[str] = []
    for col in df.columns:
        lc = str(col).lower()
        parts = re.split(r"[^a-z0-9]+", lc)
        if any(t in _ID_NAME_TOKENS for t in parts) or lc.endswith("id"):
            keys.append(col)
    return keys


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def apply_decision(
    df: "pd.DataFrame",
    card: DecisionCard,
) -> tuple["pd.DataFrame", dict[str, Any]]:
    """Apply a resolved duplicates DecisionCard."""
    action = card.action_taken or card.default_action
    out = df.copy()
    info: dict[str, Any] = {"applied": action, "card_id": card.card_id}

    # No-op cards (skipped / clear / overflow informational).
    if card.card_id in {"duplicates_near_skipped", "duplicates_near_clear"}:
        info["rows_dropped"] = 0
        return out, info

    if card.card_id == "duplicates_exact":
        key_cols = card.metadata.get("key_columns") or list(df.columns)
        if action == "remove_all_duplicates":
            before = len(out)
            out = out.drop_duplicates(subset=key_cols, keep="first")
            info["rows_dropped"] = before - len(out)
        elif action == "keep_all_duplicates":
            info["rows_dropped"] = 0
        elif action == "drop_all_duplicate_rows":
            before = len(out)
            duped = out.duplicated(subset=key_cols, keep=False)
            out = out[~duped]
            info["rows_dropped"] = before - len(out)
        else:
            raise ValueError(f"Unknown exact-duplicate action: {action!r}")
        return out, info

    # Near-duplicate (entity-key collision) card.
    if action in ("inspect", "keep_all"):
        info["rows_dropped"] = 0
        return out, info

    rows_idx = [i for i in card.metadata.get("row_indices", []) if i in out.index]
    if not rows_idx:
        info["rows_dropped"] = 0
        return out, info

    if action == "merge_keep_first":
        to_drop = rows_idx[1:]
        out = out.drop(index=to_drop)
        info["rows_dropped"] = len(to_drop)
        return out, info
    if action == "merge_keep_last":
        to_drop = rows_idx[:-1]
        out = out.drop(index=to_drop)
        info["rows_dropped"] = len(to_drop)
        return out, info

    raise ValueError(f"Unknown near-duplicate action: {action!r}")


def apply_default_for_all(
    df: "pd.DataFrame",
    cards: list[DecisionCard],
) -> "pd.DataFrame":
    """Resolve and apply every card with its default action (non-interactive)."""
    out = df
    for card in cards:
        if card.status != DecisionStatus.PENDING:
            continue
        card.resolve(
            action=card.default_action,
            confirmed_by="agent_default",
            status=DecisionStatus.AGENT_DEFAULT,
        )
        out, info = apply_decision(out, card)
        card.metadata["applied_info"] = info
    return out
