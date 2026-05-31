"""Quick end-to-end demo of the Distill Agent core library.

Runs the full six-stage pipeline non-interactively on the bundled UCI
Adult sample and prints a summary of what each stage did. Useful for
sanity-checking the install before trying the Claude Code path or the
web demo.

Usage:
    cd distill-agent
    pip install -e ".[dev]"            # one-time
    python examples/quick_demo.py                       # uses adult_sample.csv
    python examples/quick_demo.py path/to/your_data.csv # or any CSV

The Adult dataset exercises every stage: missing values encoded as "?",
leading whitespace on every categorical value, hyphenated column names,
zero-inflated `capital-gain` with a coded 99999 top value, and naturally
occurring exact-duplicate rows.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from distill import (  # noqa: E402
    PipelineState,
    Session,
    Stage,
    apply_duplicate_defaults,
    apply_missingness_defaults,
    apply_outlier_defaults,
    detect_duplicates,
    detect_missingness,
    detect_outliers,
    profile,
    profile_summary_card,
)
from distill.format import apply_renames_to_target, resolve_formats  # noqa: E402
from distill.io import load  # noqa: E402
from distill.outputs import write_all_outputs  # noqa: E402

DEFAULT_DATASET = HERE / "adult_sample.csv"
DEFAULT_TARGET = "income"


def main() -> None:
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
    target = sys.argv[2] if len(sys.argv) > 2 else (
        DEFAULT_TARGET if dataset == DEFAULT_DATASET else None
    )

    print("=" * 64)
    print("Distill Agent — quick end-to-end demo")
    print(f"dataset: {dataset.name}   target: {target!r}")
    print("=" * 64)

    df0 = load(dataset)
    print(f"\nLoaded {df0.shape[0]} rows x {df0.shape[1]} cols")

    with tempfile.TemporaryDirectory() as td:
        state = PipelineState.new(input_path=dataset, df=df0.copy(), target_column=target)
        session = Session(session_id=state.session_id, root=Path(td) / "run_artifacts")

        def record(stage, before_shape, reason=""):
            """Log a StageRowCount entry for the flowchart."""
            r0, c0 = before_shape
            r1, c1 = state.df.shape
            state.record_stage_io(stage, r0, c0, r1, c1, reason)

        # 1 — Profile
        shape = state.df.shape
        profiles = profile(state.df)
        state.profiles_initial = profiles
        state.log.append(profile_summary_card(state.df, profiles))
        n_flags = sum(len(p.flags) for p in profiles.values())
        record(Stage.PROFILE, shape, "diagnostic only — no changes")
        print(f"\n[1] Profile      {len(profiles)} variables, {n_flags} advisory flags")

        # 2 — Format resolution (autonomous)
        shape = state.df.shape
        state.df, cards = resolve_formats(state.df, profiles)
        state.log.extend(cards)
        state.target_column = apply_renames_to_target(state.target_column, cards)
        record(Stage.FORMAT, shape, "mechanical fixes (types, whitespace, names)")
        print(f"[2] Format       {len(cards)} mechanical fixes")
        for c in cards[:6]:
            print(f"      - {c.issue}")
        if len(cards) > 6:
            print(f"      ... and {len(cards) - 6} more")

        profiles = profile(state.df)  # refresh after format changes

        # 3 — Missingness
        shape = state.df.shape
        miss = detect_missingness(state.df, profiles)
        state.df = apply_missingness_defaults(state.df, miss)
        state.log.extend(miss)
        record(Stage.MISSINGNESS, shape, "rows/variables dropped or imputed")
        print(f"[3] Missingness  {len(miss)} variable(s) handled")
        for c in miss:
            print(f"      - {c.variable}: {c.action_taken}  ({c.metadata['pattern']})")

        # 4 — Outliers
        shape = state.df.shape
        out_cards = detect_outliers(state.df, profiles)
        state.df = apply_outlier_defaults(state.df, out_cards)
        state.log.extend(out_cards)
        record(Stage.OUTLIER, shape, "flagged outliers handled")
        print(f"[4] Outliers     {len(out_cards)} numeric column(s) flagged")
        for c in out_cards:
            print(f"      - {c.variable}: severity={c.metadata['severity']}, "
                  f"action={c.action_taken}")

        # 5 — Duplicates
        shape = state.df.shape
        dup = detect_duplicates(state.df, profiles)
        state.df = apply_duplicate_defaults(state.df, dup)
        state.log.extend(dup)
        record(Stage.DUPLICATE, shape, "exact duplicate rows removed")
        print(f"[5] Duplicates   {len(dup)} card(s)")
        for c in dup:
            print(f"      - {c.card_id}: {c.action_taken or c.status.value}")

        # Outputs
        result = write_all_outputs(state, session)
        print()
        print("=" * 64)
        print(f"rows: {df0.shape[0]} -> {state.df.shape[0]}    "
              f"cols: {df0.shape[1]} -> {state.df.shape[1]}    "
              f"decisions logged: {len(state.log)}")
        print("artifacts written:")
        for k, v in result["artifacts"].items():
            print(f"  {k}: {v}")
        if result["errors"]:
            print("writer errors:")
            for k, v in result["errors"].items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
