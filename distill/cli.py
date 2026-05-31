"""Command-line interface for Distill Agent.

    distill clean data.csv --target income
    distill profile data.csv

`clean` runs the full six-stage pipeline non-interactively (every judgment
decision uses the agent's default) and writes the five output artifacts.
It is the entry point the Claude Code `/clean` command falls back to for
an unattended run, and the one a CI job would call.

`profile` runs only the diagnostic profiling pass and prints the result.

Implemented with stdlib `argparse` only — no third-party CLI framework —
so the command works the moment the package is installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_clean(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"error: no such file: {input_path}", file=sys.stderr)
        return 1

    print(f"Distill Agent — cleaning {input_path.name}")

    def on_event(name: str, payload: dict) -> None:
        if name == "stage_done":
            print(f"  [done] {payload.get('stage', '')}")

    result = run_pipeline(
        input_path,
        target_column=args.target,
        out_root=Path(args.out),
        session_id=args.session,
        on_event=on_event,
    )
    s = result.summary
    print()
    print(f"Session : {s['session_id']}")
    print(f"Rows    : {s['rows_in']:,} -> {s['rows_out']:,}")
    print(f"Columns : {s['cols_in']} -> {s['cols_out']}")
    print(f"Decisions: {s['n_decisions']} logged, {s['n_overrides']} override(s)")
    print("Artifacts:")
    for k, v in result.artifacts.items():
        if isinstance(v, dict):
            for fmt, path in v.items():
                print(f"  {fmt:14s} {path}")
        else:
            print(f"  {k:14s} {v}")
    if result.errors:
        print("Writer warnings:")
        for k, v in result.errors.items():
            print(f"  {k}: {v}")
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    from .io import load
    from .profiler import profile as run_profile

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"error: no such file: {input_path}", file=sys.stderr)
        return 1

    df = load(input_path)
    profiles = run_profile(df)
    print(f"Profile — {input_path.name} ({len(df):,} rows x {len(df.columns)} cols)")
    print(f"{'variable':24s} {'type':12s} {'missing':>8s} {'unique':>9s}  flags")
    print("-" * 78)
    for name, p in profiles.items():
        flags = "; ".join(p.flags) if p.flags else ""
        print(
            f"{str(name)[:24]:24s} {p.detected_type.value:12s} "
            f"{p.missing_rate * 100:7.1f}% {p.n_unique:9,d}  {flags}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="distill",
        description="Distill Agent — AI-assisted reproducible data cleaning.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_clean = sub.add_parser(
        "clean", help="Run the full cleaning pipeline and write all artifacts."
    )
    p_clean.add_argument("input_path", help="Dataset to clean (CSV/Excel/SPSS).")
    p_clean.add_argument(
        "-t", "--target", default=None,
        help="Outcome variable (recorded for downstream use).",
    )
    p_clean.add_argument(
        "-o", "--out", default="run_artifacts", help="Output root directory.",
    )
    p_clean.add_argument(
        "-s", "--session", default=None, help="Explicit session id.",
    )
    p_clean.set_defaults(func=_cmd_clean)

    p_profile = sub.add_parser(
        "profile", help="Profile a dataset (diagnostic only)."
    )
    p_profile.add_argument("input_path", help="Dataset to profile.")
    p_profile.set_defaults(func=_cmd_profile)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
