# Examples

## `adult_sample.csv`

A 6,000-row sample of the **UCI Adult / Census Income** dataset, committed
here so the demo and the `/clean` Claude Code command run out of the box.

**Provenance.** Becker, B. & Kohavi, R. (1996), *Adult*, UCI Machine
Learning Repository, <https://doi.org/10.24432/C5XW20>. Distributed under
the Creative Commons Attribution 4.0 license, which permits
redistribution. Regenerate it with `python examples/fetch_adult.py`.

**Why this dataset.** It is naturally messy in exactly the ways
Distill Agent targets, so the demo exercises all six pipeline stages
without any artificial corruption:

| Stage | What it finds in `adult_sample.csv` |
|---|---|
| Profiling | 15 variables; numeric columns stored as strings; hyphenated names |
| Format resolution | leading whitespace on every categorical value; `?` missing-token; hyphen → snake_case column names; numeric/type coercion |
| Duplicates | naturally occurring exact-duplicate rows (retained by the sampler); no entity-key column, so near-duplicate detection is skipped |
| Missingness | `workclass` (~5.6%), `occupation` (~5.7%), `native-country` (~1.8%) — all encoded as `?` |
| Outliers | `capital-gain` / `capital-loss` extreme (heavy zero-inflation + coded 99999 top value); `fnlwgt` heavy-tailed |

## `quick_demo.py`

Runs the full five-stage pipeline non-interactively and prints a
stage-by-stage summary.

```bash
cd distill-agent
pip install -e ".[dev]"
python examples/quick_demo.py                        # uses adult_sample.csv
python examples/quick_demo.py path/to/other.csv      # any CSV
python examples/quick_demo.py path/to/other.csv target_col
```

## `fetch_adult.py`

Documents data provenance and regenerates `adult_sample.csv`, either via
the official `ucimlrepo` package or from a local `adult.data` file.
