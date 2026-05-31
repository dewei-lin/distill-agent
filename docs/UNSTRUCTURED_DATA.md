# Unstructured Data Guide — Distill Agent

This document is the **authoritative reference** the agent reads when a dataset
includes non-tabular companions. It covers:

1. Taxonomy of unstructured data types
2. Input combination matrix
3. Storage format decisions
4. Preprocessing pipelines per type
5. Code-generation prompt template
6. Reproducibility strategy for unstructured pipelines

---

## 1. Taxonomy

| Type | Subtypes | Raw formats | Key ML property |
|---|---|---|---|
| **Image** | natural photo, medical scan, map, chart | PNG, JPEG, TIFF, DICOM, SVG | spatial locality; needs resize + normalize |
| **Text** | clinical note, report, social post, OCR output | plain text, HTML, PDF, DOCX | variable length; needs tokenization |
| **Audio** | speech, recording, EEG waveform | WAV, MP3, FLAC | temporal; needs spectrogram or waveform tensor |
| **Video** | procedure recording, surveillance | MP4, AVI, MOV | spatial + temporal; frame sampling required |
| **Time series** | sensor stream, vitals, financial tick | CSV sequences, HDF5, Parquet | ordered; needs windowing / padding |
| **Document** | PDF mixing text + tables + images | PDF, DOCX | multi-modal internally; needs parser |

> **Working rule for the agent:** at intake, identify which type(s) each companion
> file belongs to by extension. Unknown extensions → log a warning, treat as binary
> blob, skip processing.

---

## 2. Input Combination Matrix

Each row = a dataset configuration. The agent checks which row applies and
selects the corresponding storage format and code template.

| ID | Structured | Companions | Storage target | Notes |
|---|---|---|---|---|
| `S` | CSV/Excel/SPSS | — | Parquet | baseline, no change |
| `S+I` | CSV | image per row | HDF5 or HF Dataset | most common multimodal |
| `S+T` | CSV | text file per row | HF Dataset | or inline text column |
| `S+I+T` | CSV | image + text per row | HF Dataset | two companion types |
| `S+A` | CSV | audio per row | HDF5 (spectrogram arrays) | audio needs heavier preprocessing |
| `S+V` | CSV | video per row | frame-sampled HDF5 | extract N frames at intake |
| `S+D` | CSV | PDF document per row | HF Dataset (text) + image HDF5 | parse PDF → text + page images |
| `I` | — | image collection | HF Dataset or folder | no tabular anchor |
| `T` | — | text collection | HF Dataset | standard NLP |

**Inline text** (notes as a column in the CSV, not a separate file) is always
combination `S` from the pipeline's perspective — the column is profiled,
cleaned, and passed through as a string. The ML code then tokenizes it.

**Split variants** — any combination above can be prefixed with `split_` to
indicate the dataset has train/val/test partitions:

| ID | Structure | Script entry point | Output |
|---|---|---|---|
| `split_S` | train CSV + val CSV | `build_datasets(train_dir, val_dir)` | `train_clean.parquet`, `val_clean.parquet` |
| `split_S+I` | train CSV + val CSV + images in each | `build_datasets(train_dir, val_dir)` | `train_dataset/`, `val_dataset/` |
| `split_S+I+T` | train CSV + val CSV + images + inline text | `build_datasets(train_dir, val_dir)` | `train_dataset/`, `val_dataset/` |

**Expand-via-category** — a special sub-pattern within split datasets where a
separate target CSV contains extra categorical key columns, causing the row count
to multiply after joining:

```
covariates.csv   : (period_id, jurisdiction)             → N rows
target.csv       : (period_id, jurisdiction, category)   → N × k rows  (k categories)
joined           : (period_id, jurisdiction, category)   → N × k rows
                   companions (images) keyed on (jurisdiction, period_id)
                   → one image shared across k rows per (period_id, jurisdiction)
```

Detection rule: if CSV_B has more rows than CSV_A and shares a subset of CSV_A's
key columns plus at least one additional categorical column, classify as
expand-via-category. Surface as an intake observation, not a decision card.

---

## 3. Storage Format Decisions

### Decision tree

```
Is the dataset > 50 GB?
├── Yes → WebDataset (.tar shards)   # streaming, cloud-native
└── No
    ├── Any audio or video companions?
    │   └── Yes → HDF5              # stores arrays efficiently
    └── No
        ├── Image or text companions?
        │   └── Yes → HuggingFace Dataset (save_to_disk)
        │             + Parquet for tabular side
        └── No (tabular only)
            └── Parquet
```

### Format properties

| Format | Self-contained | Lazy load | Cloud-friendly | Framework support |
|---|---|---|---|---|
| **Parquet** | yes (tabular only) | yes | yes (S3, GCS) | pandas, Spark, DuckDB |
| **HDF5** | yes | yes (chunked) | limited | h5py, PyTorch, R |
| **HF Dataset** | yes | yes (Arrow mmap) | yes (Hub) | HF Transformers, PyTorch |
| **WebDataset** | no (shards) | yes (streaming) | yes | PyTorch, FFCV |
| **LMDB** | yes | yes | no | custom, used in CV benchmarks |
| **Zarr** | yes | yes | yes | xarray, Dask |

### Recommendation per combination

| Combination | Primary artifact | Tabular side |
|---|---|---|
| `S` | `clean.parquet` | same |
| `S+I` | `clean_dataset/` (HF Dataset, Image feature) | embedded in dataset |
| `S+T` | `clean_dataset/` (HF Dataset, string feature) | embedded in dataset |
| `S+I+T` | `clean_dataset/` (HF Dataset, Image + string features) | embedded |
| `S+A` | `clean_dataset/` (HF Dataset, Audio feature) | embedded |
| `S+V` | `clean.h5` (frames as array datasets) + `clean.parquet` | Parquet |
| `S+D` | `clean_dataset/` (text column) + `clean.h5` (page images) | embedded |

---

## 4. Preprocessing Pipelines

### 4.1 Image

```
raw PNG/JPEG
    → decode to RGB (PIL / torchvision)
    → resize to target (default 224×224)
    → normalize: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]  (ImageNet)
    → store as float32 array  OR  store raw bytes in HF Dataset (decode at train time)
```

**Agent decision:** if the downstream model is known (e.g. ViT, ResNet), emit
the matching normalization constants. Otherwise use ImageNet defaults and note
the assumption in `report.md`.

**Reproducibility checksum:** SHA-256 of the raw file bytes stored in
`companion_checksums.json` at session time. Re-running the script recomputes and
compares. This proves the source files are unchanged without re-encoding.

### 4.2 Text (companion file)

```
raw .txt / .pdf / .html
    → extract plain text (pdfminer, BeautifulSoup, etc.)
    → normalize whitespace, fix encoding (UTF-8)
    → store as string column in HF Dataset
    → (tokenization is NOT done here — it is model-specific)
```

**Agent decision:** tokenization belongs in the training loop, not the cleaning
pipeline. The cleaning pipeline only ensures the text is clean UTF-8 strings.

**Reproducibility checksum:** SHA-256 of the extracted plain text string (after
whitespace normalization), stored per-row in `companion_checksums.json`.

### 4.3 Inline text (column in CSV)

Same as tabular missingness / format resolution. No special handling needed.
The agent profiles it as a `STRING` variable, flags high cardinality, and leaves
content unchanged. Text-specific cleaning (stopwords, stemming) is out of scope
for the data cleaning pipeline.

### 4.4 Audio

```
raw WAV/MP3
    → decode to mono float32 waveform (librosa / torchaudio)
    → resample to 16 kHz (standard for speech models)
    → store raw waveform OR compute log-mel spectrogram
    → HF Dataset with Audio feature (stores raw waveform, decodes at train time)
```

### 4.5 Video

```
raw MP4/AVI
    → sample N frames uniformly (default N=8)
    → per frame: same pipeline as 4.1 Image
    → store as (N, C, H, W) array in HDF5
```

Video is always HDF5, not HF Dataset, because the HF Audio/Image features are
single-file per row.

---

## 5. Code-Generation Prompt Template

When the agent needs to emit companion-handling code for a new session, it should
be prompted with:

```
You are generating a section of clean_script.py for a Distill Agent session.

Dataset configuration: {combination_id}   # e.g. "S+I+T"
Companion types      : {list of (extension, preprocessing_pipeline_id)}
  e.g. [(".png", "image_224_imagenet"), (".txt", "text_utf8")]
Key derivation rule  : {template}          # e.g. "{patient_id}_{visit_date}"
Key columns          : {list of columns}   # e.g. ["patient_id", "visit_date"]
Storage target       : {format}            # e.g. "HF Dataset"
Target column        : {col or None}

Using the preprocessing pipelines from UNSTRUCTURED_DATA.md §4, emit:

1. _companion_row_keys(df) — derives the stem from the key columns
2. match_companions(df, companion_dir, extensions=[...]) — resolves one
   path column per extension, adds has_<type> boolean, saves checksums
3. to_hf_dataset(df) — builds the HF Dataset with correct feature types
4. to_torch_dataset(df, feature_cols, target_col) — PyTorch Dataset wrapper

Constraints:
- No heavy deps at import time (lazy import inside functions)
- All random seeds fixed at 42
- Checksums written to companion_checksums.json alongside clean.csv
- **NEVER emit a companion_manifest.json dependency.** Companion filenames are
  fully determined by the key columns and a fixed extension. Derive paths
  directly: `companion_dir / f"{key}.png"`. The manifest is redundant and
  breaks the script when the sidecar file is absent.
- **NEVER use `.isin()` against a hardcoded set of IDs loaded from a JSON
  file** to set `has_companion`. Compute `has_companion` at runtime by
  checking whether the derived path exists on disk inside `match_companions()`.
```

The agent fills in the bracketed values from the session's intake metadata and
calls this prompt to produce the companion-handling block.

### Canonical companion-handling pattern

Every generated script must follow this pattern exactly. Do not deviate.

```python
# CORRECT — path derived directly from key columns, existence checked at runtime
def _companion_row_keys(df):
    return df['{key_col_1}'].fillna("").astype(str) + '_' + df['{key_col_2}'].fillna("").astype(str)

def match_companions(df, companion_dir, *, extensions=None):
    if extensions is None:
        extensions = ['.png']
    companion_dir = Path(companion_dir)
    keys = _companion_row_keys(df)
    out = df.copy()
    for ext in extensions:
        paths = [str(p) if (p := companion_dir / f"{k}{ext}").exists() else None for k in keys]
        out[f"{_ext_to_prefix(ext)}_path"] = paths
        out[f"has_{_ext_to_prefix(ext)}"] = [p is not None for p in paths]
    return out

# WRONG — never do this
import json
_MANIFEST = json.load(open('companion_manifest.json'))          # ← forbidden
_IDS = frozenset(_MANIFEST)                                     # ← forbidden
df['has_companion'] = _companion_row_keys(df).isin(_IDS)        # ← forbidden
```

---

## 6. Reproducibility Strategy for Unstructured Data

### The gap

The tabular reproducibility check (run `clean_script.py` → compare `clean.csv`)
only verifies deterministic transformations on the tabular side. Companion files
are referenced by path but never processed during the check. Two failure modes
go undetected:

1. The companion files on disk were replaced / corrupted between sessions.
2. The preprocessing code (resize, normalize, tokenize) produces different output
   on replay (e.g. due to a library version change).

### Solution: two-layer verification

**Layer 1 — source integrity (lightweight)**

At session time, `match_companions` computes the SHA-256 of the raw bytes of
every companion file it resolves, and writes:

```json
// companion_checksums.json
{
  "AK_0Un18Xny": {
    "path": "/data/pngs/AK_0Un18Xny.png",
    "sha256": "3a7f...",
    "type": "image",
    "size_bytes": 48210
  },
  ...
}
```

On replay, `clean_script.py` re-reads each companion at the recorded path,
recomputes SHA-256, and reports any mismatch before proceeding. This proves the
source files are unchanged without re-running any ML preprocessing.

**Layer 2 — processed output integrity (heavier, opt-in)**

For full end-to-end reproducibility, `write_all_outputs()` should also save the
processed companion representations:

| Combination | Processed artifact | How to compare |
|---|---|---|
| `S+I` | `clean_dataset/` (HF Dataset) | compare `dataset_info.json` + 10 random sample hashes |
| `S+T` | `clean_dataset/` | same |
| `S+I+T` | `clean_dataset/` | same |
| `S+A` | `clean_dataset/` | same |
| `S+V` | `clean.h5` (frame arrays) | compare shape + per-dataset checksum via `h5py` |

Layer 2 verifies that the preprocessing code itself is deterministic across
re-runs (same library versions, same random seeds).

### Implementation checklist for the agent

When emitting code for a multimodal session the agent MUST:

- [ ] Generate `match_companions()` that writes `companion_checksums.json`
- [ ] Generate `verify_companions(checksum_path, companion_dir)` that re-checks
      SHA-256 on replay and raises `RuntimeError` on mismatch
- [ ] Call `verify_companions()` near the top of `if __name__ == "__main__"`
      before any processing begins
- [ ] Save the processed artifact (`clean_dataset/` or `clean.h5`) in
      `write_all_outputs()`
- [ ] Report Layer 1 and Layer 2 results separately in `report.md`

### Boundary: what reproducibility does NOT cover

Reproducibility of the cleaning script guarantees:
- Same tabular transformations on same input CSV → same `clean.csv`
- Same companion files (verified by SHA-256) → same processed artifact

It does NOT guarantee:
- Model training is reproducible (that is the training code's responsibility)
- Companion files are semantically correct (that is the analyst's responsibility)
- Image content is clinically/scientifically valid (out of scope)

---

## 7. Split Dataset Handling

### 7.1 Detection and classification at intake

The agent inspects the uploaded files and classifies the dataset structure
before asking any questions. Classification is based on:

| Signal | Classification |
|---|---|
| Single CSV | Standard (`S`) |
| ZIP or directory with `train/` and `val/` subdirectories | Split (`split_*`) |
| Two CSVs with identical column schemas | Split — confirm with analyst |
| Two CSVs where B has more rows than A and shares a key subset | Expand-via-category — confirm |
| Two CSVs with disjoint schemas sharing only key columns | Join — confirm join keys |

### 7.2 The join-then-expand pattern

When a covariate CSV and a target CSV are present:

```
Step 1 — load both:
  covariates.csv  (N rows, keyed on covariate_keys)
  target.csv      (N×k rows, keyed on covariate_keys + [category_col])

Step 2 — join:
  merged = target.csv.merge(covariates.csv, on=covariate_keys, how='left')
  # result: N×k rows with all covariate columns + target column

Step 3 — companion matching:
  companion key derived from covariate_keys only (not category_col)
  one companion file is matched per covariate_keys group
  the same companion appears k times in the merged dataset (once per category)
  this is correct — the companion is a group-level feature

Step 4 — clean:
  all tabular cleaning operates on the merged dataset
  target column is guarded with `if col in df.columns`
  category_col is treated as a categorical feature, not cleaned further
```

### 7.3 Fit-on-train, transform-val contract

Every stateful transform saves its parameters to `fitted_params.json`:

```json
{
  "imputers": {
    "temp_avg_f":   {"method": "median", "value": 54.1},
    "precip_in":    {"method": "mice",   "feature_cols": [...], "max_iter": 10, "random_state": 42}
  },
  "scalers": {
    "labor_force":  {"method": "standard", "mean": 3200000.0, "std": 2800000.0}
  },
  "encoders": {
    "state_name":   {"method": "frequency", "table": {"CA": 0.08, "TX": 0.07, ...}}
  },
  "binners": {
    "rate_per_10000_ed_visits": {"method": "quantile", "edges": [0.05, 1.68, 3.21, 7.54, 79.99]}
  }
}
```

The generated `build_datasets()` function:
1. Cleans and fits transforms on `train_dir` → returns fitted params as an
   in-memory dict AND writes `fitted_params.json` as an output artifact
2. Passes the in-memory dict directly to the val transform — does NOT read
   `fitted_params.json` back from disk
3. Applies (`.transform()` only) to `val_dir` using the in-memory params
4. Returns `(train_dataset, val_dataset)`

`fitted_params.json` is an **output** for auditing and reproducibility checks.
It is never a required **input**. The script must run correctly even if
`fitted_params.json` does not exist before the script starts.

### 7.4 Code-generation template for split sessions

When generating `clean_script.py` for a split session, the agent uses this
extended template:

```
Dataset configuration : {split_combination_id}   # e.g. "split_S+I+T"
Train directory       : {train_dir}              # e.g. "data/train/"
Val directory         : {val_dir}                # e.g. "data/val/"
Covariate file        : {covariate_filename}     # e.g. "covariates.csv"
Target file           : {target_filename or None}# e.g. "dose_sys_train.csv"
Join keys             : {list}                   # e.g. ["period_id", "jurisdiction"]
Category column       : {col or None}            # e.g. "overdose_category"
Target column         : {col or None}            # e.g. "rate_per_10000_ed_visits"
Companion type        : {combination_id}         # e.g. "S+I+T"
Companion key template: {template}               # e.g. "{jurisdiction}_{period_id}"
Companion extensions  : {list}                   # e.g. [".png"]
Storage target        : {format}                 # e.g. "HF Dataset"

Emit:
1. load_and_join(covariate_csv, target_csv, join_keys, category_col)
   → pd.DataFrame (merged, N×k rows)
2. clean_tabular(df, fitted_params=None)
   → (pd.DataFrame, dict)  # dict = fitted_params when fitted_params is None
                            # dict = same fitted_params (unchanged) when passed in
3. match_companions(df, companion_dir, extensions=[...])
   → pd.DataFrame with image_path / text_content / etc. columns
   → writes companion_checksums.json
4. to_hf_dataset(df, split="train"|"val")
   → datasets.Dataset
5. build_datasets(train_dir, val_dir)
   → (train_dataset: datasets.Dataset, val_dataset: datasets.Dataset)
   → writes fitted_params.json, train_dataset/, val_dataset/,
             companion_checksums.json, clean.csv (train tabular only)

if __name__ == "__main__":
   build_datasets(sys.argv[1], sys.argv[2])
   verify_companions(...)
   print reproducibility report
```

### 7.5 Output artifacts for split sessions

| Artifact | When produced | Contents |
|---|---|---|
| `train_dataset/` | always for split sessions | HF Dataset: tabular + image + text + target |
| `val_dataset/` | always for split sessions | HF Dataset: tabular + image + text, no target |
| `fitted_params.json` | always for split sessions | all stateful transform parameters |
| `clean.csv` | always | train tabular only (no companions), for quick inspection |
| `companion_checksums.json` | when companions present | SHA-256 per companion file |
| `report.md` / `report.pdf` | always | includes note on which params were fitted |
| `decisions.json` | always | full decision log including fitted_params |
| `flowchart.svg` | always | CONSORT flow for rows included/excluded per split |
| `clean_script.py` | always | standalone, calls `build_datasets()` |
