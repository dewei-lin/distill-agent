"""patch_script.py — fix any Distill Agent clean_script.py in-place.

Removes companion_manifest.json dependency, fixes match_companions() to use
direct path construction, and wires __main__ to write HF Dataset output.

Usage:
    python patch_script.py clean_script.py          # patches in-place
    python patch_script.py clean_script.py out.py   # writes to new file
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Replacement blocks
# ---------------------------------------------------------------------------

_COMPANION_BLOCK = '''\
# ============================================================
# Companion files — filename rule: '{jurisdiction}_{period_id}.png'
# Paths are derived directly from key columns at runtime — no sidecar needed.
# ============================================================

def _companion_row_keys(df: pd.DataFrame) -> pd.Series:
    """Derive the companion match key for each row."""
    return df['jurisdiction'].fillna("").astype(str) + '_' + df['period_id'].fillna("").astype(str)


def _find_companion_dir(root: Path) -> 'Path | None':
    """Return the directory under root containing the most PNG files."""
    from collections import Counter
    pngs = list(root.rglob('*.png'))
    if not pngs:
        return None
    return Counter(p.parent for p in pngs).most_common(1)[0][0]

'''

_LOAD_SOURCE = '''\
def _load_source(
    source: 'str | Path | pd.DataFrame',
    companion_dir: 'str | Path | None' = None,
) -> 'tuple[pd.DataFrame, Path | None]':
    """Read raw data from a file path, ZIP archive, or DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source.copy(), Path(companion_dir) if companion_dir else None
    _p = Path(source)
    if _p.suffix.lower() == '.zip':
        _tmp = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(_p) as _zf:
            _zf.extractall(_tmp)
        _candidates = list(_tmp.rglob(_SOURCE_FILENAME))
        if not _candidates:
            _candidates = sorted(_tmp.rglob('*.csv'))
        if not _candidates:
            raise FileNotFoundError(f'No CSV found inside {_p}.')
        def _priority(p: Path) -> int:
            siblings = {s.name.lower() for s in p.parent.iterdir()}
            return 0 if any(k in s for s in siblings
                            for k in ('train', 'dose', 'target', 'label')) else 1
        _main = sorted(_candidates, key=_priority)[0]
        if companion_dir is None:
            companion_dir = _find_companion_dir(_tmp) or _main.parent
        return pd.read_csv(_main, dtype=object, keep_default_na=True), Path(companion_dir)
    _ext = _p.suffix.lower()
    if _ext in ('.xlsx', '.xls'):
        _df = pd.read_excel(_p, dtype=object)
    elif _ext == '.parquet':
        _df = pd.read_parquet(_p)
    elif _ext == '.json':
        _df = pd.read_json(_p, dtype=object)
    elif _ext in ('.sav',):
        _df = pd.read_spss(_p)
    elif _ext in ('.dta',):
        _df = pd.read_stata(_p)
    else:
        _df = pd.read_csv(_p, dtype=object, keep_default_na=True)
    if companion_dir is None:
        companion_dir = _find_companion_dir(_p.parent) or _p.parent
    return _df, Path(companion_dir) if companion_dir else None

'''

_MATCH_COMPANIONS = '''\
def match_companions(
    df: pd.DataFrame,
    companion_dir: 'str | Path',
    *,
    extensions: 'list[str] | None' = None,
) -> pd.DataFrame:
    """Add companion path columns by checking existence of derived filenames.

    Derives {jurisdiction}_{period_id}{ext} per row, checks Path.exists(),
    and adds image_path / has_image / companion_path / has_companion columns.
    No sidecar manifest file required.
    """
    _IMAGE = {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}
    _AUDIO = {'.wav', '.mp3', '.flac'}

    def _prefix(ext: str) -> str:
        e = ext.lower()
        if e in _IMAGE: return 'image'
        if e in _AUDIO: return 'audio'
        if e == '.txt': return 'text'
        return e.lstrip('.')

    if extensions is None:
        extensions = ['.png']
    companion_dir = Path(companion_dir)
    keys = _companion_row_keys(df)
    out = df.copy()
    first_image_done = False
    for ext in extensions:
        prefix = _prefix(ext)
        paths = [
            str(p) if (p := companion_dir / f"{k}{ext}").exists() else None
            for k in keys
        ]
        out[f"{prefix}_path"] = paths
        out[f"has_{prefix}"] = [p is not None for p in paths]
        if prefix == 'image' and not first_image_done:
            out['companion_path'] = out['image_path']
            out['has_companion'] = out['has_image']
            first_image_done = True
    return out


def to_hf_dataset(
    df: pd.DataFrame,
    *,
    image_col: str = 'image_path',
    target_col: 'str | None' = None,
) -> 'datasets.Dataset':
    """Convert cleaned DataFrame to a HuggingFace Dataset with a native Image column.

    Requires: pip install datasets Pillow
    """
    import datasets
    from datasets import Image as HFImage
    out = df.copy()
    if image_col in out.columns:
        out = out.rename(columns={image_col: 'image'})
    if target_col and target_col in out.columns:
        out['label'] = pd.Categorical(out[target_col]).codes
        out = out.drop(columns=[target_col])
    for col in out.select_dtypes(include='object').columns:
        if col != 'image':
            try:
                out[col] = pd.to_numeric(out[col])
            except (ValueError, TypeError):
                pass
    ds = datasets.Dataset.from_pandas(out, preserve_index=False)
    if 'image' in ds.column_names:
        ds = ds.cast_column('image', HFImage())
    return ds

'''

_MAIN_BLOCK = '''\
if __name__ == "__main__":
    _src           = sys.argv[1] if len(sys.argv) > 1 else _SOURCE_FILENAME
    _companion_dir = sys.argv[2] if len(sys.argv) > 2 else None
    cleaned = clean(_src, _companion_dir)
    cleaned.to_csv("clean.csv", index=False, lineterminator="\\n")
    print(f"Wrote clean.csv  ({cleaned.shape[0]:,} rows \\u00d7 {cleaned.shape[1]} cols)")
    try:
        cleaned.to_parquet("clean.parquet", index=False)
        print("Wrote clean.parquet")
    except Exception:
        pass
    _has_images = 'image_path' in cleaned.columns and cleaned['image_path'].notna().any()
    if _has_images:
        try:
            ds = to_hf_dataset(cleaned)
            ds.save_to_disk("clean_dataset/")
            _n = cleaned['image_path'].notna().sum()
            print(f"Wrote clean_dataset/  ({len(ds):,} rows, {_n:,} images embedded)")
        except ImportError:
            print("Skipped clean_dataset/ — pip install datasets Pillow to enable.")
    elif 'image_path' in cleaned.columns:
        print("Warning: image_path column present but no images resolved on disk. "
              "If PNGs are inside the ZIP they should be found automatically. "
              "Otherwise: python clean_script.py data.zip /path/to/pngs/")
'''


# ---------------------------------------------------------------------------
# Patch logic
# ---------------------------------------------------------------------------

def _remove_manifest_block(src: str) -> str:
    """Remove the companion_manifest.json loading block (module-level crash)."""
    # Pattern: from the companion comment block through _COMPANION_SUBJECT_IDS line
    pattern = re.compile(
        r"# [=\-]+ *\n"
        r"# Companion files.*?\n"
        r"(?:# .*?\n)*"          # any further comment lines
        r"(?:.*?\n)*?"           # any lines up to...
        r"_COMPANION_SUBJECT_IDS.*?\n",  # ...the subject IDs frozenset
        re.DOTALL,
    )
    cleaned = pattern.sub("", src)
    # Also remove any remaining bare `import json as _json` left over
    cleaned = re.sub(r"^import json as _json\n", "", cleaned, flags=re.MULTILINE)
    return cleaned


def patch(src: str) -> str:
    """Apply all fixes to the source text of a clean_script.py."""

    # 1. Remove manifest loading block
    src = _remove_manifest_block(src)

    # 2. Remove the has_companion line that uses _COMPANION_SUBJECT_IDS
    src = re.sub(
        r"[ \t]*# companion presence.*?\n"
        r"[ \t]*df\['has_companion'\] = _companion_row_keys.*?\n",
        "",
        src,
    )

    # 3. Insert companion block (key function + _find_companion_dir) before _load_source
    if "_find_companion_dir" not in src:
        src = re.sub(
            r"(def _load_source\()",
            _COMPANION_BLOCK + r"\1",
            src,
        )

    # 4. Replace _load_source body (keep signature line, replace everything up to next def)
    src = re.sub(
        r"def _load_source\(.*?\n\n\n",
        _LOAD_SOURCE + "\n\n",
        src,
        flags=re.DOTALL,
    )

    # 5. Replace match_companions (and everything after it up to if __name__)
    src = re.sub(
        r"def match_companions\(.*?(?=if __name__)",
        _MATCH_COMPANIONS,
        src,
        flags=re.DOTALL,
    )

    # 6. Replace __main__ block
    src = re.sub(
        r'if __name__ == "__main__":.*',
        _MAIN_BLOCK,
        src,
        flags=re.DOTALL,
    )

    return src


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    src_path = Path(sys.argv[1])
    dst_path = Path(sys.argv[2]) if len(sys.argv) > 2 else src_path

    original = src_path.read_text(encoding="utf-8")

    if "companion_manifest" not in original and "_find_companion_dir" in original:
        print(f"{src_path.name}: already patched, nothing to do.")
        return

    patched = patch(original)

    # Syntax check before writing
    import ast
    try:
        ast.parse(patched)
    except SyntaxError as e:
        print(f"ERROR: patch produced invalid Python: {e}")
        print("Original file left unchanged.")
        sys.exit(1)

    dst_path.write_text(patched, encoding="utf-8")
    print(f"Patched {src_path.name} → {dst_path.name}")
    if "companion_manifest" in patched:
        print("WARNING: manifest reference still present after patch — inspect manually.")


if __name__ == "__main__":
    main()
