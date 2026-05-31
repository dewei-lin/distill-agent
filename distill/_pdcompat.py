"""Small pandas-version compatibility helpers.

pandas 3.0 changed the default dtype for text columns from ``object`` to a
dedicated string dtype (``str``). Code that gated on ``is_object_dtype``
therefore silently skipped string columns on pandas 3 while working fine
on pandas 2. ``is_text_dtype`` is the version-robust replacement: it
recognises a column as text whether pandas stored it as ``object``, a
``StringDtype``, the pandas-3 ``str`` dtype, or a categorical.

Note: ``distill.io.load`` reads CSVs with ``dtype=object`` so the loaded
path is consistent across pandas versions; this helper matters when the
library is handed a DataFrame built or loaded some other way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def is_text_dtype(s: "pd.Series") -> bool:
    """True if the Series holds text or categorical values.

    Robust across pandas 2 (object default) and pandas 3 (str default).
    Implemented by exclusion — anything that is not numeric, boolean,
    datetime, or timedelta is treated as text — so it does not depend on
    what a given pandas version names its string dtype.
    """
    import pandas as pd

    if isinstance(s.dtype, pd.CategoricalDtype):
        return True
    return not (
        pd.api.types.is_numeric_dtype(s)
        or pd.api.types.is_bool_dtype(s)
        or pd.api.types.is_datetime64_any_dtype(s)
        or pd.api.types.is_timedelta64_dtype(s)
    )
