"""Model training from a Distill Agent session zip.

Usage:
    python model_from_session.py distill_20260526-204013-5b9b7c.zip

Reads clean.csv directly from the zip, trains three regression models,
and prints evaluation metrics + feature importances.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder


# ── 1. Load ───────────────────────────────────────────────────────────────────

def load_clean(zip_path: str | Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        with z.open("clean.csv") as f:
            return pd.read_csv(f)


# ── 2. Feature engineering ────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    # Target
    y = df.pop("rate_per_10000_ed_visits")

    # period_id → year + month numeric features (format: e.g. "2022-03")
    if df["period_id"].str.match(r"^\d{4}-\d{2}$").all():
        df["period_year"] = df["period_id"].str[:4].astype(int)
        df["period_month"] = df["period_id"].str[5:].astype(int)
    df = df.drop(columns=["period_id"])

    # state_doh_release: too many unique values — drop raw, keep the missingness flag
    df = df.drop(columns=["state_doh_release"])

    # Encode low-cardinality categoricals
    for col in ["overdose_category", "jurisdiction"]:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Boolean → int
    df["state_doh_release_was_missing"] = df["state_doh_release_was_missing"].astype(int)

    return df.astype(float), y


# ── 3. Evaluate ───────────────────────────────────────────────────────────────

def evaluate(name: str, model, X_tr, y_tr, X_te, y_te) -> None:
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = mean_squared_error(y_te, pred) ** 0.5
    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)
    cv_r2 = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2").mean()
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Test  RMSE : {rmse:.4f}")
    print(f"  Test  MAE  : {mae:.4f}")
    print(f"  Test  R²   : {r2:.4f}")
    print(f"  CV-5  R²   : {cv_r2:.4f}")
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=X_tr.columns)
        top = imp.nlargest(5)
        print(f"  Top-5 features:")
        for feat, val in top.items():
            print(f"    {feat:<35} {val:.4f}")


# ── 4. Main ───────────────────────────────────────────────────────────────────

def main(zip_path: str) -> None:
    print(f"Loading clean data from {zip_path} ...")
    df = load_clean(zip_path)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    X, y = build_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train: {len(X_tr):,}  Test: {len(X_te):,}")

    models = [
        ("Ridge Regression",         Ridge(alpha=1.0)),
        ("Random Forest",            RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ("Gradient Boosting (GBRT)", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                                max_depth=4, random_state=42)),
    ]

    for name, model in models:
        evaluate(name, model, X_tr, y_tr, X_te, y_te)

    print("\nDone.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "distill_20260526-204013-5b9b7c.zip"
    main(path)
