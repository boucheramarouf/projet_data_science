"""
Data loading, cleaning, and preprocessing for injury prediction dataset.

Dataset structure: 7-day sliding windows, 10 features per day (days .0 to .6),
plus Athlete ID, injury (target), Date.
Missing sessions are encoded as -0.01 in perceived metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "day_approach_maskedID_timeseries.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Features per day (10 metrics × 7 days = 70 features)
DAY_FEATURES = [
    "nr. sessions", "total km", "km Z3-4", "km Z5-T1-T2",
    "km sprinting", "strength training", "hours alternative",
    "perceived exertion", "perceived trainingSuccess", "perceived recovery"
]
N_DAYS = 7
TARGET = "injury"
DROP_COLS = ["Athlete ID", "Date", TARGET]


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in DROP_COLS]


def replace_sentinel_values(df: pd.DataFrame, sentinel: float = -0.01) -> pd.DataFrame:
    """Replace -0.01 sentinel (missing session) with 0 for non-perceived features,
    and NaN for perceived features so imputation can handle them."""
    df = df.copy()
    perceived_cols = [c for c in df.columns if "perceived" in c.lower()]
    other_cols = [c for c in get_feature_columns(df) if c not in perceived_cols]

    # For objective metrics: -0.01 means no session → 0
    for col in other_cols:
        if col in df.columns:
            df[col] = df[col].replace(sentinel, 0.0)

    # For subjective perceived metrics: -0.01 means not reported → NaN
    for col in perceived_cols:
        if col in df.columns:
            df[col] = df[col].replace(sentinel, np.nan)

    return df


def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features: 7-day totals, means, max, load ratios."""
    df = df.copy()
    feature_cols = get_feature_columns(df)

    for base in DAY_FEATURES:
        day_cols = [c for c in feature_cols if c.startswith(base)]
        if len(day_cols) == 0:
            continue
        numeric_vals = df[day_cols]
        clean_tag = base.replace(" ", "_").replace(".", "").replace("-", "_")
        df[f"agg_sum_{clean_tag}"] = numeric_vals.sum(axis=1)
        df[f"agg_mean_{clean_tag}"] = numeric_vals.mean(axis=1)
        df[f"agg_max_{clean_tag}"] = numeric_vals.max(axis=1)

    # Acute:Chronic workload ratio (last day vs 7-day mean) for total km
    last_day_km = [c for c in feature_cols if c.startswith("total km")]
    if len(last_day_km) > 0:
        last_col = last_day_km[-1]   # most recent day
        mean_7d = df[[c for c in last_day_km]].mean(axis=1)
        df["acwr_total_km"] = df[last_col] / (mean_7d + 1e-6)

    return df


def preprocess(df: pd.DataFrame, add_features: bool = True) -> tuple:
    """Full preprocessing pipeline. Returns (X, y, feature_names)."""
    df = replace_sentinel_values(df)

    if add_features:
        df = add_aggregate_features(df)

    feature_cols = get_feature_columns(df)
    # Re-include engineered features
    agg_cols = [c for c in df.columns if c.startswith("agg_") or c == "acwr_total_km"]
    all_feature_cols = feature_cols + [c for c in agg_cols if c not in feature_cols]

    X = df[all_feature_cols]
    y = df[TARGET]

    return X, y, all_feature_cols


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def build_preprocessing_pipeline(use_smote: bool = False):
    """
    Returns a sklearn-compatible pipeline that:
    1. Imputes NaN (median strategy, avoids data leakage)
    2. Scales features (StandardScaler)
    Optionally applies SMOTE for class imbalance.
    """
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    if use_smote:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        return ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
        ])

    return Pipeline(steps)


def save_processed(X_train, X_test, y_train, y_test, feature_names):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    pd.DataFrame(X_train, columns=feature_names).to_parquet(
        os.path.join(PROCESSED_DIR, "X_train.parquet"), index=False
    )
    pd.DataFrame(X_test, columns=feature_names).to_parquet(
        os.path.join(PROCESSED_DIR, "X_test.parquet"), index=False
    )
    pd.Series(y_train, name=TARGET).to_parquet(
        os.path.join(PROCESSED_DIR, "y_train.parquet"), index=False
    )
    pd.Series(y_test, name=TARGET).to_parquet(
        os.path.join(PROCESSED_DIR, "y_test.parquet"), index=False
    )
    joblib.dump(feature_names, os.path.join(PROCESSED_DIR, "feature_names.pkl"))


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Raw shape: {df.shape}")
    print(f"Target distribution:\n{df[TARGET].value_counts(normalize=True)}")
    X, y, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train injury rate: {y_train.mean():.4f}")
    save_processed(X_train, X_test, y_train, y_test, feature_names)
    print("Processed data saved.")
