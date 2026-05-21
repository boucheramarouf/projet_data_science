"""
5-Fold Stratified Cross-Validation — Injury Prediction.

Runs CV on LR, RF, XGBoost, MLP (SVM skipped — internal calibration CV already used).
Saves results to results/cv_results.csv.

Usage:
    python run_cv.py
"""

import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import load_raw_data, preprocess, split_data
from src.models import (
    build_logistic_regression, build_random_forest,
    build_xgboost, build_mlp, compute_scale_pos_weight,
)
from src.evaluation import quick_cv

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("5-Fold Stratified Cross-Validation")
    print("=" * 60)

    print("\n[1/2] Loading data...")
    df = load_raw_data()
    X, y, feature_names = preprocess(df, add_features=True)
    X_train, _, y_train, _ = split_data(X, y)
    spw = compute_scale_pos_weight(y_train)
    print(f"    Train set: {X_train.shape}")

    configs = [
        ("Logistic Regression",
         build_logistic_regression(C=0.1), False),
        ("Random Forest",
         build_random_forest(n_estimators=300, min_samples_leaf=5), False),
        ("XGBoost",
         build_xgboost(n_estimators=300, max_depth=5, scale_pos_weight=spw), False),
        ("MLP (Deep Learning)",
         build_mlp(hidden_layer_sizes=(256, 128, 64), max_iter=300), True),
    ]

    print("\n[2/2] Running 5-fold CV (this may take 10–20 min)...")
    rows = []
    for name, model, use_sw in configs:
        print(f"  → {name}...", end=" ", flush=True)
        res = quick_cv(model, X_train, y_train, cv=5, use_sample_weight=use_sw)
        res["model"] = name
        rows.append(res)
        print(f"F1={res['f1_mean']:.3f}±{res['f1_std']:.3f}  "
              f"PR-AUC={res['pr_auc_mean']:.3f}±{res['pr_auc_std']:.3f}")

    # SVM: note only, no CV (internal calibration already uses cv=5)
    rows.append({
        "model": "SVM",
        "f1_mean": float("nan"), "f1_std": float("nan"),
        "recall_mean": float("nan"), "recall_std": float("nan"),
        "roc_auc_mean": float("nan"), "roc_auc_std": float("nan"),
        "pr_auc_mean": float("nan"), "pr_auc_std": float("nan"),
    })

    cv_df = pd.DataFrame(rows).set_index("model")
    out_path = os.path.join(RESULTS_DIR, "cv_results.csv")
    cv_df.to_csv(out_path)
    print(f"\nCV results saved → {out_path}")
    print("\n" + "=" * 60)
    print(cv_df.to_string())


if __name__ == "__main__":
    main()
