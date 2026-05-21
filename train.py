"""
Full training pipeline — run this script to train all 5 models and save results.

Usage:
    python train.py

Outputs:
    saved_models/   — serialized models (.pkl / .keras)
    figures/        — evaluation plots
    results/        — comparison CSV
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import (
    load_raw_data, preprocess, split_data,
)
from src.models import (
    build_logistic_regression, build_random_forest,
    build_xgboost, build_svm, build_mlp,
    compute_scale_pos_weight, compute_sample_weights,
    save_sklearn_model,
)
from src.evaluation import (
    evaluate_model, compare_models,
    plot_confusion_matrix, plot_pr_curves, plot_roc_curves,
    plot_model_comparison, compute_shap_values, plot_shap_summary,
    get_top_shap_features,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("Injury Prediction — Training Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load & preprocess
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data...")
    df = load_raw_data()
    print(f"    Raw shape: {df.shape}")
    print(f"    Injury rate: {df['injury'].mean():.4f}")

    X, y, feature_names = preprocess(df, add_features=True)
    print(f"    Feature count after engineering: {len(feature_names)}")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"    Train: {X_train.shape}  |  Test: {X_test.shape}")

    # Imbalance weights
    spw = compute_scale_pos_weight(y_train)
    print(f"    scale_pos_weight (XGB): {spw:.2f}")

    # ------------------------------------------------------------------
    # 2. Train models
    # ------------------------------------------------------------------
    print("\n[2/6] Training models...")

    # 2a. Logistic Regression
    print("  ->Logistic Regression")
    lr = build_logistic_regression(C=0.1)
    lr.fit(X_train, y_train)
    save_sklearn_model(lr, "logistic_regression")

    # 2b. Random Forest
    print("  ->Random Forest")
    rf = build_random_forest(n_estimators=300, min_samples_leaf=5)
    rf.fit(X_train, y_train)
    save_sklearn_model(rf, "random_forest")

    # 2c. XGBoost
    print("  ->XGBoost")
    xgb = build_xgboost(n_estimators=300, max_depth=5, scale_pos_weight=spw)
    xgb.fit(X_train, y_train)
    save_sklearn_model(xgb, "xgboost")

    # 2d. SVM
    print("  ->SVM (this may take a few minutes on 34k rows)")
    svm = build_svm(C=0.1)
    svm.fit(X_train, y_train)
    save_sklearn_model(svm, "svm")

    # 2e. MLP (Deep Learning) — 3 hidden layers (256-128-64) with sample_weight
    print("  ->MLP (Deep Learning — 3 hidden layers: 256-128-64, balanced sample_weight)")
    mlp = build_mlp(hidden_layer_sizes=(256, 128, 64), max_iter=300)
    sample_weights = compute_sample_weights(y_train)
    mlp.fit(X_train, y_train, clf__sample_weight=sample_weights)
    save_sklearn_model(mlp, "mlp")

    # ------------------------------------------------------------------
    # 3. Evaluate all models
    # ------------------------------------------------------------------
    print("\n[3/6] Evaluating models...")

    results_list = []
    model_registry = [
        ("Logistic Regression",  lr),
        ("Random Forest",        rf),
        ("XGBoost",              xgb),
        ("SVM",                  svm),
        ("MLP (Deep Learning)",  mlp),
    ]
    for name, model in model_registry:
        r = evaluate_model(model, X_test, y_test, name, tune_threshold=True)
        results_list.append(r)
        print(f"  {name:25s}  F1={r['f1_injury']:.3f}  Recall={r['recall_injury']:.3f}  PR-AUC={r['pr_auc']:.3f}")

    # ------------------------------------------------------------------
    # 4. Save comparison table
    # ------------------------------------------------------------------
    comparison = compare_models(results_list)
    comparison.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
    print("\n[4/6] Model comparison:")
    print(comparison.to_string())

    # Save thresholds so the dashboard can reload them
    thresholds = {r["model"]: r["threshold"] for r in results_list}
    with open(os.path.join(RESULTS_DIR, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    # Save feature names
    with open(os.path.join(RESULTS_DIR, "feature_names.json"), "w") as f:
        json.dump(list(feature_names), f)

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    print("\n[5/6] Generating figures...")

    for r in results_list:
        plot_confusion_matrix(y_test, r["y_pred"], r["model"])

    plot_pr_curves(results_list, y_test)
    plot_roc_curves(results_list, y_test)
    plot_model_comparison(comparison)

    # ------------------------------------------------------------------
    # 6. SHAP — always on Random Forest (TreeExplainer, fast & reliable)
    # ------------------------------------------------------------------
    print("\n[6/6] Computing SHAP values (Random Forest)...")
    try:
        _, shap_vals_rf, X_test_rf = compute_shap_values(
            rf, X_train.values, X_test.values, "Random Forest", feature_names
        )
        plot_shap_summary(shap_vals_rf, X_test_rf, list(feature_names), "Random Forest")
        top_feats_rf = get_top_shap_features(shap_vals_rf, list(feature_names))
        top_feats_rf.to_csv(os.path.join(RESULTS_DIR, "shap_top_features_rf.csv"), index=False)
        print("  SHAP done — Random Forest")
    except Exception as e:
        print(f"  SHAP skipped: {e}")

    # XGBoost SHAP (also tree-based)
    try:
        _, shap_vals_xgb, X_test_xgb = compute_shap_values(
            xgb, X_train.values, X_test.values, "XGBoost", feature_names
        )
        plot_shap_summary(shap_vals_xgb, X_test_xgb, list(feature_names), "XGBoost")
        top_feats_xgb = get_top_shap_features(shap_vals_xgb, list(feature_names))
        top_feats_xgb.to_csv(os.path.join(RESULTS_DIR, "shap_top_features_xgb.csv"), index=False)
        print("  SHAP done — XGBoost")
    except Exception as e:
        print(f"  SHAP XGBoost skipped: {e}")

    print("\n" + "=" * 60)
    print("Training complete. Results saved to results/ and figures/")
    print("Run:  streamlit run dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
