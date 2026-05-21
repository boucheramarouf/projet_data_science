"""
Model evaluation, comparison, and SHAP interpretability.
Handles binary classification with severe class imbalance.
Primary metrics: F1-score, PR-AUC, Recall (macro + for class 1).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saved figures
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    ConfusionMatrixDisplay,
)
import shap
import os

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def find_best_threshold(y_true, y_prob, metric: str = "f1") -> float:
    """Return the decision threshold that maximises F1 (or recall) on the given set."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_thresh, best_score = 0.5, 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        else:
            score = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = t
    return float(best_thresh)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str,
    threshold: float = None,
    tune_threshold: bool = True,
) -> dict:
    """
    Evaluate a model (sklearn pipeline or MLPClassifierKeras).
    Returns a dict of metrics and the optimal threshold.
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)

    if threshold is None and tune_threshold:
        threshold = find_best_threshold(y_test, y_prob, metric="f1")
    elif threshold is None:
        threshold = 0.5

    y_pred = (y_prob >= threshold).astype(int)

    results = {
        "model": model_name,
        "threshold": round(threshold, 3),
        "accuracy": round((y_pred == y_test).mean(), 4),
        "f1_injury": round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "recall_injury": round(recall_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "precision_injury": round(precision_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }
    return results


def compare_models(results_list: list) -> pd.DataFrame:
    """Build a summary DataFrame from a list of evaluate_model dicts."""
    rows = []
    for r in results_list:
        rows.append({k: v for k, v in r.items() if k not in ("y_prob", "y_pred")})
    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values("pr_auc", ascending=False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_test, y_pred, model_name: str, save: bool = True):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not injured", "Injured"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, f"cm_{model_name.replace(' ', '_')}.png"), dpi=150)
    plt.close(fig)
    return fig


def plot_pr_curves(results_list: list, y_test, save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results_list:
        prec, rec, _ = precision_recall_curve(y_test, r["y_prob"])
        ax.plot(rec, prec, label=f"{r['model']} (AUC={r['pr_auc']:.3f})")
    baseline = y_test.mean()
    ax.axhline(baseline, color="gray", linestyle="--", label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, "pr_curves.png"), dpi=150)
    plt.close(fig)
    return fig


def plot_roc_curves(results_list: list, y_test, save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results_list:
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, label=f"{r['model']} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"), dpi=150)
    plt.close(fig)
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame, save: bool = True):
    metrics = ["f1_injury", "recall_injury", "precision_injury", "pr_auc", "roc_auc"]
    df_plot = comparison_df[metrics].reset_index().melt(id_vars="model", var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_plot, x="metric", y="score", hue="model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison — Key Metrics")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# SHAP interpretability
# ---------------------------------------------------------------------------

def compute_shap_values(model, X_train, X_test, model_name: str, feature_names: list):
    """
    Compute SHAP values. Works with sklearn pipelines (extracts the clf step).
    Tree-based models use TreeExplainer (fast); others use KernelExplainer.
    X_train / X_test can be DataFrames or numpy arrays.
    """
    def to_np(arr):
        return arr.values if hasattr(arr, "values") else np.array(arr)

    # Extract the clf from a pipeline and get transformed data
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.keys())
        transformer = model[:-1]
        import pandas as pd
        # Keep as DataFrame to preserve feature names when transforming
        X_tr_df = pd.DataFrame(to_np(X_train), columns=feature_names[:to_np(X_train).shape[1]])
        X_te_df = pd.DataFrame(to_np(X_test), columns=feature_names[:to_np(X_test).shape[1]])
        X_train_t = to_np(transformer.transform(X_tr_df))
        X_test_t = to_np(transformer.transform(X_te_df))
        clf = model.named_steps[steps[-1]]
    else:
        X_train_t = to_np(X_train)
        X_test_t = to_np(X_test)
        clf = model

    tree_models = ("RandomForestClassifier", "XGBClassifier", "GradientBoostingClassifier",
                   "ExtraTreesClassifier")
    clf_name = type(clf).__name__

    if clf_name in tree_models:
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_test_t)
        # Older SHAP: list [class0, class1] → take class 1
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]
        # Newer SHAP: 3-D array (n_samples, n_features, n_classes) → take class 1
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]
    else:
        np.random.seed(42)
        n_bg = min(100, len(X_train_t))
        bg_idx = np.random.choice(len(X_train_t), size=n_bg, replace=False)
        background = X_train_t[bg_idx]

        predict_fn = lambda x: clf.predict_proba(x)[:, 1]
        explainer = shap.KernelExplainer(predict_fn, background)
        n_sample = min(200, len(X_test_t))
        sample_idx = np.random.choice(len(X_test_t), size=n_sample, replace=False)
        X_sample = X_test_t[sample_idx]
        shap_vals = explainer.shap_values(X_sample, nsamples=50)
        X_test_t = X_sample

    return explainer, shap_vals, X_test_t


def plot_shap_summary(shap_vals, X_test_t, feature_names: list, model_name: str, save: bool = True):
    if isinstance(X_test_t, np.ndarray):
        X_df = pd.DataFrame(X_test_t, columns=feature_names)
    else:
        X_df = X_test_t

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_df, plot_type="bar", show=False, max_display=20)
    plt.title(f"SHAP Feature Importance — {model_name}")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, f"shap_summary_{model_name.replace(' ', '_')}.png"), dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_df, show=False, max_display=20)
    plt.title(f"SHAP Beeswarm — {model_name}")
    plt.tight_layout()
    if save:
        fig2.savefig(os.path.join(FIGURES_DIR, f"shap_beeswarm_{model_name.replace(' ', '_')}.png"), dpi=150)
    plt.close(fig2)


def get_top_shap_features(shap_vals, feature_names: list, top_n: int = 20) -> pd.DataFrame:
    sv = np.array(shap_vals)
    # 3D array (n_samples, n_features, n_classes) → class 1
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    mean_abs = np.abs(sv).mean(axis=0).ravel()
    df = pd.DataFrame({"feature": list(feature_names)[:len(mean_abs)], "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def quick_cv(model, X, y, cv: int = 5, random_state: int = 42,
             use_sample_weight: bool = False) -> dict:
    """
    StratifiedKFold cross-validation for one sklearn Pipeline.
    Returns mean ± std of F1, Recall, ROC-AUC, PR-AUC (all on the minority class).
    use_sample_weight: if True, inject clf__sample_weight at each fold (for MLP).
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    f1s, recs, rocs, praucs = [], [], [], []

    X_arr = X.values if hasattr(X, "values") else np.array(X)
    y_arr = y.values if hasattr(y, "values") else np.array(y)

    for tr_idx, val_idx in skf.split(X_arr, y_arr):
        X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

        m = clone(model)
        if use_sample_weight:
            n_neg = (y_tr == 0).sum()
            n_pos = (y_tr == 1).sum()
            total = len(y_tr)
            w = np.where(y_tr == 1, total / (2.0 * n_pos), total / (2.0 * n_neg))
            m.fit(X_tr, y_tr, clf__sample_weight=w)
        else:
            m.fit(X_tr, y_tr)

        y_prob = m.predict_proba(X_val)[:, 1]
        thresh = find_best_threshold(y_val, y_prob, metric="f1")
        y_pred = (y_prob >= thresh).astype(int)

        f1s.append(f1_score(y_val, y_pred, pos_label=1, zero_division=0))
        recs.append(recall_score(y_val, y_pred, pos_label=1, zero_division=0))
        rocs.append(roc_auc_score(y_val, y_prob))
        praucs.append(average_precision_score(y_val, y_prob))

    return {
        "f1_mean":       round(float(np.mean(f1s)), 4),
        "f1_std":        round(float(np.std(f1s)), 4),
        "recall_mean":   round(float(np.mean(recs)), 4),
        "recall_std":    round(float(np.std(recs)), 4),
        "roc_auc_mean":  round(float(np.mean(rocs)), 4),
        "roc_auc_std":   round(float(np.std(rocs)), 4),
        "pr_auc_mean":   round(float(np.mean(praucs)), 4),
        "pr_auc_std":    round(float(np.std(praucs)), 4),
    }
