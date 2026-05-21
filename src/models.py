"""
All 5 models for injury prediction:
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost
4. SVM
5. MLP (Deep Learning) — sklearn MLPClassifier, a multi-layer perceptron

Each model is wrapped in a sklearn-compatible pipeline with imputation + scaling.
class_weight='balanced' is set on applicable models to handle the ~98% class imbalance.
"""

import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")


def _base_steps():
    return [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]


def build_logistic_regression(C: float = 0.1, random_state: int = 42) -> Pipeline:
    return Pipeline(_base_steps() + [
        ("clf", LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        ))
    ])


def build_random_forest(
    n_estimators: int = 300,
    max_depth: int = None,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(_base_steps() + [
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ))
    ])


def build_xgboost(
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    random_state: int = 42,
    scale_pos_weight: float = None,
) -> Pipeline:
    """scale_pos_weight handles imbalance natively in XGBoost.
    If None, it will be computed from y_train by the caller."""
    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(_base_steps() + [("clf", clf)])


def build_svm(C: float = 0.1, random_state: int = 42) -> Pipeline:
    """
    LinearSVC + Platt scaling (sigmoid calibration) for probability estimates.
    class_weight='balanced' handles the ~98% class imbalance natively.
    """
    base_svm = LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
    )
    calibrated = CalibratedClassifierCV(base_svm, cv=5, method="sigmoid")
    return Pipeline(_base_steps() + [("clf", calibrated)])



def build_mlp(
    hidden_layer_sizes: tuple = (256, 128, 64),
    alpha: float = 1e-3,
    learning_rate_init: float = 5e-4,
    max_iter: int = 300,
    random_state: int = 42,
) -> Pipeline:
    """
    MLP deep learning classifier (3 hidden layers: 256-128-64).
    Uses standard sklearn Pipeline so sample_weight can be injected at fit time
    via pipeline.fit(X, y, clf__sample_weight=weights), which re-weights the
    cross-entropy loss in proportion to class frequency (equivalent to class_weight).
    """
    return Pipeline(_base_steps() + [
        ("clf", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            batch_size=256,
            learning_rate="adaptive",
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
            verbose=False,
        )),
    ])


def compute_sample_weights(y_train) -> np.ndarray:
    """Return per-sample weights inversely proportional to class frequency."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    total = len(y_train)
    w_neg = total / (2.0 * n_neg)
    w_pos = total / (2.0 * n_pos)
    weights = np.where(np.asarray(y_train) == 1, w_pos, w_neg)
    return weights


def compute_scale_pos_weight(y_train) -> float:
    """XGBoost scale_pos_weight = (# negative) / (# positive)."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return float(n_neg) / float(n_pos)


def save_sklearn_model(model, name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))


def load_sklearn_model(name: str):
    return joblib.load(os.path.join(MODELS_DIR, f"{name}.pkl"))
