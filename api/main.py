"""
FastAPI REST service for injury prediction.

Endpoints:
  GET  /health        — service health check
  GET  /model-info    — available models and their thresholds
  POST /predict       — predict injury risk for a 7-day training window

Usage:
    uvicorn api.main:app --reload --port 8000
"""

import os
import sys
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PredictRequest, PredictResponse
from src.preprocessing import add_aggregate_features, DAY_FEATURES
from src.models import load_sklearn_model, MODELS_DIR

app = FastAPI(
    title="Injury Prediction API",
    description="Predict weekly injury risk for competitive runners based on 7-day training load.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup — load models once
# ---------------------------------------------------------------------------

SKLEARN_MODELS = {}
THRESHOLDS = {}
FEATURE_NAMES = []

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "svm": "SVM",
    "mlp": "MLP (Deep Learning)",
}


@app.on_event("startup")
async def startup_event():
    global SKLEARN_MODELS, THRESHOLDS, FEATURE_NAMES

    for name in ["logistic_regression", "random_forest", "xgboost", "svm", "mlp"]:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            SKLEARN_MODELS[name] = load_sklearn_model(name)

    thresh_path = os.path.join(ROOT, "results", "thresholds.json")
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            raw = json.load(f)
        inv = {v: k for k, v in DISPLAY_NAMES.items()}
        THRESHOLDS = {inv.get(k, k): v for k, v in raw.items()}

    feat_path = os.path.join(ROOT, "results", "feature_names.json")
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            FEATURE_NAMES = json.load(f)


# ---------------------------------------------------------------------------
# Helper — convert request → feature row
# ---------------------------------------------------------------------------

COL_RENAMES = {
    "nr_sessions": "nr. sessions",
    "total_km": "total km",
    "km_z3_4": "km Z3-4",
    "km_z5_t1_t2": "km Z5-T1-T2",
    "km_sprinting": "km sprinting",
    "strength_training": "strength training",
    "hours_alternative": "hours alternative",
    "perceived_exertion": "perceived exertion",
    "perceived_training_success": "perceived trainingSuccess",
    "perceived_recovery": "perceived recovery",
}


def request_to_row(req: PredictRequest) -> pd.DataFrame:
    """Build a single-row DataFrame matching the training feature schema."""
    days = [req.day_0, req.day_1, req.day_2, req.day_3,
            req.day_4, req.day_5, req.day_6]
    row = {}
    for d_idx, day in enumerate(days):
        for field, val in day.dict().items():
            col_base = COL_RENAMES[field]
            col_name = col_base if d_idx == 0 else f"{col_base}.{d_idx}"
            row[col_name] = val

    row_df = pd.DataFrame([row])
    row_full = add_aggregate_features(row_df)

    for feat in FEATURE_NAMES:
        if feat not in row_full.columns:
            row_full[feat] = 0.0

    return row_full[FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(SKLEARN_MODELS.keys()),
        "n_features": len(FEATURE_NAMES),
    }


@app.get("/model-info")
def model_info():
    return {
        "available_models": list(SKLEARN_MODELS.keys()),
        "thresholds": THRESHOLDS,
        "display_names": DISPLAY_NAMES,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model_key = req.model.lower()
    if model_key not in SKLEARN_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_key}'. Available: {list(SKLEARN_MODELS.keys())}",
        )
    if not FEATURE_NAMES:
        raise HTTPException(status_code=503, detail="Feature names not loaded. Run train.py first.")

    row = request_to_row(req)
    model = SKLEARN_MODELS[model_key]
    prob = float(model.predict_proba(row)[0, 1])
    threshold = THRESHOLDS.get(model_key, 0.5)
    prediction = int(prob >= threshold)
    risk_level = "HIGH" if prediction == 1 else "LOW"

    return PredictResponse(
        model=DISPLAY_NAMES.get(model_key, model_key),
        injury_probability=round(prob, 4),
        threshold=round(threshold, 4),
        risk_level=risk_level,
        prediction=prediction,
    )
