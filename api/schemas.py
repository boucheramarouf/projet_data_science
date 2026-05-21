"""
Pydantic schemas for the FastAPI prediction service.
Each field corresponds to a 7-day training window metric.
"""

from pydantic import BaseModel, Field
from typing import Optional


class DayData(BaseModel):
    nr_sessions: float = Field(0.0, ge=0, description="Number of training sessions")
    total_km: float = Field(0.0, ge=0, description="Total km run")
    km_z3_4: float = Field(0.0, ge=0, description="Km in heart-rate zone Z3-Z4")
    km_z5_t1_t2: float = Field(0.0, ge=0, description="Km in heart-rate zone Z5-T1-T2")
    km_sprinting: float = Field(0.0, ge=0, description="Km sprinting")
    strength_training: float = Field(0.0, ge=0, le=1, description="Strength session (0/1)")
    hours_alternative: float = Field(0.0, ge=0, description="Hours of alternative training")
    perceived_exertion: float = Field(0.1, ge=0, le=1, description="RPE (normalised 0-1)")
    perceived_training_success: float = Field(0.0, ge=0, le=1)
    perceived_recovery: float = Field(0.15, ge=0, le=1)


class PredictRequest(BaseModel):
    """7-day training load for one runner. day_0 = oldest, day_6 = most recent."""
    day_0: DayData = Field(default_factory=DayData)
    day_1: DayData = Field(default_factory=DayData)
    day_2: DayData = Field(default_factory=DayData)
    day_3: DayData = Field(default_factory=DayData)
    day_4: DayData = Field(default_factory=DayData)
    day_5: DayData = Field(default_factory=DayData)
    day_6: DayData = Field(default_factory=DayData)
    model: str = Field("random_forest",
                        description="Model to use: logistic_regression | random_forest | xgboost | svm | mlp")


class PredictResponse(BaseModel):
    model: str
    injury_probability: float
    threshold: float
    risk_level: str   # "HIGH" | "LOW"
    prediction: int   # 1 = injury, 0 = no injury
