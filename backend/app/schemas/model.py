"""Pydantic schemas for Model, Metric, Feature Importance, and Prediction endpoints."""
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel


class ModelMetricResponse(BaseModel):
    split: str
    metric_name: str
    metric_value: float

    model_config = {"from_attributes": True}


from uuid import UUID
from typing import Any, Optional, Union

class FeatureImportanceResponse(BaseModel):
    feature_name: str
    importance: Optional[float] = None
    shap_mean_abs: Optional[float] = None
    rank: Optional[int] = None

    model_config = {"from_attributes": True}


class TrainedModelResponse(BaseModel):
    id: Union[UUID, str]
    experiment_id: Union[UUID, str]
    algorithm: str
    version: int
    hyperparameters: Optional[dict[str, Any]] = None
    is_best: bool
    is_production: bool
    created_at: datetime
    metrics: list[ModelMetricResponse] = []
    feature_importance: list[FeatureImportanceResponse] = []
    experiment_name: Optional[str] = None
    artifact_name: Optional[str] = None
    artifact_available: Optional[bool] = None

    model_config = {"from_attributes": True}


class PredictRequest(BaseModel):
    features: dict[str, Any]


class PredictResponse(BaseModel):
    prediction: Any
    confidence: Optional[float] = None
    model_version: str
    algorithm: str
    shap_contributions: Optional[dict[str, float]] = None
