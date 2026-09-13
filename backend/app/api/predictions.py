"""
Prediction API — serves predictions from trained and production models
with confidence scoring and instant local SHAP contribution explanation.
"""
import logging
import pickle
import uuid
from typing import Any
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.connection import get_db
from app.models.user import User
from app.models.experiment import Experiment
from app.models.model import TrainedModel, Prediction
from app.schemas.model import PredictRequest, PredictResponse
from app.services.auth_service import get_current_user
from app.tools.shap_tools import compute_local_explanation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Predictions"])


def _run_inference(model_record: TrainedModel, features_dict: dict[str, Any]) -> tuple[Any, float | None, dict | None]:
    """Load model artifact, prepare input frame, and execute prediction + local explanation."""
    if not model_record.artifact_path:
        raise HTTPException(status_code=500, detail="Model artifact path is missing")

    with open(model_record.artifact_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    label_encoder = artifact.get("label_encoder")
    expected_features = artifact.get("features", [])

    # Assemble single-row DataFrame aligned with training feature columns
    input_df = pd.DataFrame([features_dict])
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0.0
    input_df = input_df[expected_features].fillna(0.0)

    # Prediction & Confidence
    pred_raw = model.predict(input_df)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(input_df)[0]
        confidence = round(float(np.max(probas)), 4)

    # Inverse transform label if encoded
    prediction_val = pred_raw
    if label_encoder is not None:
        try:
            prediction_val = label_encoder.inverse_transform([pred_raw])[0]
        except Exception:
            prediction_val = pred_raw

    # Local SHAP explanation
    shap_contributions = None
    try:
        # Background reference
        background_sample = input_df.copy()
        local_res = compute_local_explanation(model, background_sample, input_df)
        shap_contributions = local_res.get("feature_contributions")
    except Exception as e:
        logger.warning(f"Failed to generate local SHAP contribution: {e}")

    # Convert numpy types to python standard for JSON serialization
    if hasattr(prediction_val, "item"):
        prediction_val = prediction_val.item()

    return prediction_val, confidence, shap_contributions


@router.post("/{model_id}", response_model=PredictResponse)
async def predict_model(
    model_id: str,
    data: PredictRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Serve prediction using a specific trained model ID."""
    try:
        m_uuid = uuid.UUID(str(model_id))
    except Exception:
        m_uuid = model_id

    res = await db.execute(
        select(TrainedModel)
        .join(Experiment, TrainedModel.experiment_id == Experiment.id)
        .where(TrainedModel.id == m_uuid, Experiment.user_id == current_user.id)
    )
    model = res.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    pred, conf, shap_contribs = _run_inference(model, data.features)

    # Record prediction in DB
    pred_record = Prediction(
        id=uuid.uuid4(),
        model_id=model.id,
        user_id=current_user.id,
        input_data=data.features,
        prediction={"result": pred},
        confidence=conf,
        shap_values=shap_contribs,
    )
    db.add(pred_record)
    await db.commit()

    return PredictResponse(
        prediction=pred,
        confidence=conf,
        model_version=f"{model.algorithm}_v{model.version}",
        algorithm=model.algorithm,
        shap_contributions=shap_contribs,
    )


@router.post("/production", response_model=PredictResponse)
async def predict_production(
    data: PredictRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Serve prediction using the currently active production model."""
    res = await db.execute(
        select(TrainedModel)
        .join(Experiment, TrainedModel.experiment_id == Experiment.id)
        .where(TrainedModel.is_production == True, Experiment.user_id == current_user.id)
        .order_by(TrainedModel.created_at.desc())
    )
    prod_model = res.scalars().first()
    if not prod_model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No model has been promoted to production yet.",
        )

    pred, conf, shap_contribs = _run_inference(prod_model, data.features)

    # Record prediction in DB
    pred_record = Prediction(
        id=uuid.uuid4(),
        model_id=prod_model.id,
        user_id=current_user.id,
        input_data=data.features,
        prediction={"result": pred},
        confidence=conf,
        shap_values=shap_contribs,
    )
    db.add(pred_record)
    await db.commit()

    return PredictResponse(
        prediction=pred,
        confidence=conf,
        model_version=f"{prod_model.algorithm}_v{prod_model.version}",
        algorithm=prod_model.algorithm,
        shap_contributions=shap_contribs,
    )
