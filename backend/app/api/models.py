import os
import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

from app.database.connection import get_db
from app.models.user import User
from app.models.experiment import Experiment
from app.models.model import TrainedModel, ModelMetric, FeatureImportance, Prediction
from app.schemas.model import TrainedModelResponse, ModelMetricResponse, FeatureImportanceResponse
from app.services.auth_service import get_current_user
from app.services.mlflow_service import MLflowService

from app.config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["Models"])


async def _sync_disk_models_to_db(db: AsyncSession, current_user: User, exp_id_filter: Optional[str] = None) -> None:
    """Discover any trained model .pkl files on disk and register them in the DB if missing."""
    import uuid
    import re
    try:
        models_base_dir = Path(settings.upload_dir) / str(current_user.id) / "models"
        if not models_base_dir.is_dir():
            return

        # Target specific experiment or all experiment folders for this user
        if exp_id_filter:
            exp_dirs = [models_base_dir / exp_id_filter] if (models_base_dir / exp_id_filter).is_dir() else []
        else:
            exp_dirs = [d for d in models_base_dir.iterdir() if d.is_dir()]

        added_count = 0
        for exp_dir in exp_dirs:
            exp_id_str = exp_dir.name
            try:
                exp_uuid = uuid.UUID(exp_id_str)
            except Exception:
                continue

            # Verify experiment exists for current user
            exp_check = await db.execute(
                select(Experiment).where(Experiment.id == exp_uuid, Experiment.user_id == current_user.id)
            )
            exp_record = exp_check.scalar_one_or_none()
            if not exp_record:
                continue

            pkl_files = list(exp_dir.glob("*.pkl"))
            if not pkl_files:
                continue

            # Check if any model is already marked as best for this experiment
            existing_best_check = await db.execute(
                select(TrainedModel).where(TrainedModel.experiment_id == exp_uuid, TrainedModel.is_best == True)
            )
            has_best = existing_best_check.scalar_one_or_none() is not None

            for idx, pkl_path in enumerate(pkl_files):
                stem = pkl_path.stem
                # Format is AlgoName_UUID
                uuid_match = re.search(r'([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})', stem)
                if uuid_match:
                    model_id_str = uuid_match.group(1)
                    algo_name = stem[:uuid_match.start()].rstrip("_")
                else:
                    model_id_str = str(uuid.uuid4())
                    algo_name = stem.split("_")[0] if "_" in stem else stem

                try:
                    m_uuid = uuid.UUID(model_id_str)
                except Exception:
                    m_uuid = uuid.uuid4()

                # Check if already in DB
                existing = await db.execute(select(TrainedModel).where(TrainedModel.id == m_uuid))
                if existing.scalar_one_or_none() is None:
                    is_best_val = (not has_best) and (idx == 0 or "RandomForest" in algo_name or "XGBoost" in algo_name)
                    if is_best_val:
                        has_best = True
                    tm = TrainedModel(
                        id=m_uuid,
                        experiment_id=exp_uuid,
                        algorithm=algo_name or "TrainedModel",
                        version=1,
                        hyperparameters={},
                        artifact_path=str(pkl_path),
                        is_best=is_best_val,
                        is_production=False,
                    )
                    db.add(tm)
                    added_count += 1

        if added_count > 0:
            await db.commit()
            logger.info("Synced %d disk models into trained_models database table.", added_count)
    except Exception as e:
        logger.warning("Could not auto-sync disk models: %s", e)


def _build_model_response(model: TrainedModel, exp_name: Optional[str], metrics_scalars, fi_scalars) -> TrainedModelResponse:
    """Safely construct a TrainedModelResponse without triggering lazy-load MissingGreenlet errors."""
    return TrainedModelResponse(
        id=str(model.id),
        experiment_id=str(model.experiment_id),
        algorithm=model.algorithm,
        version=model.version,
        hyperparameters=model.hyperparameters or {},
        is_best=model.is_best,
        is_production=model.is_production,
        created_at=model.created_at,
        metrics=[
            ModelMetricResponse(
                split=met.split,
                metric_name=met.metric_name,
                metric_value=float(met.metric_value),
            )
            for met in metrics_scalars
        ],
        feature_importance=[
            FeatureImportanceResponse(
                feature_name=fi.feature_name,
                importance=fi.importance,
                shap_mean_abs=fi.shap_mean_abs,
                rank=fi.rank,
            )
            for fi in fi_scalars
        ],
        experiment_name=exp_name,
        artifact_name=Path(model.artifact_path).name if model.artifact_path else None,
        artifact_available=bool(model.artifact_path and os.path.isfile(model.artifact_path)),
    )


@router.get("", response_model=list[TrainedModelResponse])
async def list_models(
    experiment_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all trained models belonging to the current user, optionally filtered by experiment."""
    import uuid

    # Sync any untracked models on disk first
    await _sync_disk_models_to_db(db, current_user, experiment_id)

    query = (
        select(TrainedModel, Experiment.name.label("exp_name"))
        .join(Experiment, TrainedModel.experiment_id == Experiment.id)
        .where(Experiment.user_id == current_user.id)
    )
    if experiment_id:
        try:
            exp_uuid = uuid.UUID(str(experiment_id))
            query = query.where(TrainedModel.experiment_id == exp_uuid)
        except Exception:
            query = query.where(TrainedModel.experiment_id == experiment_id)
    query = query.order_by(TrainedModel.is_best.desc(), TrainedModel.created_at.desc())

    res = await db.execute(query)
    rows = res.all()

    output = []
    for model, exp_name in rows:
        metrics_res = await db.execute(select(ModelMetric).where(ModelMetric.model_id == model.id))
        fi_res = await db.execute(select(FeatureImportance).where(FeatureImportance.model_id == model.id).order_by(FeatureImportance.rank))
        output.append(_build_model_response(model, exp_name, metrics_res.scalars().all(), fi_res.scalars().all()))

    return output


@router.get("/experiment/{experiment_id}", response_model=list[TrainedModelResponse])
async def get_experiment_models(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all models trained in a given experiment."""
    import uuid
    try:
        exp_uuid = uuid.UUID(str(experiment_id))
    except Exception:
        exp_uuid = experiment_id

    # Auto-sync disk models for this experiment if missing
    await _sync_disk_models_to_db(db, current_user, str(experiment_id))

    # Verify experiment ownership
    exp_res = await db.execute(
        select(Experiment).where(Experiment.id == exp_uuid, Experiment.user_id == current_user.id)
    )
    exp = exp_res.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")

    models_res = await db.execute(
        select(TrainedModel).where(TrainedModel.experiment_id == exp_uuid).order_by(TrainedModel.is_best.desc(), TrainedModel.created_at.desc())
    )
    models = models_res.scalars().all()

    output = []
    for m in models:
        metrics_res = await db.execute(select(ModelMetric).where(ModelMetric.model_id == m.id))
        fi_res = await db.execute(select(FeatureImportance).where(FeatureImportance.model_id == m.id).order_by(FeatureImportance.rank))
        output.append(_build_model_response(m, exp.name, metrics_res.scalars().all(), fi_res.scalars().all()))

    return output


@router.get("/{model_id}", response_model=TrainedModelResponse)
async def get_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get single model details including all split metrics and feature importance."""
    res = await db.execute(
        select(TrainedModel, Experiment.name.label("exp_name"))
        .join(Experiment, TrainedModel.experiment_id == Experiment.id)
        .where(TrainedModel.id == model_id, Experiment.user_id == current_user.id)
    )
    row = res.first()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    model, exp_name = row
    metrics_res = await db.execute(select(ModelMetric).where(ModelMetric.model_id == model.id))
    fi_res = await db.execute(select(FeatureImportance).where(FeatureImportance.model_id == model.id).order_by(FeatureImportance.rank))

    return _build_model_response(model, exp_name, metrics_res.scalars().all(), fi_res.scalars().all())



@router.post("/{model_id}/promote", status_code=status.HTTP_200_OK)
async def promote_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Promote a model to production stage in both the database and MLflow registry."""
    res = await db.execute(
        select(TrainedModel)
        .join(Experiment, TrainedModel.experiment_id == Experiment.id)
        .where(TrainedModel.id == model_id, Experiment.user_id == current_user.id)
    )
    model = res.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    # Demote any other production models for this experiment
    await db.execute(
        update(TrainedModel)
        .where(TrainedModel.experiment_id == model.experiment_id)
        .values(is_production=False)
    )

    model.is_production = True
    await db.commit()

    # Register in MLflow if URI available
    mlflow_svc = MLflowService()
    if model.mlflow_model_uri:
        mlflow_svc.register_production_model(
            model_name=f"Model_{model.algorithm}",
            model_uri=model.mlflow_model_uri,
        )

    return {
        "status": "promoted",
        "model_id": str(model.id),
        "algorithm": model.algorithm,
        "is_production": True,
    }


@router.delete("/{model_id}", status_code=status.HTTP_200_OK)
async def delete_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Permanently delete a trained model from database and remove its artifact file from disk."""
    res = await db.execute(
        select(TrainedModel)
        .join(Experiment, TrainedModel.experiment_id == Experiment.id)
        .where(TrainedModel.id == model_id, Experiment.user_id == current_user.id)
    )
    model = res.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    deleted_artifact = None
    if model.artifact_path and os.path.exists(model.artifact_path):
        try:
            os.remove(model.artifact_path)
            deleted_artifact = model.artifact_path
            logger.info("Successfully deleted model artifact from disk: %s", model.artifact_path)
        except OSError as e:
            logger.warning("Could not delete artifact file %s: %s", model.artifact_path, e)

    # Delete related records
    await db.execute(delete(Prediction).where(Prediction.model_id == model.id))
    await db.execute(delete(ModelMetric).where(ModelMetric.model_id == model.id))
    await db.execute(delete(FeatureImportance).where(FeatureImportance.model_id == model.id))

    was_best = model.is_best
    exp_id = model.experiment_id

    # Delete the model itself
    await db.delete(model)
    await db.commit()

    # If the champion was deleted, promote the next available model in this experiment
    if was_best:
        next_res = await db.execute(
            select(TrainedModel)
            .where(TrainedModel.experiment_id == exp_id)
            .order_by(TrainedModel.created_at.desc())
        )
        next_model = next_res.scalars().first()
        if next_model:
            next_model.is_best = True
            await db.commit()

    return {
        "status": "deleted",
        "model_id": str(model_id),
        "algorithm": model.algorithm,
        "deleted_artifact": deleted_artifact,
        "message": "Model and physical artifact deleted successfully",
    }
