"""Experiment API router — creation, listing, real-time SSE progress, cancellation, deletion."""
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete as sql_delete

from app.database.connection import get_db
from app.models.user import User
from app.models.experiment import Experiment, AgentRun, PreprocessingStep
from app.models.dataset import Dataset
from app.models.model import FeatureImportance, ModelMetric, Report, TrainedModel, Prediction
from app.config.settings import settings
from app.schemas.experiment import (
    ExperimentCreateRequest,
    ExperimentResponse,
    AgentRunResponse,
    PreprocessingStepResponse,
    InteractiveCommandRequest,
)
from app.services.auth_service import get_current_user
from app.services.sse_service import event_stream
from app.tasks.experiment_tasks import run_experiment_in_background

router = APIRouter(prefix="/experiments", tags=["Experiments"])


async def _owned_experiment(
    experiment_id: str, current_user: User, db: AsyncSession
) -> Experiment:
    """Fetch an experiment only when it belongs to the authenticated user."""
    result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.user_id == current_user.id,
        )
    )
    experiment = result.scalar_one_or_none()
    if not experiment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")
    return experiment


def _chart_title(filename: str) -> str:
    """Turn a chart file name into a readable UI title."""
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


@router.post("", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    data: ExperimentCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create and start an autonomous data science experiment."""
    # Verify dataset exists and belongs to user
    ds_res = await db.execute(
        select(Dataset).where(Dataset.id == data.dataset_id, Dataset.user_id == current_user.id)
    )
    dataset = ds_res.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    experiment_id = uuid.uuid4()
    experiment = Experiment(
        id=experiment_id,
        user_id=current_user.id,
        dataset_id=dataset.id,
        name=data.name,
        description=data.description,
        target_column=data.target_column,
        primary_metric=data.primary_metric,
        status="pending",
        config={"commands": data.commands or []},
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)

    # Launch background thread
    task_id = run_experiment_in_background(
        experiment_id=str(experiment_id),
        user_id=str(current_user.id),
        dataset_id=str(dataset.id),
        target_column=data.target_column,
        config={"commands": data.commands or [], "primary_metric": data.primary_metric},
    )

    experiment.celery_task_id = task_id
    await db.commit()

    return experiment



@router.get("", response_model=list[ExperimentResponse])
async def list_experiments(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all experiments for the current user."""
    result = await db.execute(
        select(Experiment).where(Experiment.user_id == current_user.id).order_by(Experiment.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get experiment details."""
    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id, Experiment.user_id == current_user.id)
    )
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")
    return exp


@router.get("/{experiment_id}/results")
async def get_experiment_results(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return every persisted result required by the experiment results screen.

    The UI deliberately gets this from one ownership-checked endpoint so a completed
    experiment never appears empty just because its charts, model metrics and report
    were fetched through disconnected requests.
    """
    experiment = await _owned_experiment(experiment_id, current_user, db)

    # Sync disk models if any were not tracked in DB
    from app.api.models import _sync_disk_models_to_db
    await _sync_disk_models_to_db(db, current_user, str(experiment.id))

    models_res = await db.execute(
        select(TrainedModel)
        .where(TrainedModel.experiment_id == experiment.id)
        .order_by(TrainedModel.is_best.desc(), TrainedModel.created_at.asc())
    )
    model_payload = []
    for model in models_res.scalars().all():
        metrics_res = await db.execute(
            select(ModelMetric).where(ModelMetric.model_id == model.id)
        )
        importance_res = await db.execute(
            select(FeatureImportance)
            .where(FeatureImportance.model_id == model.id)
            .order_by(FeatureImportance.rank.asc())
        )
        model_payload.append(
            {
                "id": str(model.id),
                "algorithm": model.algorithm,
                "version": model.version,
                "is_best": model.is_best,
                "is_production": model.is_production,
                "hyperparameters": model.hyperparameters or {},
                "artifact_name": Path(model.artifact_path).name if model.artifact_path else None,
                "artifact_available": bool(model.artifact_path and os.path.isfile(model.artifact_path)),
                "metrics": [
                    {
                        "split": metric.split,
                        "metric_name": metric.metric_name,
                        "metric_value": metric.metric_value,
                    }
                    for metric in metrics_res.scalars().all()
                ],
                "feature_importance": [
                    {
                        "feature_name": item.feature_name,
                        "importance": item.importance,
                        "shap_mean_abs": item.shap_mean_abs,
                        "rank": item.rank,
                    }
                    for item in importance_res.scalars().all()
                ],
            }
        )

    charts_dir = (
        Path(settings.upload_dir)
        / str(current_user.id)
        / "charts"
        / str(experiment.id)
    )
    allowed_image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    charts = []
    if charts_dir.is_dir():
        for file_path in sorted(charts_dir.iterdir(), key=lambda item: item.name.lower()):
            if file_path.is_file() and file_path.suffix.lower() in allowed_image_suffixes:
                charts.append(
                    {
                        "name": _chart_title(file_path.name),
                        "filename": file_path.name,
                        "url": f"/api/experiments/{experiment.id}/charts/{file_path.name}",
                    }
                )

    report_res = await db.execute(
        select(Report).where(Report.experiment_id == experiment.id)
    )
    report = report_res.scalar_one_or_none()
    agent_runs_res = await db.execute(
        select(AgentRun)
        .where(AgentRun.experiment_id == experiment.id)
        .order_by(AgentRun.completed_at.asc())
    )
    preprocessing_res = await db.execute(
        select(PreprocessingStep)
        .where(PreprocessingStep.experiment_id == experiment.id)
        .order_by(PreprocessingStep.step_order.asc())
    )

    return {
        "experiment": {
            "id": str(experiment.id),
            "name": experiment.name,
            "status": experiment.status,
            "target_column": experiment.target_column,
            "problem_type": experiment.problem_type,
            "primary_metric": experiment.primary_metric,
        },
        "models": model_payload,
        "charts": charts,
        "report": {
            "available": bool(report and os.path.isfile(report.file_path)),
            "generated_at": report.generated_at if report else None,
        },
        "agent_runs": [
            {
                "agent_name": run.agent_name,
                "status": run.status,
                "reasoning": run.reasoning,
                "error_message": run.error_message,
                "duration_ms": run.duration_ms,
                "completed_at": run.completed_at,
            }
            for run in agent_runs_res.scalars().all()
        ],
        "preprocessing": [
            {
                "step_order": step.step_order,
                "step_type": step.step_type,
                "column_name": step.column_name,
                "operation": step.operation,
                "parameters": step.parameters,
                "reason": step.reason,
                "rows_affected": step.rows_affected,
            }
            for step in preprocessing_res.scalars().all()
        ],
    }


@router.get("/{experiment_id}/charts/{filename}")
async def download_experiment_chart(
    experiment_id: str,
    filename: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Serve a generated chart after validating both ownership and file location."""
    experiment = await _owned_experiment(experiment_id, current_user, db)
    requested_name = Path(filename).name
    if requested_name != filename or Path(requested_name).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chart not found")

    chart_dir = (Path(settings.upload_dir) / str(current_user.id) / "charts" / str(experiment.id)).resolve()
    chart_path = (chart_dir / requested_name).resolve()
    if chart_dir not in chart_path.parents or not chart_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chart not found")
    return FileResponse(chart_path)


@router.get("/{experiment_id}/progress")
async def stream_progress(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """SSE endpoint: streams live agent progress events to the client."""
    # Verify ownership
    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id, Experiment.user_id == current_user.id)
    )
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")

    return StreamingResponse(
        event_stream(experiment_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{experiment_id}/agents", response_model=list[AgentRunResponse])
async def get_agent_runs(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all agent execution records for an experiment."""
    # Verify ownership
    res = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id, Experiment.user_id == current_user.id)
    )
    if not res.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")

    runs = await db.execute(
        select(AgentRun).where(AgentRun.experiment_id == experiment_id).order_by(AgentRun.started_at)
    )
    return runs.scalars().all()


@router.get("/{experiment_id}/preprocessing", response_model=list[PreprocessingStepResponse])
async def get_preprocessing_steps(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all logged cleaning and preprocessing steps."""
    steps = await db.execute(
        select(PreprocessingStep).where(PreprocessingStep.experiment_id == experiment_id).order_by(PreprocessingStep.step_order)
    )
    return steps.scalars().all()


@router.post("/{experiment_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_experiment(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Cancel a running experiment."""
    res = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id, Experiment.user_id == current_user.id)
    )
    exp = res.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")

    exp.status = "cancelled"
    await db.commit()
    return {"status": "cancelled", "experiment_id": experiment_id}


@router.delete("/{experiment_id}", status_code=status.HTTP_200_OK)
async def delete_experiment(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Permanently delete an experiment and all its related data and files."""
    experiment = await _owned_experiment(experiment_id, current_user, db)

    # 1. Find all trained models for this experiment
    models_res = await db.execute(
        select(TrainedModel).where(TrainedModel.experiment_id == experiment.id)
    )
    models = models_res.scalars().all()

    # 2. Delete each model's artifact file + related DB records
    for model in models:
        if model.artifact_path and os.path.exists(model.artifact_path):
            try:
                os.remove(model.artifact_path)
            except OSError as exc:
                pass  # log and continue

        await db.execute(sql_delete(Prediction).where(Prediction.model_id == model.id))
        await db.execute(sql_delete(ModelMetric).where(ModelMetric.model_id == model.id))
        await db.execute(sql_delete(FeatureImportance).where(FeatureImportance.model_id == model.id))

    # 3. Delete trained model records
    await db.execute(sql_delete(TrainedModel).where(TrainedModel.experiment_id == experiment.id))

    # 4. Delete report file and DB record
    report_res = await db.execute(select(Report).where(Report.experiment_id == experiment.id))
    report = report_res.scalar_one_or_none()
    if report:
        if report.file_path and os.path.exists(report.file_path):
            try:
                os.remove(report.file_path)
            except OSError:
                pass
        await db.delete(report)

    # 5. Delete charts and models directory
    charts_dir = (
        Path(settings.upload_dir) / str(current_user.id) / "charts" / str(experiment.id)
    )
    if charts_dir.exists():
        shutil.rmtree(charts_dir, ignore_errors=True)

    models_dir = (
        Path(settings.upload_dir) / str(current_user.id) / "models" / str(experiment.id)
    )
    if models_dir.exists():
        shutil.rmtree(models_dir, ignore_errors=True)

    # 6. Delete chat messages, progress events, agent runs, preprocessing steps
    from app.models.chat import ChatMessage
    from app.models.experiment import ProgressEvent
    await db.execute(sql_delete(ChatMessage).where(ChatMessage.experiment_id == experiment.id))
    await db.execute(sql_delete(ProgressEvent).where(ProgressEvent.experiment_id == experiment.id))
    await db.execute(sql_delete(AgentRun).where(AgentRun.experiment_id == experiment.id))
    await db.execute(sql_delete(PreprocessingStep).where(PreprocessingStep.experiment_id == experiment.id))

    # 7. Delete the experiment itself
    await db.delete(experiment)
    await db.commit()

    return {"status": "deleted", "experiment_id": experiment_id}
