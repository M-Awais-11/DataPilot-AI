"""
Experiment execution tasks.
Supports both Celery task execution and in-process background thread execution
without needing external message brokers or Docker.
"""
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.services.sse_service import publish_event_sync

logger = logging.getLogger(__name__)


def _get_sync_db() -> Session:
    """Create a synchronous SQLAlchemy session for SQL Server."""
    engine = create_engine(settings.get_sync_db_url(), pool_pre_ping=True)
    return Session(engine)


def execute_pipeline(
    experiment_id: str,
    user_id: str,
    dataset_id: str,
    target_column: Optional[str] = None,
    config: Optional[dict] = None,
    task_id: Optional[str] = None,
):
    """
    Core pipeline execution: runs the complete LangGraph data science pipeline.
    Updates experiment status and emits SSE progress events throughout.
    """
    from app.models.experiment import Experiment
    from app.models.dataset import Dataset

    config = config or {}
    db = _get_sync_db()

    def emit(message: str, agent: str = "supervisor", level: str = "info"):
        publish_event_sync(experiment_id, message, agent, level)
        logger.info(f"[{experiment_id}] [{agent}] {message}")

    try:
        # ── Mark experiment as running ──────────────────────────────────────
        db.execute(
            update(Experiment)
            .where(Experiment.id == experiment_id)
            .values(
                status="running",
                celery_task_id=task_id or str(experiment_id),
                updated_at=datetime.utcnow(),
            )
        )
        db.commit()

        emit("🚀 Autonomous Data Scientist Agent started", level="success")

        # ── Load dataset ────────────────────────────────────────────────────
        dataset_row = db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        ).scalar_one_or_none()

        if not dataset_row:
            raise ValueError(f"Dataset {dataset_id} not found")

        emit(f"📂 Dataset loaded: {dataset_row.original_name}", agent="supervisor")

        # ── Build initial LangGraph state ───────────────────────────────────
        from app.agents.state import DataScienceState
        from langchain_core.messages import SystemMessage

        initial_state: DataScienceState = {
            "experiment_id": experiment_id,
            "user_id": user_id,
            "dataset_id": dataset_id,
            "original_file_path": dataset_row.file_path,
            "working_file_path": dataset_row.file_path,
            "target_column": target_column,
            "target_confirmed": target_column is not None,
            "target_confidence": 1.0 if target_column else 0.0,
            "cleaning_log": [],
            "eda_insights": [],
            "visualization_paths": [],
            "feature_engineering_log": [],
            "agent_history": [],
            "user_commands": config.get("commands", []),
            "should_stop": False,
            "iteration_count": 0,
            "max_iterations": 3,
            "current_agent": "data_understanding",
            "messages": [
                SystemMessage(
                    content=(
                        "You are an expert AI Data Scientist. Analyze datasets, "
                        "make intelligent decisions about preprocessing, model selection, "
                        "and evaluation. Always base conclusions on computed data."
                    )
                )
            ],
            "leakage_warnings": [],
            "dropped_columns": [],
            "dropped_features": [],
            "data_quality_issues": [],
            "class_imbalance": False,
            "overfitting_detected": False,
            "trained_models": [],
            "candidate_models": [],
            "feature_importance": [],
            "evaluation_results": {},
            "model_comparison_table": [],
            "primary_metric": config.get("primary_metric", ""),
        }

        # ── Run LangGraph workflow ──────────────────────────────────────────
        from app.agents.supervisor import build_graph

        graph = build_graph(
            experiment_id=experiment_id,
            db_session=db,
            emit_fn=emit,
        )

        final_state = graph.invoke(initial_state, config={"recursion_limit": 100})

        # A run is only complete when the deliverables the user asked for actually
        # exist.  Previously a reporting exception was swallowed by the reporting
        # node and this task still labelled the experiment "completed", leaving an
        # empty results screen and no downloadable report.
        if final_state.get("error"):
            raise RuntimeError(final_state["error"])
        if not final_state.get("trained_models"):
            raise RuntimeError("Pipeline finished without any trained model artifacts")
        report_path = final_state.get("report_path")
        if not report_path or not Path(report_path).is_file() or Path(report_path).stat().st_size == 0:
            raise RuntimeError("Pipeline finished without a valid PDF report")

        # Background threads cannot be force-killed safely. Respect a user
        # cancellation that happened while a long-running model was finishing so
        # it is never overwritten with a misleading completed status.
        current_status = db.execute(
            select(Experiment.status).where(Experiment.id == experiment_id)
        ).scalar_one_or_none()
        if current_status == "cancelled":
            emit("Experiment cancelled by user", agent="supervisor", level="warning")
            return {"status": "cancelled", "experiment_id": experiment_id}

        # ── Mark experiment as completed ────────────────────────────────────
        db.execute(
            update(Experiment)
            .where(Experiment.id == experiment_id)
            .values(
                status="completed",
                problem_type=final_state.get("problem_type"),
                target_column=final_state.get("target_column"),
                primary_metric=final_state.get("primary_metric"),
                updated_at=datetime.utcnow(),
            )
        )
        db.commit()

        emit("✅ Experiment completed successfully!", agent="supervisor", level="done")
        return {"status": "completed", "experiment_id": experiment_id}

    except Exception as e:
        logger.exception(f"Experiment {experiment_id} failed: {e}")
        current_status = db.execute(
            select(Experiment.status).where(Experiment.id == experiment_id)
        ).scalar_one_or_none()
        if current_status == "cancelled":
            emit("Experiment cancellation confirmed", agent="supervisor", level="warning")
            return {"status": "cancelled", "experiment_id": experiment_id}
        db.execute(
            update(Experiment)
            .where(Experiment.id == experiment_id)
            .values(status="failed", updated_at=datetime.utcnow())
        )
        db.commit()
        emit(f"❌ Experiment failed: {str(e)}", agent="supervisor", level="fatal")
        raise

    finally:
        db.close()


def run_experiment_in_background(
    experiment_id: str,
    user_id: str,
    dataset_id: str,
    target_column: Optional[str] = None,
    config: Optional[dict] = None,
):
    """Launch execution in a detached background thread (no Celery/Redis daemon required)."""
    thread = threading.Thread(
        target=execute_pipeline,
        args=(experiment_id, user_id, dataset_id, target_column, config, f"thread_{experiment_id[:8]}"),
        daemon=True,
    )
    thread.start()
    return f"thread_{experiment_id[:8]}"


# Alias for backward compatibility
run_experiment = run_experiment_in_background
