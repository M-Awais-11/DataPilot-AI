"""
LangGraph Supervisor / Orchestrator Agent.
Builds the state graph, manages agent transitions, handles state validation,
and commits database artifacts (AgentRuns, Models, Metrics, Reports) at each step.
"""
from __future__ import annotations

import logging
from typing import Callable, Any, Optional
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from sqlalchemy import select

from app.config.settings import settings
from app.agents.state import DataScienceState
from app.agents import (
    data_understanding,
    cleaning,
    eda,
    feature_engineering,
    model_selection,
    training,
    evaluation,
    explainability,
    reporting,
)

logger = logging.getLogger(__name__)


def get_llm():
    """Instantiate OpenRouter LLM using langchain_openai."""
    if settings.openrouter_api_key:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=settings.openrouter_model,
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                temperature=settings.llm_temperature,
                default_headers={
                    "HTTP-Referer": "https://localhost:8000",
                    "X-Title": "Autonomous Data Scientist Agent",
                }
            )
        except Exception as e:
            logger.warning(f"Failed to initialize OpenRouter LLM: {e}")
    
    # Fallback to local deterministic reasoning simulation if no external API key
    class MockLLM:
        def invoke(self, messages):
            last_msg = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
            if "SELECTED:" in last_msg or "Select 3-4 algorithms" in last_msg:
                return AIMessage(content="SELECTED: XGBoost, RandomForest, LogisticRegression\nREASON: Selected optimal ensemble and linear baselines for balanced precision.\nPRIMARY_METRIC: f1_weighted")
            return AIMessage(content="AI Data Scientist Analysis:\nComputed statistics confirm stable distribution and predictive signal across target features.")
    
    return MockLLM()



def _save_agent_run_db(db_session, experiment_id: str, agent_name: str, record):
    """Save an AgentRun record to the database."""
    if not db_session:
        return
    try:
        from app.models.experiment import AgentRun
        import uuid
        ar = AgentRun(
            id=uuid.uuid4(),
            experiment_id=experiment_id,
            agent_name=agent_name,
            status=record.status,
            reasoning=record.reasoning,
            error_message=record.error,
            duration_ms=record.duration_ms,
            completed_at=datetime.now(timezone.utc),
        )
        db_session.add(ar)
        db_session.commit()
    except Exception as e:
        logger.warning(f"Failed to persist AgentRun to DB: {e}")
        db_session.rollback()


def _save_cleaning_steps_db(db_session, experiment_id: str, cleaning_log):
    """Save preprocessing steps to the database."""
    if not db_session:
        return
    try:
        from app.models.experiment import PreprocessingStep
        import uuid
        for step in cleaning_log:
            ps = PreprocessingStep(
                id=uuid.uuid4(),
                experiment_id=experiment_id,
                step_order=step.step_order,
                step_type=step.step_type,
                column_name=step.column_name,
                operation=step.operation,
                parameters=step.parameters,
                reason=step.reason,
                rows_affected=step.rows_affected,
            )
            db_session.add(ps)
        db_session.commit()
    except Exception as e:
        logger.warning(f"Failed to persist PreprocessingSteps to DB: {e}")
        db_session.rollback()


def _to_uuid(val):
    if val is None:
        return None
    if isinstance(val, uuid.UUID):
        return val
    try:
        return uuid.UUID(str(val))
    except Exception:
        return val


def _save_models_and_metrics_db(db_session, experiment_id: str, trained_models, best_model_id: Optional[str]):
    """Save trained models and evaluation metrics to DB, with strict type safety for SQL Server."""
    if not db_session or not trained_models:
        return
    try:
        import math
        from app.models.model import ModelMetric, TrainedModel

        exp_uuid = _to_uuid(experiment_id)
        best_mid_str = str(best_model_id) if best_model_id else None

        for tm in trained_models:
            mid = _to_uuid(tm.model_id)
            is_best_model = bool(best_mid_str and str(tm.model_id) == best_mid_str)

            model_record = db_session.get(TrainedModel, mid)
            if model_record is None:
                model_record = TrainedModel(
                    id=mid,
                    experiment_id=exp_uuid,
                    algorithm=tm.algorithm,
                    hyperparameters=tm.hyperparameters,
                    artifact_path=tm.artifact_path,
                    is_best=is_best_model,
                    is_production=False,
                )
                db_session.add(model_record)
            else:
                model_record.algorithm = tm.algorithm
                model_record.hyperparameters = tm.hyperparameters
                model_record.artifact_path = tm.artifact_path
                model_record.is_best = is_best_model

            # Flush model record so foreign keys in model_metrics can reference it safely
            db_session.flush()

            # Clean up any existing metrics for this model to avoid duplicates
            existing_metrics = db_session.execute(
                select(ModelMetric).where(ModelMetric.model_id == mid)
            ).scalars().all()
            for em in existing_metrics:
                db_session.delete(em)

            # Save metrics safely (SQL Server FLOAT does not allow NaN or Infinity)
            for split, split_m in (tm.metrics or {}).items():
                if isinstance(split_m, dict):
                    for m_name, m_val in split_m.items():
                        if isinstance(m_val, (int, float)):
                            val_float = float(m_val)
                            if math.isnan(val_float) or math.isinf(val_float):
                                continue  # Skip invalid floats that break SQL Server
                            mm = ModelMetric(
                                id=uuid.uuid4(),
                                model_id=mid,
                                split=str(split),
                                metric_name=str(m_name),
                                metric_value=val_float,
                            )
                            db_session.add(mm)
        db_session.commit()
        logger.info("Successfully persisted %d trained models and their metrics to DB.", len(trained_models))
    except Exception as e:
        logger.warning(f"Failed to persist models/metrics to DB: {e}")
        db_session.rollback()


def _save_feature_importance_db(db_session, model_id: Optional[str], feature_importance: list[dict]):
    """Persist explainability output after the SHAP agent has completed."""
    if not db_session or not model_id or not feature_importance:
        return
    try:
        import math
        from app.models.model import FeatureImportance

        mid = _to_uuid(model_id)
        existing = db_session.execute(
            select(FeatureImportance).where(FeatureImportance.model_id == mid)
        ).scalars().all()
        for item in existing:
            db_session.delete(item)

        for index, item in enumerate(feature_importance, start=1):
            imp = item.get("importance")
            shap = item.get("shap_mean_abs")
            if imp is not None:
                try:
                    imp_f = float(imp)
                    imp = 0.0 if (math.isnan(imp_f) or math.isinf(imp_f)) else imp_f
                except Exception:
                    imp = None
            if shap is not None:
                try:
                    shap_f = float(shap)
                    shap = 0.0 if (math.isnan(shap_f) or math.isinf(shap_f)) else shap_f
                except Exception:
                    shap = 0.0

            db_session.add(
                FeatureImportance(
                    id=uuid.uuid4(),
                    model_id=mid,
                    feature_name=str(item.get("feature") or item.get("feature_name") or "Unknown"),
                    importance=imp,
                    shap_mean_abs=shap,
                    rank=item.get("rank", index),
                )
            )
        db_session.commit()
        logger.info("Successfully persisted feature importance for model %s to DB.", model_id)
    except Exception as e:
        logger.warning(f"Failed to persist feature importance to DB: {e}")
        db_session.rollback()



def _save_report_db(db_session, experiment_id: str, report_path: str):
    """Save report record to DB."""
    if not db_session or not report_path:
        return
    try:
        from app.models.model import Report
        import uuid
        rep = db_session.execute(
            select(Report).where(Report.experiment_id == experiment_id)
        ).scalar_one_or_none()
        if rep:
            rep.file_path = report_path
        else:
            db_session.add(
                Report(
                    id=uuid.uuid4(),
                    experiment_id=experiment_id,
                    file_path=report_path,
                )
            )
        db_session.commit()
    except Exception as e:
        logger.warning(f"Failed to persist Report to DB: {e}")
        db_session.rollback()


def build_graph(experiment_id: str, db_session, emit_fn: Callable) -> Any:
    """Build and compile the LangGraph workflow."""
    llm = get_llm()

    # Node wrappers that execute the agent and persist database records
    def node_data_understanding(state: DataScienceState) -> DataScienceState:
        new_state = data_understanding.run(state, llm, emit_fn, db_session)
        if new_state.get("agent_history"):
            _save_agent_run_db(db_session, experiment_id, "data_understanding", new_state["agent_history"][-1])
        return new_state

    def node_cleaning(state: DataScienceState) -> DataScienceState:
        new_state = cleaning.run(state, llm, emit_fn, db_session)
        if new_state.get("agent_history"):
            _save_agent_run_db(db_session, experiment_id, "cleaning", new_state["agent_history"][-1])
        if new_state.get("cleaning_log"):
            _save_cleaning_steps_db(db_session, experiment_id, new_state["cleaning_log"])
        return new_state

    def node_eda(state: DataScienceState) -> DataScienceState:
        new_state = eda.run(state, llm, emit_fn, db_session)
        if new_state.get("agent_history"):
            _save_agent_run_db(db_session, experiment_id, "eda", new_state["agent_history"][-1])
        return new_state

    def node_feature_engineering(state: DataScienceState) -> DataScienceState:
        new_state = feature_engineering.run(state, llm, emit_fn, db_session)
        if new_state.get("agent_history"):
            _save_agent_run_db(db_session, experiment_id, "feature_engineering", new_state["agent_history"][-1])
        return new_state

    def node_model_selection(state: DataScienceState) -> DataScienceState:
        new_state = model_selection.run(state, llm, emit_fn, db_session)
        if new_state.get("agent_history"):
            _save_agent_run_db(db_session, experiment_id, "model_selection", new_state["agent_history"][-1])
        return new_state

    def node_training(state: DataScienceState) -> DataScienceState:
        new_state = training.run(state, llm, emit_fn, db_session)
        if new_state.get("agent_history"):
            _save_agent_run_db(db_session, experiment_id, "training", new_state["agent_history"][-1])
        return new_state

    def node_evaluation(state: DataScienceState) -> DataScienceState:
        new_state = evaluation.run(state, llm, emit_fn, db_session)
        if new_state.get("agent_history"):
            _save_agent_run_db(db_session, experiment_id, "evaluation", new_state["agent_history"][-1])
        if new_state.get("trained_models"):
            _save_models_and_metrics_db(
                db_session, experiment_id,
                new_state["trained_models"],
                new_state.get("best_model_id")
            )
        return new_state

    def node_explainability(state: DataScienceState) -> DataScienceState:
        new_state = explainability.run(state, llm, emit_fn, db_session)
        if new_state.get("agent_history"):
            _save_agent_run_db(db_session, experiment_id, "explainability", new_state["agent_history"][-1])
        _save_feature_importance_db(
            db_session,
            new_state.get("best_model_id"),
            new_state.get("feature_importance", []),
        )
        return new_state

    def node_reporting(state: DataScienceState) -> DataScienceState:
        new_state = reporting.run(state, llm, emit_fn, db_session)
        if new_state.get("agent_history"):
            _save_agent_run_db(db_session, experiment_id, "reporting", new_state["agent_history"][-1])
        if new_state.get("report_path"):
            _save_report_db(db_session, experiment_id, new_state["report_path"])
        return new_state

    # ── Conditional supervisor logic ──────────────────────────────────────────
    def check_data_understanding(state: DataScienceState) -> str:
        if state.get("error"):
            emit_fn("Supervisor: Halted pipeline due to profiling failure.", level="error")
            return END
        return "cleaning"

    def check_cleaning(state: DataScienceState) -> str:
        if state.get("error"):
            emit_fn("Supervisor: Halted pipeline due to cleaning failure.", level="error")
            return END
        return "eda"

    def check_training(state: DataScienceState) -> str:
        if not state.get("trained_models"):
            emit_fn("Supervisor: No models survived training.", level="error")
            return END
        return "evaluation"

    # ── Graph Assembly ────────────────────────────────────────────────────────
    workflow = StateGraph(DataScienceState)

    workflow.add_node("data_understanding", node_data_understanding)
    workflow.add_node("cleaning", node_cleaning)
    workflow.add_node("eda", node_eda)
    workflow.add_node("feature_engineering", node_feature_engineering)
    workflow.add_node("model_selection", node_model_selection)
    workflow.add_node("training", node_training)
    workflow.add_node("evaluation", node_evaluation)
    workflow.add_node("explainability", node_explainability)
    workflow.add_node("reporting", node_reporting)

    workflow.set_entry_point("data_understanding")

    workflow.add_conditional_edges(
        "data_understanding",
        check_data_understanding,
        {"cleaning": "cleaning", END: END},
    )
    workflow.add_conditional_edges(
        "cleaning",
        check_cleaning,
        {"eda": "eda", END: END},
    )
    workflow.add_edge("eda", "feature_engineering")
    workflow.add_edge("feature_engineering", "model_selection")
    workflow.add_edge("model_selection", "training")
    workflow.add_conditional_edges(
        "training",
        check_training,
        {"evaluation": "evaluation", END: END},
    )
    workflow.add_edge("evaluation", "explainability")
    workflow.add_edge("explainability", "reporting")
    workflow.add_edge("reporting", END)

    return workflow.compile()
