"""
AI Data Scientist Chatbot Service.
Implements grounded RAG over the exact experiment metadata, preprocessing steps,
model evaluation results, and SHAP explainability to prevent hallucination.
Can also recognize interactive user commands (e.g., retrain, drop column, change metric).
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from langchain_core.messages import SystemMessage, HumanMessage

from app.models.experiment import Experiment, AgentRun, PreprocessingStep
from app.models.dataset import Dataset, DatasetProfile
from app.models.model import TrainedModel, ModelMetric, FeatureImportance
from app.models.chat import ChatMessage
from app.models.user import User
from app.agents.supervisor import get_llm
from app.tasks.experiment_tasks import run_experiment

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_experiment_context(self, experiment_id: str, user_id: uuid.UUID) -> dict[str, Any]:
        """Fetch all factual experiment artifacts to provide grounded context to the LLM."""
        exp_res = await self.db.execute(
            select(Experiment).where(Experiment.id == experiment_id, Experiment.user_id == user_id)
        )
        experiment = exp_res.scalar_one_or_none()
        if not experiment:
            return {}

        # Preprocessing steps
        steps_res = await self.db.execute(
            select(PreprocessingStep).where(PreprocessingStep.experiment_id == experiment_id).order_by(PreprocessingStep.step_order)
        )
        steps = steps_res.scalars().all()

        # Agent runs
        runs_res = await self.db.execute(
            select(AgentRun).where(AgentRun.experiment_id == experiment_id).order_by(AgentRun.started_at)
        )
        runs = runs_res.scalars().all()

        # Models & metrics
        models_res = await self.db.execute(
            select(TrainedModel).where(TrainedModel.experiment_id == experiment_id)
        )
        models = models_res.scalars().all()

        model_summaries = []
        for m in models:
            m_metrics = await self.db.execute(
                select(ModelMetric).where(ModelMetric.model_id == m.id)
            )
            metrics_dict = {f"{metric.split}_{metric.metric_name}": metric.metric_value for metric in m_metrics.scalars().all()}
            
            # Feature importance
            fi_res = await self.db.execute(
                select(FeatureImportance).where(FeatureImportance.model_id == m.id).order_by(FeatureImportance.rank).limit(5)
            )
            top_feats = [f"{fi.feature_name} (shap={fi.shap_mean_abs})" for fi in fi_res.scalars().all()]

            model_summaries.append({
                "algorithm": m.algorithm,
                "is_best": m.is_best,
                "metrics": metrics_dict,
                "top_features": top_feats,
                "hyperparameters": m.hyperparameters,
            })

        return {
            "experiment_name": experiment.name,
            "status": experiment.status,
            "target_column": experiment.target_column,
            "problem_type": experiment.problem_type,
            "primary_metric": experiment.primary_metric,
            "preprocessing_steps": [
                f"Step {s.step_order} [{s.step_type}]: {s.column_name or 'dataset'} -> {s.operation} ({s.reason})"
                for s in steps
            ],
            "agent_runs": [
                f"{r.agent_name} ({r.status}): {r.reasoning[:200] if r.reasoning else ''}"
                for r in runs
            ],
            "models": model_summaries,
        }

    async def handle_chat(self, experiment_id: str, user_message: str, user: User) -> tuple[str, Optional[dict]]:
        """Process chat message, check for interactive commands, or ground with experiment context."""
        context = await self.get_experiment_context(experiment_id, user.id)

        # Check for interactive command patterns
        lower_msg = user_message.lower().strip()
        command_action = None

        if "retrain" in lower_msg or "train again" in lower_msg:
            # Trigger a new background run
            command_action = {"action": "retrain", "experiment_id": experiment_id}
            # Start background execution
            run_experiment(
                experiment_id=experiment_id,
                user_id=str(user.id),
                dataset_id=str(context.get("dataset_id", "")),
                target_column=context.get("target_column"),
                config={"commands": [user_message]},
            )

            response_text = (
                f"I've initiated a re-training workflow with your instructions: '{user_message}'. "
                f"You can monitor live progress on the Pipeline page."
            )
        else:
            # Standard grounded conversational QA
            llm = get_llm()
            system_prompt = f"""
You are the AI Data Scientist who executed this analysis. Answer the user's question accurately
using ONLY the factual experiment data provided below. Do not make up metrics or features.

=== EXPERIMENT FACTS ===
Experiment Name: {context.get('experiment_name')}
Status: {context.get('status')}
Target Column: {context.get('target_column')}
Problem Type: {context.get('problem_type')}
Primary Metric: {context.get('primary_metric')}

Preprocessing Performed:
{chr(10).join(context.get('preprocessing_steps', [])) or 'None logged yet'}

Model Results & Metrics:
{context.get('models')}

Agent Insights:
{chr(10).join(context.get('agent_runs', []))}
=========================

Guidelines:
1. Explain technical ML concepts clearly and professionally.
2. If asked 'Why did you select X?', refer directly to the model comparison metrics and data characteristics.
3. If asked about missing values or preprocessing, quote the exact operations logged.
4. Keep answers concise, factual, and informative.
"""
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
            llm_response = llm.invoke(messages)
            response_text = llm_response.content

        # Save message history to DB
        user_msg_record = ChatMessage(
            id=uuid.uuid4(),
            experiment_id=experiment_id,
            user_id=user.id,
            role="user",
            content=user_message,
        )
        ai_msg_record = ChatMessage(
            id=uuid.uuid4(),
            experiment_id=experiment_id,
            user_id=user.id,
            role="assistant",
            content=response_text,
            context_used=command_action or {"target": context.get("target_column")},
        )
        self.db.add(user_msg_record)
        self.db.add(ai_msg_record)
        await self.db.commit()

        return response_text, command_action
