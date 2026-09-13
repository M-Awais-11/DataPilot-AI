"""
Explainability Agent.
Loads the best model, computes SHAP values, and uses LLM to explain them.
"""
from __future__ import annotations

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from app.agents.state import DataScienceState, AgentRunRecord, SHAPResults
from app.config.settings import settings
from app.tools.shap_tools import compute_shap

logger = logging.getLogger(__name__)
AGENT_NAME = "explainability"


def run(state: DataScienceState, llm, emit_fn, db_session) -> DataScienceState:
    start = datetime.now(timezone.utc)
    emit_fn("🔬 Explainability Agent started...", agent=AGENT_NAME)

    try:
        best_model_id = state.get("best_model_id")
        trained_models = state.get("trained_models", [])

        best_model_result = next((m for m in trained_models if m.model_id == best_model_id), None)
        if not best_model_result:
            best_model_result = trained_models[0] if trained_models else None

        if not best_model_result:
            raise ValueError("No trained model found for explainability")

        # ── Load model artifact ────────────────────────────────────────────────
        with open(best_model_result.artifact_path, "rb") as f:
            artifact = pickle.load(f)
        model = artifact["model"]
        features = artifact.get("features", [])

        # ── Load feature-engineered dataset ───────────────────────────────────
        file_path = state.get("feature_engineered_file_path") or state.get("working_file_path")
        df = pd.read_csv(file_path, low_memory=False)
        target_col = state.get("target_column")

        X = df[[c for c in features if c in df.columns]].select_dtypes(include=np.number).fillna(0)
        y = df[target_col] if target_col and target_col in df.columns else None

        # ── Compute SHAP ───────────────────────────────────────────────────────
        chart_dir = Path(settings.upload_dir) / str(state.get("user_id", "unknown")) / "charts" / state["experiment_id"]
        chart_dir.mkdir(parents=True, exist_ok=True)

        emit_fn(f"Computing SHAP values for {best_model_result.algorithm}...", agent=AGENT_NAME)

        from sklearn.model_selection import train_test_split
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=settings.random_seed)

        shap_result = compute_shap(
            model=model,
            X_train=X_train,
            X_test=X_test,
            output_dir=str(chart_dir),
            problem_type=state.get("problem_type", "classification"),
        )

        if not shap_result:
            raise ValueError("SHAP computation returned no results")

        feature_importance = shap_result.get("feature_importance", [])
        emit_fn(f"SHAP complete — top feature: {feature_importance[0]['feature'] if feature_importance else 'N/A'}", agent=AGENT_NAME)

        shap_results = SHAPResults(
            model_id=best_model_result.model_id,
            summary_plot_path=shap_result.get("summary_plot_path", ""),
            bar_plot_path=shap_result.get("bar_plot_path", ""),
            feature_importance=feature_importance,
            local_explanation_available=shap_result.get("local_explanation_available", False),
        )

        # ── LLM interprets SHAP results ────────────────────────────────────────
        top_features = feature_importance[:10]
        feat_text = "\n".join([
            f"  {i+1}. {f['feature']}: mean |SHAP| = {f['shap_mean_abs']:.4f}"
            for i, f in enumerate(top_features)
        ])

        prompt = f"""
SHAP explainability analysis is complete for {best_model_result.algorithm}.

Top 10 most important features by mean |SHAP value|:
{feat_text}

Target column: {target_col}
Problem type: {state.get('problem_type', 'classification')}

Please:
1. Explain what SHAP values reveal about the model's decision-making
2. Interpret the top 3 features in plain English (what do high values mean?)
3. Note any surprising or counterintuitive findings
4. Recommend any actions based on these insights (e.g., collect more data for important features)
"""
        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content=prompt))
        llm_response = llm.invoke(messages)
        reasoning = llm_response.content

        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        emit_fn(f"✓ Explainability complete in {duration_ms}ms", agent=AGENT_NAME, level="success")

        return {
            **state,
            "shap_results": shap_results,
            "feature_importance": feature_importance,
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [
                AgentRunRecord(agent_name=AGENT_NAME, status="completed", reasoning=reasoning, duration_ms=duration_ms)
            ],
            "messages": messages + [llm_response],
        }

    except Exception as e:
        logger.exception(f"Explainability Agent failed: {e}")
        emit_fn(f"⚠️ Explainability skipped: {e}", agent=AGENT_NAME, level="warning")
        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        return {
            **state,
            "error": None,  # Non-fatal
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [
                AgentRunRecord(agent_name=AGENT_NAME, status="failed", error=str(e), duration_ms=duration_ms)
            ],
        }
