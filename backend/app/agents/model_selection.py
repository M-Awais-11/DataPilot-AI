"""
Model Selection Agent.
Uses LLM reasoning over dataset characteristics to select a shortlist of algorithms.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage

from app.agents.state import DataScienceState, AgentRunRecord

logger = logging.getLogger(__name__)
AGENT_NAME = "model_selection"

# ── Algorithm catalogs ─────────────────────────────────────────────────────────
CLASSIFICATION_MODELS = [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "GradientBoosting",
    "XGBoost",
    "LightGBM",
    "SVM",
    "KNN",
]

REGRESSION_MODELS = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "XGBoostRegressor",
    "LightGBMRegressor",
    "SVR",
]


def run(state: DataScienceState, llm, emit_fn, db_session) -> DataScienceState:
    start = datetime.now(timezone.utc)
    emit_fn("🤖 Model Selection Agent started...", agent=AGENT_NAME)

    try:
        problem_type = state.get("problem_type", "classification")
        n_rows, n_cols = state.get("df_shape", (0, 0))
        n_features = len(state.get("selected_features", []))
        class_imbalance = state.get("class_imbalance", False)
        numerical_ratio = len(state.get("numerical_columns", [])) / max(n_cols, 1)
        categorical_ratio = len(state.get("categorical_columns", [])) / max(n_cols, 1)

        all_models = CLASSIFICATION_MODELS if problem_type == "classification" else REGRESSION_MODELS

        # ── Ask LLM to select models based on dataset characteristics ──────────
        prompt = f"""
You are selecting ML algorithms for a {problem_type} task.

Dataset characteristics:
- Rows: {n_rows}
- Features after selection: {n_features}
- Numerical feature ratio: {numerical_ratio:.0%}
- Categorical feature ratio: {categorical_ratio:.0%}
- Class imbalance: {class_imbalance}

Available algorithms: {all_models}

Select 3-4 algorithms and explain your choices. Consider:
1. Dataset size (large → tree-based, small → linear)
2. Feature types (mixed → tree-based)
3. Interpretability requirements
4. Class imbalance handling

Respond in this exact format:
SELECTED: Algorithm1, Algorithm2, Algorithm3
REASON: Your explanation here (2-3 sentences)
PRIMARY_METRIC: metric_name (e.g., f1_weighted, accuracy, roc_auc for classification; rmse, r2 for regression)
"""
        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content=prompt))
        llm_response = llm.invoke(messages)
        response_text = llm_response.content

        # Parse LLM response
        selected_models, primary_metric, reason = _parse_model_selection(
            response_text, all_models, problem_type
        )

        emit_fn(f"Selected models: {selected_models}", agent=AGENT_NAME)
        emit_fn(f"Primary metric: {primary_metric}", agent=AGENT_NAME)
        emit_fn(f"Reason: {reason[:200]}", agent=AGENT_NAME)

        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        emit_fn(f"✓ Model Selection complete in {duration_ms}ms", agent=AGENT_NAME, level="success")

        # Ensure primary metric is compatible with problem_type
        configured_metric = state.get("primary_metric")
        if problem_type == "regression":
            final_metric = configured_metric if configured_metric in ("rmse", "r2", "mae") else (primary_metric if primary_metric in ("rmse", "r2", "mae") else "rmse")
        else:
            final_metric = configured_metric if configured_metric in ("f1_weighted", "accuracy", "roc_auc") else (primary_metric if primary_metric in ("f1_weighted", "accuracy", "roc_auc") else "f1_weighted")

        return {
            **state,
            "candidate_models": selected_models,
            "model_selection_reason": reason,
            "primary_metric": final_metric,
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [
                AgentRunRecord(agent_name=AGENT_NAME, status="completed", reasoning=reason, duration_ms=duration_ms)
            ],
            "messages": messages + [llm_response],
        }

    except Exception as e:
        logger.exception(f"Model Selection Agent failed: {e}")
        emit_fn(f"❌ Model Selection failed: {e}", agent=AGENT_NAME, level="error")
        # Fallback: use sensible defaults
        problem_type = state.get("problem_type", "classification")
        defaults = (
            ["LogisticRegression", "RandomForest", "XGBoost"]
            if problem_type == "classification"
            else ["Ridge", "RandomForestRegressor", "XGBoostRegressor"]
        )
        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        return {
            **state,
            "candidate_models": defaults,
            "model_selection_reason": f"Fallback defaults due to error: {e}",
            "primary_metric": "f1_weighted" if problem_type == "classification" else "rmse",
            "error": str(e),
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [
                AgentRunRecord(agent_name=AGENT_NAME, status="failed", error=str(e), duration_ms=duration_ms)
            ],
        }


def _parse_model_selection(
    text: str,
    available: list[str],
    problem_type: str,
) -> tuple[list[str], str, str]:
    """Parse LLM model selection response."""
    selected = []
    primary_metric = "f1_weighted" if problem_type == "classification" else "rmse"
    reason = text

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("SELECTED:"):
            raw = line.replace("SELECTED:", "").strip()
            candidates = [m.strip().replace(" ", "") for m in raw.split(",")]
            # Fuzzy match against available
            for cand in candidates:
                for avail in available:
                    if cand.lower() == avail.lower() or cand.lower() in avail.lower():
                        if avail not in selected:
                            selected.append(avail)
                        break
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()
        elif line.startswith("PRIMARY_METRIC:"):
            primary_metric = line.replace("PRIMARY_METRIC:", "").strip().split()[0]

    # Ensure at least 2 models
    if len(selected) < 2:
        defaults = (
            ["RandomForest", "XGBoost", "LogisticRegression"]
            if problem_type == "classification"
            else ["Ridge", "RandomForestRegressor", "XGBoostRegressor"]
        )
        selected = defaults

    return selected[:4], primary_metric, reason
