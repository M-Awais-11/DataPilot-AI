"""
Evaluation Agent.
Computes full metric suites, detects overfitting, selects the best model,
and builds the model comparison table.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage

from app.agents.state import DataScienceState, AgentRunRecord, ModelEvaluation

logger = logging.getLogger(__name__)
AGENT_NAME = "evaluation"


def run(state: DataScienceState, llm, emit_fn, db_session) -> DataScienceState:
    start = datetime.now(timezone.utc)
    emit_fn("📏 Evaluation Agent started...", agent=AGENT_NAME)

    evaluation_results: dict[str, ModelEvaluation] = {}
    comparison_table: list[dict] = []

    try:
        trained_models = state.get("trained_models", [])
        problem_type = state.get("problem_type", "classification")
        primary_metric = state.get("primary_metric", "f1_weighted")
        class_imbalance = state.get("class_imbalance", False)

        if not trained_models:
            raise ValueError("No trained models to evaluate")

        # ── Determine most important metric ────────────────────────────────────
        if problem_type == "regression":
            if not primary_metric or primary_metric not in ("rmse", "r2", "mae"):
                primary_metric = "rmse"
        else:
            if not primary_metric or primary_metric not in ("f1_weighted", "accuracy", "roc_auc"):
                primary_metric = "f1_weighted"
            # For imbalanced classification, prefer f1 over accuracy
            if class_imbalance and primary_metric == "accuracy":
                primary_metric = "f1_weighted"
                emit_fn("⚠️ Class imbalance detected — switching primary metric to F1 (weighted)", agent=AGENT_NAME, level="warning")

        # ── Build evaluation results and comparison table ──────────────────────
        lower_is_better = primary_metric in ("rmse", "mae", "mse")

        for model_result in trained_models:
            val_metrics = model_result.metrics.get("validation", {})
            train_metrics = model_result.metrics.get("train", {})
            test_metrics = model_result.metrics.get("test", {})

            primary_val = val_metrics.get(primary_metric, float('inf') if lower_is_better else 0.0)
            primary_train = train_metrics.get(primary_metric, float('inf') if lower_is_better else 0.0)

            if lower_is_better:
                overfitting_score = primary_val - primary_train
                is_overfitting = primary_train > 0 and (overfitting_score / primary_train > 0.2)
            else:
                overfitting_score = primary_train - primary_val
                is_overfitting = overfitting_score > 0.1   # >10% gap

            eval_result = ModelEvaluation(
                model_id=model_result.model_id,
                algorithm=model_result.algorithm,
                primary_metric=primary_metric,
                primary_metric_value=primary_val,
                all_metrics={**val_metrics},
                overfitting_score=overfitting_score,
                is_overfitting=is_overfitting,
            )
            evaluation_results[model_result.model_id] = eval_result

            row = {
                "model_id": model_result.model_id,
                "algorithm": model_result.algorithm,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
                "cv_mean": model_result.metrics.get("cv_mean", {}).get(primary_metric),
                "cv_std": model_result.metrics.get("cv_std", {}).get(primary_metric),
                "overfitting_score": round(overfitting_score, 4) if isinstance(overfitting_score, (int, float)) else 0.0,
                "is_overfitting": is_overfitting,
                "training_time_s": model_result.training_time_seconds,
            }
            comparison_table.append(row)

        # Sort: lower is better for error metrics (rmse, mae, mse), higher is better for score metrics (f1, accuracy, r2)
        if lower_is_better:
            comparison_table.sort(key=lambda r: r.get(f"val_{primary_metric}", float('inf')), reverse=False)
        else:
            comparison_table.sort(key=lambda r: r.get(f"val_{primary_metric}", 0.0), reverse=True)

        # ── Select best model ─────────────────────────────────────────────────
        best_row = comparison_table[0]
        best_model_id = best_row["model_id"]
        best_model_name = best_row["algorithm"]
        emit_fn(f"Best model: {best_model_name} (val_{primary_metric}={best_row.get(f'val_{primary_metric}', 'N/A')})", agent=AGENT_NAME, level="success")

        # ── Overfitting analysis ──────────────────────────────────────────────
        overfitting_detected = any(r["is_overfitting"] for r in comparison_table)
        overfitting_details = None
        if overfitting_detected:
            overfit_models = [r["algorithm"] for r in comparison_table if r["is_overfitting"]]
            emit_fn(f"⚠️ Overfitting detected in: {overfit_models}", agent=AGENT_NAME, level="warning")
            overfitting_details = (
                f"Potential overfitting in: {overfit_models}. "
                f"Consider regularization, reducing model complexity, or more data."
            )

        # ── LLM selects best model and explains evaluation ─────────────────────
        table_text = _format_comparison_table(comparison_table, primary_metric)
        prompt = f"""
Model Evaluation Results:

Primary metric: {primary_metric}
Problem type: {problem_type}
Class imbalance: {class_imbalance}

{table_text}

Tasks:
1. Confirm or challenge the automatic selection of {best_model_name} as the best model
2. Comment on the overfitting analysis
3. Explain what the metrics mean for this use case
4. Give a confidence rating (1-10) in the best model's real-world performance
"""
        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content=prompt))
        llm_response = llm.invoke(messages)
        reasoning = llm_response.content

        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        emit_fn(f"✓ Evaluation complete in {duration_ms}ms", agent=AGENT_NAME, level="success")

        return {
            **state,
            "evaluation_results": evaluation_results,
            "model_comparison_table": comparison_table,
            "best_model_id": best_model_id,
            "best_model_name": best_model_name,
            "primary_metric": primary_metric,
            "overfitting_detected": overfitting_detected,
            "overfitting_details": overfitting_details,
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [
                AgentRunRecord(agent_name=AGENT_NAME, status="completed", reasoning=reasoning, duration_ms=duration_ms)
            ],
            "messages": messages + [llm_response],
        }

    except Exception as e:
        logger.exception(f"Evaluation Agent failed: {e}")
        emit_fn(f"❌ Evaluation failed: {e}", agent=AGENT_NAME, level="error")
        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        return {
            **state,
            "evaluation_results": evaluation_results,
            "model_comparison_table": comparison_table,
            "error": str(e),
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [
                AgentRunRecord(agent_name=AGENT_NAME, status="failed", error=str(e), duration_ms=duration_ms)
            ],
        }


def _format_comparison_table(table: list[dict], primary_metric: str) -> str:
    lines = [f"{'Algorithm':<30} {'val_' + primary_metric:<15} {'cv_mean':<12} {'overfit':<10}"]
    lines.append("-" * 70)
    for row in table:
        val = row.get(f"val_{primary_metric}", "N/A")
        cv = row.get("cv_mean", "N/A")
        overfit = "⚠️ YES" if row.get("is_overfitting") else "NO"
        lines.append(f"{row['algorithm']:<30} {str(val):<15} {str(cv):<12} {overfit:<10}")
    return "\n".join(lines)
