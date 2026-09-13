"""
Report Generation Agent.
Synthesizes all findings across profiling, cleaning, EDA, modeling, and explainability
into an executive narrative using the LLM and produces the final PDF report.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage

from app.agents.state import DataScienceState, AgentRunRecord
from app.config.settings import settings
from app.tools.report_tools import generate_pdf_report

logger = logging.getLogger(__name__)
AGENT_NAME = "reporting"


def run(state: DataScienceState, llm, emit_fn, db_session) -> DataScienceState:
    start = datetime.now(timezone.utc)
    emit_fn("Report Agent started", agent=AGENT_NAME)

    try:
        experiment_id = state["experiment_id"]
        user_id = str(state.get("user_id", "unknown"))
        problem_type = state.get("problem_type", "classification")
        target_col = state.get("target_column")
        primary_metric = state.get("primary_metric", "f1_weighted")
        best_model_name = state.get("best_model_name", "Champion Model")
        df_shape = state.get("df_shape", (0, 0))

        # Output directory for reports
        report_dir = Path(settings.reports_dir) / user_id
        report_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = str(report_dir / f"Report_{experiment_id}.pdf")

        # ── LLM generates Executive Recommendations & Next Steps ─────────────
        comparison_table = state.get("model_comparison_table", [])
        shap_res = state.get("shap_results")
        feature_importance = state.get("feature_importance", [])

        prompt = f"""
You are the Lead AI Data Scientist finalizing the executive report.

Experiment Overview:
- Problem Type: {problem_type}
- Target: {target_col}
- Dataset Shape: {df_shape[0]} rows, {df_shape[1]} features
- Best Performing Algorithm: {best_model_name}
- Primary Metric: {primary_metric}
- Overfitting Detected: {state.get('overfitting_detected', False)}
- Top Features: {[f.get('feature', f.get('feature_name', 'Unknown')) for f in feature_importance[:5]] if feature_importance else 'N/A'}

Write 2 concise paragraphs:
1. Executive interpretation of model readiness and business impact.
2. Concrete next steps, production monitoring guidelines, and data collection recommendations.
"""
        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content=prompt))
        try:
            llm_response = llm.invoke(messages)
            recommendations_text = str(llm_response.content).strip()
        except Exception as llm_error:
            # Report generation is deterministic and must remain available when an
            # optional external LLM is offline or rate-limited.
            logger.warning("Could not generate LLM recommendations: %s", llm_error)
            llm_response = None
            recommendations_text = (
                f"The selected model, {best_model_name}, should be monitored against "
                f"the {primary_metric} validation baseline before production release. "
                "Track input drift, prediction quality, missing-data rates, and retrain "
                "when the production metric materially degrades."
            )

        # ── Call PDF Generation Tool ──────────────────────────────────────────
        emit_fn(f"Compiling PDF report at {pdf_path}...", agent=AGENT_NAME)

        shap_plots = []
        if shap_res:
            if getattr(shap_res, "summary_plot_path", None):
                shap_plots.append(shap_res.summary_plot_path)
            if getattr(shap_res, "bar_plot_path", None):
                shap_plots.append(shap_res.bar_plot_path)

        # Get best model metrics
        best_eval = None
        best_id = state.get("best_model_id")
        if best_id and state.get("evaluation_results"):
            eval_obj = state["evaluation_results"].get(best_id)
            if eval_obj:
                best_eval = eval_obj.all_metrics

        best_eval = best_eval or {}

        model_details = [
            {
                "algorithm": model.algorithm,
                "artifact_path": model.artifact_path,
                "hyperparameters": model.hyperparameters,
                "metrics": model.metrics,
                "training_time_seconds": model.training_time_seconds,
            }
            for model in state.get("trained_models", [])
        ]

        generated_path = generate_pdf_report(
            output_path=pdf_path,
            experiment_id=experiment_id,
            dataset_name=Path(state.get("original_file_path", "Dataset")).stem,
            problem_type=problem_type,
            target_col=target_col,
            primary_metric=primary_metric,
            df_shape=df_shape,
            cleaning_summary=f"{len(state.get('cleaning_log', []))} transformations applied (imputation, scaling, encoding).",
            eda_summary=state.get("eda_summary", "Exploratory analysis identified key distribution and correlation insights."),
            chart_paths=state.get("visualization_paths", []),
            model_comparison_table=comparison_table,
            best_model_name=best_model_name,
            best_model_metrics=best_eval,
            shap_plot_paths=shap_plots,
            feature_importance=feature_importance,
            overfitting_analysis=state.get("overfitting_details"),
            leakage_warnings=state.get("leakage_warnings", []),
            recommendations=recommendations_text,
            model_details=model_details,
        )
        if not Path(generated_path).is_file() or Path(generated_path).stat().st_size == 0:
            raise RuntimeError("PDF generator did not produce a readable report file")

        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        emit_fn(f"PDF report generated in {duration_ms}ms", agent=AGENT_NAME, level="success")

        agent_record = AgentRunRecord(
            agent_name=AGENT_NAME,
            status="completed",
            reasoning=recommendations_text[:400],
            duration_ms=duration_ms,
        )

        return {
            **state,
            "report_path": pdf_path,
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [agent_record],
            "messages": messages + ([llm_response] if llm_response else []),
        }

    except Exception as e:
        logger.exception(f"Report Agent failed: {e}")
        emit_fn(f"Report generation failed: {e}", agent=AGENT_NAME, level="error")
        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        return {
            **state,
            "error": f"Report generation failed: {e}",
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [
                AgentRunRecord(agent_name=AGENT_NAME, status="failed", error=str(e), duration_ms=duration_ms)
            ],
        }
