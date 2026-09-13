"""
EDA Agent — runs all EDA visualizations and uses LLM to generate insights.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage

from app.agents.state import DataScienceState, AgentRunRecord, EDAInsight
from app.config.settings import settings
from app.tools.eda_tools import (
    plot_boxplots,
    plot_categorical_distributions,
    plot_correlation_heatmap,
    plot_feature_vs_target,
    plot_missing_values,
    plot_numerical_distributions,
    plot_target_distribution,
)

logger = logging.getLogger(__name__)
AGENT_NAME = "eda"


def run(state: DataScienceState, llm, emit_fn, db_session) -> DataScienceState:
    start = datetime.now(timezone.utc)
    emit_fn("📊 EDA Agent started...", agent=AGENT_NAME)

    insights: list[EDAInsight] = []
    viz_paths: list[str] = []

    try:
        file_path = state.get("cleaned_file_path") or state.get("working_file_path")
        df = pd.read_csv(file_path, low_memory=False)

        target_col = state.get("target_column")
        numerical_cols = state.get("numerical_columns", [])
        categorical_cols = state.get("categorical_columns", [])
        problem_type = state.get("problem_type", "classification")

        # Create output dir for charts
        chart_dir = Path(settings.upload_dir) / str(state.get("user_id", "unknown")) / "charts" / state["experiment_id"]
        chart_dir.mkdir(parents=True, exist_ok=True)
        chart_dir_str = str(chart_dir)

        emit_fn(f"Generating visualizations in {chart_dir_str}...", agent=AGENT_NAME)

        def add_insight(path: str, chart_type: str, columns: list[str], raw_insight: str, importance: str = "medium"):
            if path:
                insights.append(EDAInsight(
                    chart_type=chart_type,
                    columns=columns,
                    chart_path=path,
                    insight=raw_insight,
                    importance=importance,
                ))
                viz_paths.append(path)

        # ── 1. Target distribution ─────────────────────────────────────────
        if target_col and target_col in df.columns:
            emit_fn(f"Plotting target distribution for '{target_col}'...", agent=AGENT_NAME)
            path, insight = plot_target_distribution(df, target_col, chart_dir_str)
            add_insight(path, "bar_chart", [target_col], insight, "high")

        # ── 2. Missing values ──────────────────────────────────────────────
        emit_fn("Plotting missing values chart...", agent=AGENT_NAME)
        path, insight = plot_missing_values(df, chart_dir_str)
        add_insight(path, "bar_chart", ["all"], insight, "high")

        # ── 3. Correlation heatmap ─────────────────────────────────────────
        if len(numerical_cols) >= 2:
            emit_fn("Plotting correlation heatmap...", agent=AGENT_NAME)
            path, insight = plot_correlation_heatmap(df, chart_dir_str)
            add_insight(path, "heatmap", numerical_cols[:20], insight, "high")

        # ── 4. Numerical distributions ────────────────────────────────────
        if numerical_cols:
            emit_fn("Plotting numerical distributions...", agent=AGENT_NAME)
            results = plot_numerical_distributions(df, numerical_cols, chart_dir_str)
            for path, insight in results:
                add_insight(path, "histogram", numerical_cols[:12], insight)

        # ── 5. Box plots ──────────────────────────────────────────────────
        if numerical_cols:
            emit_fn("Plotting box plots...", agent=AGENT_NAME)
            results = plot_boxplots(df, numerical_cols, target_col, chart_dir_str)
            for path, insight in results:
                add_insight(path, "boxplot", numerical_cols[:10], insight)

        # ── 6. Categorical distributions ──────────────────────────────────
        if categorical_cols:
            emit_fn("Plotting categorical distributions...", agent=AGENT_NAME)
            results = plot_categorical_distributions(df, categorical_cols, chart_dir_str)
            for path, insight in results:
                add_insight(path, "bar_chart", [path.split("cat_")[1].split(".")[0] if "cat_" in path else "unknown"], insight)

        # ── 7. Feature vs target ───────────────────────────────────────────
        if target_col and numerical_cols:
            emit_fn("Plotting feature vs target relationships...", agent=AGENT_NAME)
            results = plot_feature_vs_target(df, numerical_cols, target_col, problem_type, chart_dir_str)
            for path, insight in results:
                add_insight(path, "scatter" if problem_type == "regression" else "boxplot",
                            numerical_cols[:6] + [target_col], insight, "high")

        emit_fn(f"Generated {len(viz_paths)} visualizations", agent=AGENT_NAME)

        # ── LLM generates EDA summary ─────────────────────────────────────
        insight_text = "\n".join([f"- [{i.chart_type}] {i.insight}" for i in insights])
        prompt = f"""
You are a Data Scientist reviewing EDA results.

Dataset summary:
- Shape: {df.shape}
- Target column: {target_col}
- Problem type: {problem_type}

EDA Findings:
{insight_text}

Write a comprehensive EDA summary (3-5 paragraphs) covering:
1. Overall data distribution and quality
2. Key feature patterns and relationships
3. Class balance / target distribution insights
4. Potential feature engineering opportunities
5. Recommendations for modeling
"""
        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content=prompt))
        llm_response = llm.invoke(messages)
        eda_summary = llm_response.content

        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        emit_fn(f"✓ EDA complete in {duration_ms}ms — {len(insights)} insights generated", agent=AGENT_NAME, level="success")

        return {
            **state,
            "eda_insights": insights,
            "visualization_paths": viz_paths,
            "eda_summary": eda_summary,
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [
                AgentRunRecord(agent_name=AGENT_NAME, status="completed", reasoning=eda_summary[:500], duration_ms=duration_ms)
            ],
            "messages": messages + [llm_response],
        }

    except Exception as e:
        logger.exception(f"EDA Agent failed: {e}")
        emit_fn(f"❌ EDA failed: {e}", agent=AGENT_NAME, level="error")
        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        return {
            **state,
            "eda_insights": insights,
            "visualization_paths": viz_paths,
            "error": str(e),
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [
                AgentRunRecord(agent_name=AGENT_NAME, status="failed", error=str(e), duration_ms=duration_ms)
            ],
        }
