"""
Data Understanding Agent.
Runs deterministic profiling and then uses the LLM to interpret results
and write human-readable reasoning.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage

from app.agents.state import DataScienceState, AgentRunRecord
from app.tools.profiling import profile_dataframe, load_csv_safe

logger = logging.getLogger(__name__)

AGENT_NAME = "data_understanding"


def run(state: DataScienceState, llm, emit_fn, db_session) -> DataScienceState:
    """
    Data Understanding Agent node.
    1. Load the dataset and run deterministic profiling.
    2. Pass the profile to the LLM for interpretation.
    3. Update state with profile data and reasoning.
    """
    start = datetime.now(timezone.utc)
    emit_fn("🔍 Data Understanding Agent started...", agent=AGENT_NAME)

    try:
        # ── Step 1: Deterministic profiling (no LLM) ──────────────────────────
        file_path = state["original_file_path"]
        target_col = state.get("target_column")

        emit_fn(f"Loading dataset from {file_path}...", agent=AGENT_NAME)
        df, encoding = load_csv_safe(file_path)
        emit_fn(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns (encoding: {encoding})", agent=AGENT_NAME)

        emit_fn("Running dataset profiling...", agent=AGENT_NAME)
        profile = profile_dataframe(df, target_col=target_col)

        emit_fn(
            f"Profile complete: {profile.total_missing} missing values, "
            f"{profile.duplicate_rows} duplicates, "
            f"{len(profile.data_quality_issues)} quality issues detected",
            agent=AGENT_NAME,
        )

        # ── Step 2: LLM interprets the profile ───────────────────────────────
        profile_summary = _build_profile_summary(profile)
        prompt = f"""
You are an expert Data Scientist reviewing a dataset profile.

Dataset Profile:
{profile_summary}

Tasks:
1. Confirm or correct the suggested target column: "{profile.target_column_suggestion}" (confidence: {profile.target_confidence:.0%})
2. Confirm the detected problem type: "{profile.problem_type_suggestion}"
3. List the top 3 data quality concerns.
4. Describe any notable patterns you observe.
5. Flag any potential data leakage risks.

Be concise and specific. Base your analysis ONLY on the profile data provided.
"""
        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content=prompt))
        llm_response = llm.invoke(messages)
        reasoning = llm_response.content

        emit_fn(f"AI Analysis: {reasoning[:200]}...", agent=AGENT_NAME, level="info")

        # ── Step 3: Build updated state ───────────────────────────────────────
        profile_dict = profile.to_dict()
        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)

        agent_record = AgentRunRecord(
            agent_name=AGENT_NAME,
            status="completed",
            reasoning=reasoning,
            duration_ms=duration_ms,
        )

        emit_fn(
            f"✓ Data Understanding complete in {duration_ms}ms — "
            f"Target: {profile.target_column_suggestion} | "
            f"Problem: {profile.problem_type_suggestion}",
            agent=AGENT_NAME,
            level="success",
        )

        return {
            **state,
            "profile_json": profile_dict,
            "df_shape": (profile.row_count, profile.col_count),
            "numerical_columns": profile.numerical_columns,
            "categorical_columns": profile.categorical_columns,
            "boolean_columns": profile.boolean_columns,
            "datetime_columns": profile.datetime_columns,
            "id_columns": profile.id_columns,
            "constant_columns": profile.constant_columns,
            "target_column": profile.target_column_suggestion if not state.get("target_confirmed") else state["target_column"],
            "target_confidence": profile.target_confidence,
            "problem_type": profile.problem_type_suggestion,
            "class_distribution": profile.class_distribution,
            "class_imbalance": profile.class_imbalance,
            "data_quality_issues": profile.data_quality_issues,
            "leakage_warnings": [
                f"Column '{col.name}' has {col.leakage_risk} leakage risk"
                for col in profile.columns
                if col.leakage_risk in ("medium", "high") and col.name != profile.target_column_suggestion
            ],
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [agent_record],
            "messages": messages + [llm_response],
        }

    except Exception as e:
        logger.exception(f"Data Understanding Agent failed: {e}")
        emit_fn(f"❌ Data Understanding failed: {e}", agent=AGENT_NAME, level="error")
        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        return {
            **state,
            "error": str(e),
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [
                AgentRunRecord(agent_name=AGENT_NAME, status="failed", error=str(e), duration_ms=duration_ms)
            ],
        }


def _build_profile_summary(profile) -> str:
    """Convert profile to a concise text summary for the LLM."""
    lines = [
        f"Rows: {profile.row_count}, Columns: {profile.col_count}",
        f"Missing values: {profile.total_missing} ({profile.total_missing_pct:.1f}%)",
        f"Duplicate rows: {profile.duplicate_rows} ({profile.duplicate_pct:.1f}%)",
        f"Memory usage: {profile.memory_mb:.1f} MB",
        f"Numerical columns ({len(profile.numerical_columns)}): {profile.numerical_columns[:10]}",
        f"Categorical columns ({len(profile.categorical_columns)}): {profile.categorical_columns[:10]}",
        f"Boolean columns: {profile.boolean_columns}",
        f"Datetime columns: {profile.datetime_columns}",
        f"ID columns (likely to drop): {profile.id_columns}",
        f"Constant columns (to drop): {profile.constant_columns}",
        f"Near-zero variance columns: {profile.near_zero_variance_columns}",
        f"Suggested target: {profile.target_column_suggestion} (confidence: {profile.target_confidence:.0%})",
        f"Detected problem type: {profile.problem_type_suggestion}",
    ]
    if profile.class_distribution:
        lines.append(f"Class distribution: {profile.class_distribution}")
        lines.append(f"Class imbalance: {profile.class_imbalance}")

    if profile.data_quality_issues:
        lines.append("Data quality issues:")
        for issue in profile.data_quality_issues:
            lines.append(f"  - {issue}")

    if profile.correlation_matrix:
        top_corrs = list(profile.correlation_matrix.items())[:5]
        lines.append("Top correlations:")
        for pair, val in top_corrs:
            lines.append(f"  - {pair}: {val:.3f}")

    # Per-column details (top 15)
    lines.append("\nColumn details (sample):")
    for col in profile.columns[:15]:
        line = f"  {col.name} [{col.semantic_type}]: nulls={col.null_pct:.1f}%, unique={col.unique_count}"
        if col.leakage_risk != "none":
            line += f" ⚠️ leakage_risk={col.leakage_risk}"
        lines.append(line)

    return "\n".join(lines)
