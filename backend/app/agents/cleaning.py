"""
Data Cleaning Agent.
Orchestrates the deterministic cleaning pipeline using cleaning_tools.py.
The LLM interprets results and generates human-readable reasoning.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage

from app.agents.state import DataScienceState, AgentRunRecord, PreprocessingStepRecord
from app.config.settings import settings
from app.tools.cleaning_tools import (
    choose_imputation_strategy,
    choose_scaler,
    drop_columns,
    encode_one_hot,
    handle_outliers_clip,
    impute_categorical,
    impute_numerical,
    remove_duplicates,
)

logger = logging.getLogger(__name__)
AGENT_NAME = "cleaning"


def run(state: DataScienceState, llm, emit_fn, db_session) -> DataScienceState:
    start = datetime.now(timezone.utc)
    emit_fn("🧹 Data Cleaning Agent started...", agent=AGENT_NAME)

    cleaning_log: list[PreprocessingStepRecord] = []
    step_order = 0

    def log_step(result, semantic_type: str = "", operation_override: str = ""):
        nonlocal step_order
        step_order += 1
        cleaning_log.append(PreprocessingStepRecord(
            step_order=step_order,
            step_type=semantic_type or result.operation,
            column_name=result.column,
            operation=operation_override or result.operation,
            parameters=result.parameters,
            reason=result.details,
            rows_affected=result.rows_affected,
        ))

    try:
        df = pd.read_csv(state["original_file_path"], low_memory=False)
        original_shape = df.shape
        emit_fn(f"Dataset shape: {original_shape[0]} rows × {original_shape[1]} columns", agent=AGENT_NAME)

        profile = state.get("profile_json", {})
        target_col = state.get("target_column")
        numerical_cols = state.get("numerical_columns", [])
        categorical_cols = state.get("categorical_columns", [])
        id_cols = state.get("id_columns", [])
        constant_cols = state.get("constant_columns", [])
        dropped = []

        # ── 1. Remove duplicates ──────────────────────────────────────────────
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            emit_fn(f"Removing {dup_count} duplicate rows...", agent=AGENT_NAME)
            df, result = remove_duplicates(df)
            log_step(result, "duplicates")

        # ── 2. Drop constant columns ──────────────────────────────────────────
        if constant_cols:
            emit_fn(f"Dropping {len(constant_cols)} constant columns: {constant_cols}", agent=AGENT_NAME)
            df, results = drop_columns(df, constant_cols, "Constant column — zero variance")
            for r in results:
                log_step(r, "feature_removal")
            dropped.extend(constant_cols)

        # ── 3. Drop ID columns ────────────────────────────────────────────────
        id_to_drop = [c for c in id_cols if c != target_col]
        if id_to_drop:
            emit_fn(f"Dropping ID columns: {id_to_drop}", agent=AGENT_NAME)
            df, results = drop_columns(df, id_to_drop, "ID column — not predictive")
            for r in results:
                log_step(r, "feature_removal")
            dropped.extend(id_to_drop)

        # ── 4. Handle missing values (numerical) ──────────────────────────────
        col_profiles = {cp["name"]: cp for cp in profile.get("columns", [])}
        for col in numerical_cols:
            if col not in df.columns or col == target_col:
                continue
            null_pct = df[col].isnull().mean() * 100
            if null_pct > 0:
                strategy = choose_imputation_strategy(df, col, "numerical", null_pct)
                emit_fn(f"Imputing '{col}': {null_pct:.1f}% missing → {strategy}", agent=AGENT_NAME)
                df, result = impute_numerical(df, col, strategy)
                log_step(result, "imputation")

        # ── 5. Handle missing values (categorical) ────────────────────────────
        for col in categorical_cols:
            if col not in df.columns or col == target_col:
                continue
            null_pct = df[col].isnull().mean() * 100
            if null_pct > 0:
                strategy = choose_imputation_strategy(df, col, "categorical", null_pct)
                emit_fn(f"Imputing '{col}': {null_pct:.1f}% missing → {strategy}", agent=AGENT_NAME)
                df, result = impute_categorical(df, col, strategy)
                log_step(result, "imputation")

        # ── 6. Handle outliers (numerical, non-target) ────────────────────────
        for col in numerical_cols:
            if col not in df.columns or col == target_col:
                continue
            from app.tools.cleaning_tools import detect_outliers_iqr
            outlier_mask = detect_outliers_iqr(df, col)
            outlier_pct = outlier_mask.mean() * 100
            if outlier_pct > 0.5:
                emit_fn(f"Clipping outliers in '{col}': {outlier_pct:.1f}% outlier rate", agent=AGENT_NAME)
                df, result = handle_outliers_clip(df, col)
                log_step(result, "outlier_handling")

        # ── 7. Drop leakage columns (high risk) ───────────────────────────────
        leakage_high = [
            cp["name"] for cp in profile.get("columns", [])
            if cp.get("leakage_risk") == "high" and cp["name"] != target_col
        ]
        if leakage_high:
            emit_fn(f"⚠️ Dropping high-risk leakage columns: {leakage_high}", agent=AGENT_NAME, level="warning")
            df, results = drop_columns(df, leakage_high, "High data leakage risk — removed before training")
            for r in results:
                log_step(r, "leakage_removal")
            dropped.extend(leakage_high)

        # ── 8. One-hot encode categorical columns (non-target) ────────────────
        cat_to_encode = [c for c in categorical_cols if c in df.columns and c != target_col]
        if cat_to_encode:
            emit_fn(f"One-hot encoding {len(cat_to_encode)} categorical columns...", agent=AGENT_NAME)
            df, results = encode_one_hot(df, cat_to_encode)
            for r in results:
                log_step(r, "encoding")

        # ── 9. Save cleaned dataset ───────────────────────────────────────────
        user_dir = Path(settings.upload_dir) / str(state.get("user_id", "unknown"))
        user_dir.mkdir(parents=True, exist_ok=True)
        cleaned_path = str(user_dir / f"{state['experiment_id']}_cleaned.csv")
        df.to_csv(cleaned_path, index=False)

        emit_fn(
            f"Cleaned dataset saved: {df.shape[0]} rows × {df.shape[1]} columns → {cleaned_path}",
            agent=AGENT_NAME,
        )

        # ── 10. LLM summarizes cleaning decisions ─────────────────────────────
        cleaning_summary = _build_cleaning_summary(cleaning_log, original_shape, df.shape)
        prompt = f"""
You performed the following data cleaning operations:

{cleaning_summary}

Provide a concise paragraph explaining:
1. The key decisions made and why
2. Any trade-offs or concerns
3. Overall data quality after cleaning
"""
        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content=prompt))
        llm_response = llm.invoke(messages)
        reasoning = llm_response.content

        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        emit_fn(f"✓ Cleaning complete in {duration_ms}ms — {len(cleaning_log)} operations performed", agent=AGENT_NAME, level="success")

        return {
            **state,
            "cleaning_log": cleaning_log,
            "cleaned_file_path": cleaned_path,
            "working_file_path": cleaned_path,
            "dropped_columns": dropped,
            "numerical_columns": [c for c in numerical_cols if c in df.columns],
            "categorical_columns": [c for c in categorical_cols if c in df.columns],
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [
                AgentRunRecord(agent_name=AGENT_NAME, status="completed", reasoning=reasoning, duration_ms=duration_ms)
            ],
            "messages": messages + [llm_response],
        }

    except Exception as e:
        logger.exception(f"Cleaning Agent failed: {e}")
        emit_fn(f"❌ Cleaning failed: {e}", agent=AGENT_NAME, level="error")
        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        return {
            **state,
            "cleaning_log": cleaning_log,
            "error": str(e),
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [
                AgentRunRecord(agent_name=AGENT_NAME, status="failed", error=str(e), duration_ms=duration_ms)
            ],
        }


def _build_cleaning_summary(log: list[PreprocessingStepRecord], original: tuple, final: tuple) -> str:
    lines = [
        f"Original shape: {original[0]} rows × {original[1]} columns",
        f"Final shape: {final[0]} rows × {final[1]} columns",
        "",
        "Operations performed:",
    ]
    for step in log:
        lines.append(f"  {step.step_order}. [{step.step_type}] {step.column_name or 'all'}: {step.reason}")
    return "\n".join(lines)
