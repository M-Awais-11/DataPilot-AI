"""
Feature Engineering Agent.
Applies useful transformations, then selects the most predictive features.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, f_classif, f_regression

from app.agents.state import DataScienceState, AgentRunRecord, FeatureOperation
from app.config.settings import settings

logger = logging.getLogger(__name__)
AGENT_NAME = "feature_engineering"


def run(state: DataScienceState, llm, emit_fn, db_session) -> DataScienceState:
    start = datetime.now(timezone.utc)
    emit_fn("⚙️ Feature Engineering Agent started...", agent=AGENT_NAME)

    operations: list[FeatureOperation] = []

    try:
        file_path = state.get("cleaned_file_path") or state.get("working_file_path")
        df = pd.read_csv(file_path, low_memory=False)
        original_cols = list(df.columns)

        target_col = state.get("target_column")
        problem_type = state.get("problem_type", "classification")
        numerical_cols = [c for c in state.get("numerical_columns", []) if c in df.columns and c != target_col]

        # ── 1. Log transformations for skewed columns ─────────────────────────
        log_transformed = []
        for col in numerical_cols:
            skewness = df[col].skew()
            if abs(skewness) > 1.5 and df[col].min() > 0:
                df[f"{col}_log"] = np.log1p(df[col])
                log_transformed.append(col)

        if log_transformed:
            emit_fn(f"Log-transformed {len(log_transformed)} skewed columns: {log_transformed}", agent=AGENT_NAME)
            operations.append(FeatureOperation(
                operation="log_transform",
                columns=log_transformed,
                new_columns=[f"{c}_log" for c in log_transformed],
                reason="Applied log1p transformation to reduce skewness (|skew| > 1.5)",
            ))

        # ── 2. Interaction features (top correlated pairs) ────────────────────
        if len(numerical_cols) >= 2 and target_col and target_col in df.columns:
            try:
                corr = df[numerical_cols].corr().abs()
                pairs = []
                for i, c1 in enumerate(numerical_cols[:10]):
                    for c2 in numerical_cols[i + 1:10]:
                        pairs.append((c1, c2, corr.loc[c1, c2]))
                pairs.sort(key=lambda x: x[2], reverse=True)
                top_pairs = [(c1, c2) for c1, c2, _ in pairs[:3]]
                new_interaction = []
                for c1, c2 in top_pairs:
                    feat_name = f"{c1}_x_{c2}"
                    df[feat_name] = df[c1] * df[c2]
                    new_interaction.append(feat_name)

                if new_interaction:
                    emit_fn(f"Created {len(new_interaction)} interaction features", agent=AGENT_NAME)
                    operations.append(FeatureOperation(
                        operation="interaction_features",
                        columns=[c1 for c1, _ in top_pairs] + [c2 for _, c2 in top_pairs],
                        new_columns=new_interaction,
                        reason="Created product interaction features for top correlated numerical pairs",
                    ))
            except Exception as e:
                logger.warning(f"Interaction features failed: {e}")

        # ── 3. Feature Selection with Mutual Information ───────────────────────
        selected_features = _select_features(df, target_col, problem_type, emit_fn)

        # ── 4. Save feature-engineered dataset ───────────────────────────────
        user_dir = Path(settings.upload_dir) / str(state.get("user_id", "unknown"))
        fe_path = str(user_dir / f"{state['experiment_id']}_features.csv")
        df.to_csv(fe_path, index=False)
        emit_fn(f"Feature-engineered dataset: {df.shape[0]} rows × {df.shape[1]} columns", agent=AGENT_NAME)

        # ── 5. LLM summarizes feature engineering ─────────────────────────────
        new_cols = [c for c in df.columns if c not in original_cols]
        dropped_feats = [c for c in original_cols if c not in selected_features and c != target_col]

        ops_summary = "\n".join([
            f"- {op.operation}: {op.columns} → {op.new_columns}. Reason: {op.reason}"
            for op in operations
        ])
        prompt = f"""
Feature Engineering Summary:

New features created: {new_cols}
Features selected for modeling: {selected_features[:20]}
Features dropped from selection: {dropped_feats[:10]}

Operations performed:
{ops_summary}

Explain (2-3 sentences):
1. Why these features were engineered
2. Why the feature selection decisions make sense
3. Expected impact on model performance
"""
        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content=prompt))
        llm_response = llm.invoke(messages)
        reasoning = llm_response.content

        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        emit_fn(f"✓ Feature Engineering complete in {duration_ms}ms — {len(selected_features)} features selected", agent=AGENT_NAME, level="success")

        return {
            **state,
            "feature_engineering_log": operations,
            "selected_features": selected_features,
            "dropped_features": dropped_feats,
            "feature_engineered_file_path": fe_path,
            "working_file_path": fe_path,
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [
                AgentRunRecord(agent_name=AGENT_NAME, status="completed", reasoning=reasoning, duration_ms=duration_ms)
            ],
            "messages": messages + [llm_response],
        }

    except Exception as e:
        logger.exception(f"Feature Engineering Agent failed: {e}")
        emit_fn(f"❌ Feature Engineering failed: {e}", agent=AGENT_NAME, level="error")
        duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        return {
            **state,
            "feature_engineering_log": operations,
            "error": str(e),
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [
                AgentRunRecord(agent_name=AGENT_NAME, status="failed", error=str(e), duration_ms=duration_ms)
            ],
        }


def _select_features(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    emit_fn,
    max_features: int = 50,
) -> list[str]:
    """Use Mutual Information for feature selection. Returns list of selected feature names."""
    if not target_col or target_col not in df.columns:
        return [c for c in df.columns if c != target_col]

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].select_dtypes(include=np.number)
    y = df[target_col]

    if len(X.columns) == 0:
        return feature_cols

    # Handle NaN
    X = X.fillna(X.median())
    y = y.fillna(y.mode().iloc[0] if problem_type == "classification" else y.median())

    try:
        if problem_type == "classification":
            mi_func = mutual_info_classif
        else:
            mi_func = mutual_info_regression

        mi_scores = mi_func(X, y, random_state=settings.random_seed)
        scores = dict(zip(X.columns.tolist(), mi_scores))
        sorted_feats = sorted(scores, key=lambda k: scores[k], reverse=True)

        # Keep top N or all features with MI > 0
        selected = [f for f in sorted_feats if scores[f] > 0][:max_features]

        # Add non-numerical features back
        non_num = [c for c in feature_cols if c not in X.columns]
        selected = selected + non_num

        emit_fn(f"Feature selection: kept {len(selected)} / {len(feature_cols)} features via Mutual Information", agent=AGENT_NAME)
        return selected

    except Exception as e:
        logger.warning(f"Mutual information selection failed: {e}. Using all features.")
        return feature_cols
