"""
Training Agent.
Trains candidate models with cross-validation and Optuna optimization.
All computation is deterministic Python/sklearn — LLM only summarizes results.
"""
from __future__ import annotations

import logging
import pickle
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder

from app.agents.state import DataScienceState, AgentRunRecord, TrainedModelResult
from app.config.settings import settings
from app.ml.classifiers import get_classifier, get_regressor
from app.ml.optimization import optimize_hyperparameters

logger = logging.getLogger(__name__)
AGENT_NAME = "training"


def run(state: DataScienceState, llm, emit_fn, db_session) -> DataScienceState:
    start_time = datetime.now(timezone.utc)
    emit_fn("🏋️ Training Agent started...", agent=AGENT_NAME)

    trained_models: list[TrainedModelResult] = []

    try:
        file_path = state.get("feature_engineered_file_path") or state.get("working_file_path")
        df = pd.read_csv(file_path, low_memory=False)

        target_col = state.get("target_column")
        problem_type = state.get("problem_type", "classification")
        selected_features = state.get("selected_features", [])
        candidate_models = state.get("candidate_models", [])
        primary_metric = state.get("primary_metric")
        if not primary_metric or primary_metric not in ("f1_weighted", "accuracy", "roc_auc", "rmse", "r2", "mae"):
            primary_metric = "f1_weighted" if problem_type == "classification" else "rmse"
        elif problem_type == "regression" and primary_metric in ("f1_weighted", "accuracy", "roc_auc"):
            primary_metric = "rmse"
        elif problem_type == "classification" and primary_metric in ("rmse", "r2", "mae"):
            primary_metric = "f1_weighted"

        if not target_col or target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        # ── Prepare features ───────────────────────────────────────────────────
        available_features = [c for c in selected_features if c in df.columns and c != target_col]
        if not available_features:
            available_features = [c for c in df.columns if c != target_col]

        X = df[available_features].select_dtypes(include=np.number).fillna(0)
        y = df[target_col].fillna(df[target_col].mode().iloc[0] if problem_type == "classification" else df[target_col].median())

        # Label encode target for classification
        label_encoder = None
        if problem_type == "classification":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)
            label_encoder = le

        emit_fn(f"Features: {X.shape[1]}, Samples: {X.shape[0]}", agent=AGENT_NAME)

        # ── Train/val/test split ───────────────────────────────────────────────
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=settings.test_size,
            random_state=settings.random_seed,
            stratify=y if problem_type == "classification" else None,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=settings.validation_size / (1 - settings.test_size),
            random_state=settings.random_seed,
            stratify=y_temp if problem_type == "classification" else None,
        )
        emit_fn(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}", agent=AGENT_NAME)

        # Save split data for evaluation agent
        model_dir = Path(settings.upload_dir) / str(state.get("user_id", "unknown")) / "models" / state["experiment_id"]
        model_dir.mkdir(parents=True, exist_ok=True)

        # ── Cross-validation setup ─────────────────────────────────────────────
        if problem_type == "classification":
            cv = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_seed)
        else:
            cv = KFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_seed)

        sklearn_metric = _map_primary_metric(primary_metric, problem_type)

        # ── Train each candidate model ────────────────────────────────────────
        for algo_name in candidate_models:
            emit_fn(f"Training {algo_name}...", agent=AGENT_NAME)
            t0 = time.time()

            try:
                # Hyperparameter optimization (reduced trials for speed)
                opt_trials = min(settings.optuna_trials, 30 if X_train.shape[0] > 10000 else 20)
                best_params = optimize_hyperparameters(
                    algo_name, X_train, y_train, problem_type,
                    primary_metric=primary_metric,
                    n_trials=opt_trials,
                    emit_fn=emit_fn,
                )

                # Train final model with best params
                if problem_type == "classification":
                    model = get_classifier(algo_name, best_params)
                else:
                    model = get_regressor(algo_name, best_params)

                model.fit(X_train, y_train)

                # Cross-validation scores
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=sklearn_metric, n_jobs=-1)

                # Compute metrics on all splits
                metrics = _compute_metrics(model, X_train, y_train, X_val, y_val, X_test, y_test, problem_type)
                metrics["cv_mean"] = {primary_metric: float(np.mean(cv_scores))}
                metrics["cv_std"] = {primary_metric: float(np.std(cv_scores))}

                training_time = time.time() - t0
                emit_fn(
                    f"✓ {algo_name}: val_{primary_metric}={metrics.get('validation', {}).get(primary_metric, 0):.4f} "
                    f"(cv={np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}) [{training_time:.1f}s]",
                    agent=AGENT_NAME, level="success"
                )

                # Save model artifact
                model_id = str(uuid.uuid4())
                artifact_path = str(model_dir / f"{algo_name}_{model_id}.pkl")
                with open(artifact_path, "wb") as f:
                    pickle.dump({"model": model, "label_encoder": label_encoder, "features": list(X.columns)}, f)

                trained_models.append(TrainedModelResult(
                    model_id=model_id,
                    algorithm=algo_name,
                    hyperparameters=best_params,
                    artifact_path=artifact_path,
                    metrics=metrics,
                    cv_scores={primary_metric: cv_scores.tolist()},
                    training_time_seconds=training_time,
                ))

            except Exception as e:
                logger.warning(f"Failed to train {algo_name}: {e}")
                emit_fn(f"⚠️ {algo_name} training failed: {e}", agent=AGENT_NAME, level="warning")

        if not trained_models:
            raise RuntimeError("All model training attempts failed")

        # ── LLM summarizes training results ────────────────────────────────────
        results_text = _format_results(trained_models, primary_metric)
        prompt = f"""
Model training is complete. Here are the results:

{results_text}

Primary metric: {primary_metric}

Provide a concise training summary (2-3 sentences):
1. Which model performed best and by how much?
2. Are there signs of overfitting (large gap between train and val)?
3. Any recommendations for further improvement?
"""
        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content=prompt))
        llm_response = llm.invoke(messages)
        reasoning = llm_response.content

        duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        emit_fn(f"✓ Training complete in {duration_ms}ms — {len(trained_models)} models trained", agent=AGENT_NAME, level="success")

        return {
            **state,
            "trained_models": trained_models,
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [
                AgentRunRecord(agent_name=AGENT_NAME, status="completed", reasoning=reasoning, duration_ms=duration_ms)
            ],
            "messages": messages + [llm_response],
        }

    except Exception as e:
        logger.exception(f"Training Agent failed: {e}")
        emit_fn(f"❌ Training failed: {e}", agent=AGENT_NAME, level="error")
        duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        return {
            **state,
            "trained_models": trained_models,
            "error": str(e),
            "current_agent": AGENT_NAME,
            "agent_history": list(state.get("agent_history", [])) + [
                AgentRunRecord(agent_name=AGENT_NAME, status="failed", error=str(e), duration_ms=duration_ms)
            ],
        }


def _compute_metrics(model, X_train, y_train, X_val, y_val, X_test, y_test, problem_type: str) -> dict:
    """Compute train/val/test metrics."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
        mean_absolute_error, mean_squared_error, r2_score,
    )
    result = {}
    splits = [("train", X_train, y_train), ("validation", X_val, y_val), ("test", X_test, y_test)]

    for split_name, X, y in splits:
        pred = model.predict(X)
        m = {}

        if problem_type == "classification":
            m["accuracy"] = round(float(accuracy_score(y, pred)), 4)
            m["f1_weighted"] = round(float(f1_score(y, pred, average="weighted", zero_division=0)), 4)
            m["precision_weighted"] = round(float(precision_score(y, pred, average="weighted", zero_division=0)), 4)
            m["recall_weighted"] = round(float(recall_score(y, pred, average="weighted", zero_division=0)), 4)
            try:
                proba = model.predict_proba(X)
                n_classes = len(np.unique(y))
                if n_classes == 2:
                    m["roc_auc"] = round(float(roc_auc_score(y, proba[:, 1])), 4)
                else:
                    m["roc_auc"] = round(float(roc_auc_score(y, proba, multi_class="ovr", average="weighted")), 4)
            except Exception:
                pass
        else:
            m["mae"] = round(float(mean_absolute_error(y, pred)), 4)
            m["mse"] = round(float(mean_squared_error(y, pred)), 4)
            m["rmse"] = round(float(np.sqrt(mean_squared_error(y, pred))), 4)
            m["r2"] = round(float(r2_score(y, pred)), 4)

        result[split_name] = m

    return result


def _map_primary_metric(metric: str, problem_type: str) -> str:
    mapping = {
        "f1_weighted": "f1_weighted",
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
    }
    return mapping.get(metric, "f1_weighted" if problem_type == "classification" else "neg_root_mean_squared_error")


def _format_results(models: list[TrainedModelResult], metric: str) -> str:
    lines = []
    for m in models:
        val_score = m.metrics.get("validation", {}).get(metric, "N/A")
        train_score = m.metrics.get("train", {}).get(metric, "N/A")
        cv_mean = m.metrics.get("cv_mean", {}).get(metric, "N/A")
        lines.append(
            f"{m.algorithm}: val_{metric}={val_score}, train_{metric}={train_score}, "
            f"cv_{metric}={cv_mean} [time: {m.training_time_seconds:.1f}s]"
        )
    return "\n".join(lines)
