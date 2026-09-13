"""
Optuna hyperparameter optimization.
Defines search spaces for each algorithm and runs trials.
"""
from __future__ import annotations

import logging
import warnings
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from app.config.settings import settings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_hyperparameters(
    algorithm: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    primary_metric: str = "f1_weighted",
    n_trials: int = None,
    emit_fn: Callable = None,
) -> dict:
    """
    Run Optuna optimization for a given algorithm.
    Returns best hyperparameters dict.
    """
    n_trials = n_trials or settings.optuna_trials

    def emit(msg):
        if emit_fn:
            emit_fn(msg, agent="optimization")

    emit(f"Starting Optuna optimization for {algorithm} ({n_trials} trials)...")

    if problem_type == "classification":
        from app.ml.classifiers import get_classifier
        model_factory = get_classifier
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.random_seed)
        scoring = _map_metric(primary_metric, problem_type)
    else:
        from app.ml.classifiers import get_regressor
        model_factory = get_regressor
        cv = KFold(n_splits=3, shuffle=True, random_state=settings.random_seed)
        scoring = _map_metric(primary_metric, problem_type)

    space_fn = _get_search_space(algorithm)
    if space_fn is None:
        emit(f"No search space defined for {algorithm} — skipping optimization")
        return {}

    def objective(trial: optuna.Trial) -> float:
        params = space_fn(trial)
        try:
            model = model_factory(algorithm, params)
        except Exception:
            return float("-inf") if problem_type != "regression" else float("inf")

        try:
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring=scoring, n_jobs=1, error_score="raise"
            )
            return float(np.mean(scores))
        except Exception:
            return float("-inf") if problem_type != "regression" else float("inf")

    direction = "maximize" if _metric_direction(scoring) == "maximize" else "minimize"
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=settings.random_seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    try:
        study.optimize(objective, n_trials=n_trials, timeout=600, show_progress_bar=False)
        best_params = study.best_params
        best_value = study.best_value
        emit(f"✓ Optimization complete: {algorithm} best {scoring}={best_value:.4f} params={best_params}")
        return best_params
    except Exception as e:
        logger.warning(f"Optuna optimization failed for {algorithm}: {e}")
        return {}


def _get_search_space(algorithm: str):
    """Return a trial → params function for the given algorithm."""
    spaces = {
        "LogisticRegression": lambda t: {
            "C": t.suggest_float("C", 0.001, 10.0, log=True),
            "penalty": t.suggest_categorical("penalty", ["l1", "l2"]),
            "solver": "liblinear",
        },
        "RandomForest": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 50, 400),
            "max_depth": t.suggest_int("max_depth", 3, 20),
            "min_samples_split": t.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
        },
        "RandomForestRegressor": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 50, 400),
            "max_depth": t.suggest_int("max_depth", 3, 20),
            "min_samples_split": t.suggest_int("min_samples_split", 2, 20),
        },
        "XGBoost": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 50, 500),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": t.suggest_int("max_depth", 3, 10),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": t.suggest_int("min_child_weight", 1, 10),
        },
        "XGBoostRegressor": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 50, 500),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": t.suggest_int("max_depth", 3, 10),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0),
        },
        "LightGBM": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 50, 500),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": t.suggest_int("max_depth", 3, 15),
            "num_leaves": t.suggest_int("num_leaves", 20, 200),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0),
        },
        "LightGBMRegressor": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 50, 500),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": t.suggest_int("max_depth", 3, 15),
            "num_leaves": t.suggest_int("num_leaves", 20, 200),
        },
        "Ridge": lambda t: {"alpha": t.suggest_float("alpha", 0.001, 100.0, log=True)},
        "Lasso": lambda t: {"alpha": t.suggest_float("alpha", 0.001, 10.0, log=True)},
        "GradientBoosting": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 50, 300),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": t.suggest_int("max_depth", 2, 8),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
        },
        "GradientBoostingRegressor": lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 50, 300),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": t.suggest_int("max_depth", 2, 8),
        },
        "SVM": lambda t: {
            "C": t.suggest_float("C", 0.01, 100.0, log=True),
            "kernel": t.suggest_categorical("kernel", ["rbf", "linear"]),
        },
        "KNN": lambda t: {
            "n_neighbors": t.suggest_int("n_neighbors", 3, 30),
            "weights": t.suggest_categorical("weights", ["uniform", "distance"]),
        },
    }
    return spaces.get(algorithm)


def _map_metric(metric: str, problem_type: str) -> str:
    """Map our metric names to sklearn scoring strings."""
    mapping = {
        "f1_weighted": "f1_weighted",
        "f1": "f1_macro",
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
    }
    return mapping.get(metric, "f1_weighted" if problem_type == "classification" else "neg_root_mean_squared_error")


def _metric_direction(scoring: str) -> str:
    return "minimize" if scoring.startswith("neg_") else "maximize"
