"""
ML model factories — instantiate classifiers and regressors by name.
All models use deterministic random seeds for reproducibility.
"""
from __future__ import annotations

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from app.config.settings import settings

SEED = settings.random_seed


def get_classifier(name: str, hyperparams: dict | None = None):
    """Return a scikit-learn compatible classifier instance."""
    hp = hyperparams or {}
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=SEED, **_filter(hp, ["C", "penalty", "solver"])
        ),
        "DecisionTree": DecisionTreeClassifier(
            random_state=SEED, **_filter(hp, ["max_depth", "min_samples_split", "min_samples_leaf"])
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=hp.get("n_estimators", 200),
            max_depth=hp.get("max_depth", None),
            random_state=SEED,
            n_jobs=-1,
            **_filter(hp, ["min_samples_split", "min_samples_leaf", "max_features"]),
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=SEED,
            **_filter(hp, ["n_estimators", "learning_rate", "max_depth", "subsample"]),
        ),
    }
    if xgb is not None:
        models["XGBoost"] = xgb.XGBClassifier(
            random_state=SEED,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
            verbosity=0,
            **_filter(hp, ["n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree", "min_child_weight"]),
        )
    if lgb is not None:
        models["LightGBM"] = lgb.LGBMClassifier(
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
            **_filter(hp, ["n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree", "num_leaves"]),
        )
    models.update({
        "SVM": SVC(
            probability=True,
            random_state=SEED,
            **_filter(hp, ["C", "kernel", "gamma"]),
        ),
        "KNN": KNeighborsClassifier(
            **_filter(hp, ["n_neighbors", "weights", "metric"]),
        ),
    })
    if name not in models:
        raise ValueError(f"Unknown classifier: {name}. Available: {list(models.keys())}")
    return models[name]


def get_regressor(name: str, hyperparams: dict | None = None):
    """Return a scikit-learn compatible regressor instance."""
    hp = hyperparams or {}
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(
            **_filter(hp, ["alpha"])
        ),
        "Lasso": Lasso(
            max_iter=2000,
            **_filter(hp, ["alpha"]),
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=hp.get("n_estimators", 200),
            random_state=SEED,
            n_jobs=-1,
            **_filter(hp, ["max_depth", "min_samples_split", "min_samples_leaf"]),
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            random_state=SEED,
            **_filter(hp, ["n_estimators", "learning_rate", "max_depth", "subsample"]),
        ),
    }
    if xgb is not None:
        models["XGBoostRegressor"] = xgb.XGBRegressor(
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
            **_filter(hp, ["n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree"]),
        )
    if lgb is not None:
        models["LightGBMRegressor"] = lgb.LGBMRegressor(
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
            **_filter(hp, ["n_estimators", "learning_rate", "max_depth", "num_leaves"]),
        )
    models.update({
        "SVR": SVR(
            **_filter(hp, ["C", "kernel", "gamma", "epsilon"]),
        ),
    })
    if name not in models:
        raise ValueError(f"Unknown regressor: {name}. Available: {list(models.keys())}")
    return models[name]



def _filter(hp: dict, allowed: list[str]) -> dict:
    """Return only keys present in both hp and allowed."""
    return {k: v for k, v in hp.items() if k in allowed}
