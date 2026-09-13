"""
SHAP Explainability Tools — deterministic SHAP computation.
No LLM involvement.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def compute_shap(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: str,
    problem_type: str = "classification",
    max_samples: int = 500,
) -> dict:
    """
    Compute SHAP values and generate summary/bar plots.
    Returns a dict with plot paths and feature importance list.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Sample for speed if large dataset
    if len(X_train) > max_samples:
        X_background = shap.sample(X_train, max_samples, random_state=42)
    else:
        X_background = X_train.copy()

    X_explain = X_test.copy().head(min(200, len(X_test)))

    # Choose explainer based on model type
    explainer = _get_explainer(model, X_background, problem_type)
    if explainer is None:
        logger.warning("Could not create SHAP explainer for this model type")
        return {}

    try:
        shap_values = explainer(X_explain)
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return {}

    # Handle multi-class case — use class 1 or mean
    sv = _extract_shap_values(shap_values, problem_type)

    # ── Feature importance from SHAP ──────────────────────────────────────────
    mean_abs_shap = np.abs(sv).mean(axis=0)
    feature_names = X_explain.columns.tolist()
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "shap_mean_abs": mean_abs_shap,
    }).sort_values("shap_mean_abs", ascending=False).reset_index(drop=True)
    importance_df["rank"] = range(1, len(importance_df) + 1)

    # ── SHAP Summary Plot ─────────────────────────────────────────────────────
    summary_path = str(Path(output_dir) / "shap_summary.png")
    fig, ax = plt.subplots(figsize=(11, 7))
    shap.summary_plot(sv, X_explain, show=False, max_display=15)
    plt.title("SHAP Feature Attribution Summary", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout(pad=1.5)
    plt.savefig(summary_path, dpi=250, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close("all")

    # ── SHAP Bar Plot ─────────────────────────────────────────────────────────
    bar_path = str(Path(output_dir) / "shap_bar.png")
    fig, ax = plt.subplots(figsize=(11, 7))
    top_n = min(15, len(feature_names))
    top_features = importance_df.head(top_n)
    
    vals = top_features["shap_mean_abs"][::-1].values
    feat_labels = top_features["feature"][::-1].values
    
    bars = ax.barh(feat_labels, vals, color="#2563EB", edgecolor="#1E3A8A", linewidth=1.2, alpha=0.9)
    ax.set_xlabel("Mean Absolute SHAP Value (Feature Impact)", fontsize=12, labelpad=10)
    ax.set_title("Global Feature Importance (SHAP)", fontsize=16, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    
    # Direct numeric annotations on bars
    max_x = max(vals) if len(vals) else 1
    ax.set_xlim(0, max_x * 1.18)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_width() + (max_x * 0.015),
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#1E293B",
        )
        
    plt.tight_layout(pad=1.5)
    plt.savefig(bar_path, dpi=250, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close("all")

    return {
        "summary_plot_path": summary_path,
        "bar_plot_path": bar_path,
        "feature_importance": importance_df.to_dict(orient="records"),
        "local_explanation_available": True,
        "n_samples_explained": len(X_explain),
    }


def compute_local_explanation(
    model,
    X_background: pd.DataFrame,
    input_data: pd.DataFrame,
    problem_type: str = "classification",
) -> dict:
    """Compute SHAP explanation for a single prediction."""
    explainer = _get_explainer(model, X_background, problem_type)
    if explainer is None:
        return {}

    try:
        shap_values = explainer(input_data)
        sv = _extract_shap_values(shap_values, problem_type)

        feature_contributions = {
            feat: round(float(val), 4)
            for feat, val in zip(input_data.columns, sv[0])
        }
        # Sort by absolute contribution
        sorted_contributions = dict(
            sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        return {"feature_contributions": sorted_contributions}
    except Exception as e:
        logger.warning(f"Local SHAP explanation failed: {e}")
        return {}


def _get_explainer(model, X_background: pd.DataFrame, problem_type: str):
    """Select appropriate SHAP explainer."""
    try:
        # Tree-based models
        model_class = type(model).__name__
        if model_class in ("XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor",
                           "RandomForestClassifier", "RandomForestRegressor",
                           "GradientBoostingClassifier", "GradientBoostingRegressor",
                           "DecisionTreeClassifier"):
            return shap.TreeExplainer(model)

        # Linear models
        if model_class in ("LogisticRegression", "LinearRegression", "Ridge", "Lasso"):
            return shap.LinearExplainer(model, X_background, feature_perturbation="interventional")

        # Generic (slower)
        return shap.KernelExplainer(
            model.predict_proba if problem_type == "classification" and hasattr(model, "predict_proba") else model.predict,
            shap.sample(X_background, min(100, len(X_background))),
        )
    except Exception as e:
        logger.warning(f"Could not create SHAP explainer: {e}")
        return None


def _extract_shap_values(shap_values, problem_type: str) -> np.ndarray:
    """Extract 2D SHAP value array from potentially multi-class output."""
    sv = shap_values.values if hasattr(shap_values, "values") else shap_values

    if isinstance(sv, list):
        # Multi-class: use class 1 for binary, or mean abs for multiclass
        if len(sv) == 2:
            return sv[1]
        return np.mean(np.abs(sv), axis=0)

    if sv.ndim == 3:
        # [samples, features, classes]
        if sv.shape[2] == 2:
            return sv[:, :, 1]
        return np.mean(np.abs(sv), axis=2)

    return sv
