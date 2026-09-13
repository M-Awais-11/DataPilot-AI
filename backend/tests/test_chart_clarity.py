from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image as PILImage

from app.tools.eda_tools import (
    plot_target_distribution,
    plot_missing_values,
    plot_correlation_heatmap,
    plot_numerical_distributions,
    plot_boxplots,
    plot_categorical_distributions,
    plot_feature_vs_target,
)
from app.tools.report_tools import generate_pdf_report


def test_eda_tools_generate_high_clarity_charts(tmp_path):
    """Verify that EDA charts are created with high resolution and crisp dimensions."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "target": np.random.choice(["Churn", "Retain"], size=n, p=[0.3, 0.7]),
        "numeric_1": np.random.normal(50, 15, size=n),
        "numeric_2": np.random.exponential(10, size=n),
        "category_1": np.random.choice(["Tier A", "Tier B", "Tier C"], size=n),
    })
    # Inject a few missing values
    df.loc[0:5, "numeric_1"] = np.nan

    out_dir = str(tmp_path)
    
    # 1. Target distribution
    tgt_path, tgt_insight = plot_target_distribution(df, "target", out_dir)
    assert Path(tgt_path).is_file()
    with PILImage.open(tgt_path) as img:
        # 250 DPI with 10 inches should be around 2500 pixels wide
        assert img.width >= 1500
        assert img.height >= 800

    # 2. Missing values
    mv_path, mv_insight = plot_missing_values(df, out_dir)
    assert Path(mv_path).is_file()

    # 3. Correlation heatmap
    corr_path, corr_insight = plot_correlation_heatmap(df, out_dir)
    assert Path(corr_path).is_file()
    with PILImage.open(corr_path) as img:
        assert img.width >= 1500

    # 4. Numerical distributions
    num_res = plot_numerical_distributions(df, ["numeric_1", "numeric_2"], out_dir)
    assert len(num_res) > 0
    assert Path(num_res[0][0]).is_file()

    # 5. Boxplots
    box_res = plot_boxplots(df, ["numeric_1", "numeric_2"], "target", out_dir)
    assert len(box_res) > 0
    assert Path(box_res[0][0]).is_file()

    # 6. Categorical
    cat_res = plot_categorical_distributions(df, ["category_1"], out_dir)
    assert len(cat_res) > 0
    assert Path(cat_res[0][0]).is_file()

    # 7. Feature vs Target
    fvt_res = plot_feature_vs_target(df, ["numeric_1", "numeric_2"], "target", "classification", out_dir)
    assert len(fvt_res) > 0
    assert Path(fvt_res[0][0]).is_file()

    # 8. Report PDF Generation with all high-res charts
    all_charts = [tgt_path, mv_path, corr_path, num_res[0][0], box_res[0][0], cat_res[0][0], fvt_res[0][0]]
    pdf_out = str(tmp_path / "clarity_report.pdf")
    generated = generate_pdf_report(
        output_path=pdf_out,
        experiment_id="clarity-exp",
        dataset_name="clarity_dataset",
        problem_type="classification",
        target_col="target",
        primary_metric="f1",
        df_shape=(n, 4),
        cleaning_summary="Imputed numeric_1, clean target.",
        eda_summary="Normal distributions and clear class balance.",
        chart_paths=all_charts,
        model_comparison_table=[{"algorithm": "XGBoost", "val_f1": 0.89, "cv_mean": 0.87, "cv_std": 0.02, "is_overfitting": False}],
        best_model_name="XGBoost",
        best_model_metrics={"f1": 0.89},
        shap_plot_paths=[],
        feature_importance=[{"feature": "numeric_1", "shap_mean_abs": 0.45, "rank": 1}],
        overfitting_analysis=None,
        leakage_warnings=[],
        recommendations="Deploy model to serving sandbox.",
    )

    assert Path(generated).is_file()
    assert Path(generated).stat().st_size > 10_000
