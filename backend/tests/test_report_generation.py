from pathlib import Path

from PIL import Image as PILImage

from app.tools.report_tools import generate_pdf_report


def test_report_includes_all_available_charts_and_model_details(tmp_path):
    """A report remains a real PDF when charts and complete model metadata exist."""
    chart_one = tmp_path / "target_distribution.png"
    chart_two = tmp_path / "correlation_heatmap.png"
    for path, color in ((chart_one, "#2196F3"), (chart_two, "#90CAF9")):
        image = PILImage.new("RGB", (800, 420), color=color)
        image.save(path)

    output_path = tmp_path / "experiment_report.pdf"
    generated = generate_pdf_report(
        output_path=str(output_path),
        experiment_id="test-experiment",
        dataset_name="house_prices",
        problem_type="regression",
        target_col="price",
        primary_metric="r2",
        df_shape=(120, 12),
        cleaning_summary="Missing numeric values were imputed and duplicate rows were removed.",
        eda_summary="The target is right-skewed and several numeric columns are correlated.",
        chart_paths=[str(chart_one), str(chart_two)],
        model_comparison_table=[
            {
                "algorithm": "RandomForestRegressor",
                "val_r2": 0.88,
                "cv_mean": 0.85,
                "cv_std": 0.03,
                "is_overfitting": False,
                "training_time_s": 1.2,
            }
        ],
        best_model_name="RandomForestRegressor",
        best_model_metrics={"r2": 0.88},
        shap_plot_paths=[],
        feature_importance=[
            {"feature": "size", "shap_mean_abs": 0.41, "rank": 1},
            {"feature": "bedrooms", "shap_mean_abs": 0.22, "rank": 2},
        ],
        overfitting_analysis=None,
        leakage_warnings=[],
        recommendations="Monitor prediction drift and retrain after material performance degradation.",
        model_details=[
            {
                "algorithm": "RandomForestRegressor",
                "artifact_path": str(tmp_path / "RandomForestRegressor.pkl"),
                "hyperparameters": {"n_estimators": 300, "max_depth": 12},
                "metrics": {
                    "train": {"r2": 0.96, "rmse": 1012.2},
                    "validation": {"r2": 0.88, "rmse": 1489.7},
                    "test": {"r2": 0.86, "rmse": 1510.1},
                },
                "training_time_seconds": 1.2,
            }
        ],
    )

    pdf_bytes = Path(generated).read_bytes()
    assert Path(generated) == output_path
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 5_000
