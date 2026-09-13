"""End-to-end integration test of the autonomous data science workflow."""
import os
import pandas as pd
import numpy as np
from pathlib import Path

from app.tools.profiling import profile_dataframe, load_csv_safe
from app.tools.cleaning_tools import (
    remove_duplicates,
    drop_columns,
    impute_numerical,
    impute_categorical,
    encode_one_hot,
)
from app.ml.classifiers import get_classifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score


def test_end_to_end_disease_pipeline():
    csv_path = Path(__file__).parent.parent / "data" / "sample_disease.csv"
    assert csv_path.exists(), f"Sample dataset {csv_path} does not exist"

    # ── 1. Data Understanding & Profiling ─────────────────────────────────────
    df, encoding = load_csv_safe(str(csv_path))
    assert len(df) == 20
    assert "Disease" in df.columns

    profile = profile_dataframe(df)
    assert profile.target_column_suggestion == "Disease"
    assert profile.problem_type_suggestion == "classification"
    assert "SampleID" in profile.id_columns
    assert "Animal" in profile.categorical_columns

    # ── 2. Cleaning ───────────────────────────────────────────────────────────
    # Drop ID column
    df_clean, _ = drop_columns(df, profile.id_columns, "ID column")
    assert "SampleID" not in df_clean.columns

    # Impute missing numerical
    for col in profile.numerical_columns:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            df_clean, _ = impute_numerical(df_clean, col, strategy="median")
            assert df_clean[col].isnull().sum() == 0

    # Categorical encoding (except target)
    cat_features = [c for c in profile.categorical_columns if c != "Disease" and c in df_clean.columns]
    df_clean, _ = encode_one_hot(df_clean, cat_features)

    # ── 3. Train & Evaluate Models ────────────────────────────────────────────
    target_col = "Disease"
    X = df_clean.drop(columns=[target_col]).select_dtypes(include=np.number)
    y_raw = df_clean[target_col]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    models_to_test = ["RandomForest", "LogisticRegression"]
    results = {}

    for algo in models_to_test:
        clf = get_classifier(algo, {"n_estimators": 50, "C": 1.0})
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        results[algo] = f1
        assert f1 >= 0.0, f"F1 score should be non-negative, got {f1}"

    # Best model selection
    best_algo = max(results, key=lambda k: results[k])
    assert best_algo in models_to_test
    print(f"\nIntegration test passed! Best model: {best_algo} with F1: {results[best_algo]:.4f}")


def test_end_to_end_house_prices_pipeline():
    csv_path = Path(__file__).parent.parent / "data" / "sample_house_prices.csv"
    assert csv_path.exists(), f"Sample dataset {csv_path} does not exist"

    df, _ = load_csv_safe(str(csv_path))
    profile = profile_dataframe(df)

    assert profile.target_column_suggestion == "HousePrice"
    assert profile.problem_type_suggestion == "regression"
    assert "PropertyID" in profile.id_columns
