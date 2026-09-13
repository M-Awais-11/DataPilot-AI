"""Unit tests for deterministic profiling tools."""
import pandas as pd
import numpy as np
from app.tools.profiling import profile_dataframe, load_csv_safe


def test_profiling_classification():
    # Synthetic classification dataset (like Titanic or Disease)
    data = {
        "Age": [22, 38, 26, 35, np.nan, 54, 2, 27, 14, 4],
        "Fare": [7.25, 71.28, 7.92, 53.1, 8.05, 51.86, 21.07, 11.13, 30.07, 16.7],
        "Sex": ["male", "female", "female", "female", "male", "male", "male", "male", "female", "female"],
        "Survived": [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
        "PassengerId": list(range(1, 11)),
    }
    df = pd.DataFrame(data)

    profile = profile_dataframe(df)

    assert profile.row_count == 10
    assert profile.col_count == 5
    assert profile.target_column_suggestion == "Survived"
    assert profile.problem_type_suggestion == "classification"
    assert "PassengerId" in profile.id_columns
    assert "Age" in profile.numerical_columns
    assert "Sex" in profile.categorical_columns


def test_profiling_regression():
    # Synthetic regression dataset (House price)
    data = {
        "SquareFeet": [1500, 2000, 1200, 2500, 1800],
        "Bedrooms": [3, 4, 2, 5, 3],
        "House_Price": [300000, 420000, 240000, 510000, 360000],
    }
    df = pd.DataFrame(data)

    profile = profile_dataframe(df)

    assert profile.target_column_suggestion == "House_Price"
    assert profile.problem_type_suggestion == "regression"
