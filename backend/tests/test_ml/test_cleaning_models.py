"""Unit tests for deterministic cleaning and model tools."""
import pandas as pd
import numpy as np
from app.tools.cleaning_tools import (
    remove_duplicates,
    drop_columns,
    impute_numerical,
    impute_categorical,
    handle_outliers_clip,
    encode_one_hot,
)
from app.ml.classifiers import get_classifier, get_regressor


def test_cleaning_pipeline():
    df = pd.DataFrame({
        "A": [1, 2, 2, 4, np.nan],
        "B": ["cat", "dog", "dog", "bird", None],
        "Const": [1, 1, 1, 1, 1],
    })

    # Duplicates
    df_dedup, res_dedup = remove_duplicates(df)
    assert res_dedup.rows_affected == 1

    # Drop constant
    df_dropped, res_drop = drop_columns(df_dedup, ["Const"], "constant")
    assert "Const" not in df_dropped.columns

    # Impute numerical
    df_imp_num, res_num = impute_numerical(df_dropped, "A", strategy="mean")
    assert df_imp_num["A"].isnull().sum() == 0

    # Impute categorical
    df_imp_cat, res_cat = impute_categorical(df_imp_num, "B", strategy="mode")
    assert df_imp_cat["B"].isnull().sum() == 0


def test_model_factories():
    clf = get_classifier("RandomForest", {"n_estimators": 10})
    assert clf is not None
    assert clf.n_estimators == 10

    reg = get_regressor("Ridge", {"alpha": 2.0})
    assert reg is not None
    assert reg.alpha == 2.0
