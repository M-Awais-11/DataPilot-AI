"""
Deterministic data cleaning tools.
All transformation logic lives here — no LLM.
The Cleaning Agent calls these functions and passes results to the LLM for logging.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class CleaningResult:
    operation: str
    column: Optional[str]
    details: str
    rows_affected: int
    parameters: dict


# ── Duplicate Removal ─────────────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningResult]:
    original = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    removed = original - len(df_clean)
    return df_clean, CleaningResult(
        operation="remove_duplicates",
        column=None,
        details=f"Removed {removed} duplicate rows",
        rows_affected=removed,
        parameters={},
    )


# ── Drop Constant & ID Columns ────────────────────────────────────────────────
def drop_columns(df: pd.DataFrame, columns: list[str], reason: str) -> tuple[pd.DataFrame, list[CleaningResult]]:
    results = []
    cols_to_drop = [c for c in columns if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    for col in cols_to_drop:
        results.append(CleaningResult(
            operation="drop_column",
            column=col,
            details=reason,
            rows_affected=0,
            parameters={"reason": reason},
        ))
    return df, results


# ── Missing Value Imputation ──────────────────────────────────────────────────
def impute_numerical(
    df: pd.DataFrame,
    column: str,
    strategy: str = "median",
    knn_neighbors: int = 5,
) -> tuple[pd.DataFrame, CleaningResult]:
    """strategy: mean | median | knn"""
    null_count = int(df[column].isnull().sum())
    if null_count == 0:
        return df, CleaningResult("impute_skip", column, "No missing values", 0, {})

    if strategy == "knn":
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        df[num_cols] = imputer.fit_transform(df[num_cols])
        details = f"KNN imputation (k={knn_neighbors}) on {null_count} missing values"
    else:
        fill_value = df[column].mean() if strategy == "mean" else df[column].median()
        df[column] = df[column].fillna(fill_value)
        details = f"{strategy.capitalize()} imputation ({fill_value:.4g}) on {null_count} missing values"

    return df, CleaningResult(
        operation=f"impute_{strategy}",
        column=column,
        details=details,
        rows_affected=null_count,
        parameters={"strategy": strategy},
    )


def impute_categorical(
    df: pd.DataFrame,
    column: str,
    strategy: str = "mode",
) -> tuple[pd.DataFrame, CleaningResult]:
    """strategy: mode | unknown"""
    null_count = int(df[column].isnull().sum())
    if null_count == 0:
        return df, CleaningResult("impute_skip", column, "No missing values", 0, {})

    if strategy == "mode":
        fill_value = df[column].mode().iloc[0] if len(df[column].dropna()) > 0 else "Unknown"
    else:
        fill_value = "Unknown"

    df[column] = df[column].fillna(fill_value)
    return df, CleaningResult(
        operation=f"impute_{strategy}",
        column=column,
        details=f"Filled {null_count} missing values with '{fill_value}'",
        rows_affected=null_count,
        parameters={"strategy": strategy, "fill_value": str(fill_value)},
    )


def choose_imputation_strategy(
    df: pd.DataFrame,
    column: str,
    semantic_type: str,
    missing_pct: float,
) -> str:
    """
    Rule-based imputation strategy selector.
    Returns strategy string for use with impute_numerical / impute_categorical.
    """
    if semantic_type == "numerical":
        if missing_pct > 20:
            return "knn"
        # Check skewness
        skewness = df[column].skew()
        if abs(skewness) > 1.5:
            return "median"
        return "mean"
    elif semantic_type == "categorical":
        cardinality = df[column].nunique()
        if cardinality > 20:
            return "unknown"
        return "mode"
    return "mode"


# ── Outlier Detection & Handling ──────────────────────────────────────────────
def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.Series:
    """Return a boolean mask: True = outlier."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (df[column] < lower) | (df[column] > upper)


def handle_outliers_clip(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5,
) -> tuple[pd.DataFrame, CleaningResult]:
    """Cap outliers at IQR bounds (winsorization) rather than removing them."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    mask = detect_outliers_iqr(df, column, multiplier)
    n_outliers = int(mask.sum())
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df, CleaningResult(
        operation="clip_outliers_iqr",
        column=column,
        details=f"Clipped {n_outliers} outliers to [{lower:.4g}, {upper:.4g}]",
        rows_affected=n_outliers,
        parameters={"lower": lower, "upper": upper, "multiplier": multiplier},
    )


# ── Categorical Encoding ──────────────────────────────────────────────────────
def encode_one_hot(
    df: pd.DataFrame,
    columns: list[str],
    max_categories: int = 20,
) -> tuple[pd.DataFrame, list[CleaningResult]]:
    """Apply one-hot encoding. Drops original columns."""
    results = []
    cols = [c for c in columns if c in df.columns and df[c].nunique() <= max_categories]
    if cols:
        df = pd.get_dummies(df, columns=cols, drop_first=False, dtype=int)
        for col in cols:
            results.append(CleaningResult(
                operation="one_hot_encode",
                column=col,
                details=f"One-hot encoded column",
                rows_affected=len(df),
                parameters={"max_categories": max_categories},
            ))
    return df, results


def encode_label(
    df: pd.DataFrame,
    column: str,
) -> tuple[pd.DataFrame, CleaningResult, LabelEncoder]:
    """Label-encode a single column. Returns encoder for inverse transform."""
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    return df, CleaningResult(
        operation="label_encode",
        column=column,
        details=f"Label encoded — {len(le.classes_)} unique classes",
        rows_affected=len(df),
        parameters={"classes": list(le.classes_)},
    ), le


def encode_ordinal(
    df: pd.DataFrame,
    column: str,
    categories: list[str],
) -> tuple[pd.DataFrame, CleaningResult]:
    enc = OrdinalEncoder(categories=[categories], handle_unknown="use_encoded_value", unknown_value=-1)
    df[column] = enc.fit_transform(df[[column]])
    return df, CleaningResult(
        operation="ordinal_encode",
        column=column,
        details=f"Ordinal encoded with order: {categories}",
        rows_affected=len(df),
        parameters={"categories": categories},
    )


# ── Numerical Scaling ─────────────────────────────────────────────────────────
def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: list[str],
    method: str = "standard",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object, CleaningResult]:
    """
    Fit scaler on training data only, transform all splits.
    method: standard | minmax | robust
    """
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    scaler = scalers.get(method, StandardScaler())
    cols = [c for c in columns if c in X_train.columns]

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_val[cols] = scaler.transform(X_val[cols])
    X_test[cols] = scaler.transform(X_test[cols])

    result = CleaningResult(
        operation=f"scale_{method}",
        column=None,
        details=f"{method.capitalize()}Scaler applied to {len(cols)} columns",
        rows_affected=len(X_train),
        parameters={"method": method, "columns": cols},
    )
    return X_train, X_val, X_test, scaler, result


def choose_scaler(df: pd.DataFrame, columns: list[str]) -> str:
    """
    Choose scaling method based on data distribution.
    Returns: standard | robust | minmax
    """
    outlier_fraction = 0.0
    for col in columns:
        if col not in df.columns:
            continue
        mask = detect_outliers_iqr(df, col)
        outlier_fraction += mask.mean()

    avg_outlier_fraction = outlier_fraction / max(len(columns), 1)
    if avg_outlier_fraction > 0.05:
        return "robust"
    return "standard"
