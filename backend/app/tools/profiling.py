"""
Deterministic DataFrame profiling tool.
No LLM involvement — only pure Pandas/NumPy computation.
The agent will receive this structured result and use the LLM to reason about it.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ── Data classes for structured output ────────────────────────────────────────
@dataclass
class ColumnProfile:
    name: str
    dtype: str
    semantic_type: str          # numerical | categorical | boolean | datetime | text | id | constant
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    sample_values: list[Any]
    # Numerical only
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    # Categorical only
    top_values: Optional[dict] = None      # value -> count
    # Flags
    is_constant: bool = False
    is_near_zero_variance: bool = False
    is_likely_id: bool = False
    is_possible_target: bool = False
    leakage_risk: str = "none"           # none | low | medium | high


@dataclass
class DatasetProfile:
    row_count: int
    col_count: int
    duplicate_rows: int
    duplicate_pct: float
    total_missing: int
    total_missing_pct: float
    memory_mb: float
    columns: list[ColumnProfile]
    numerical_columns: list[str]
    categorical_columns: list[str]
    boolean_columns: list[str]
    datetime_columns: list[str]
    text_columns: list[str]
    id_columns: list[str]
    constant_columns: list[str]
    near_zero_variance_columns: list[str]
    correlation_matrix: Optional[dict] = None   # top correlations only
    target_column_suggestion: Optional[str] = None
    target_confidence: float = 0.0
    problem_type_suggestion: Optional[str] = None
    class_distribution: Optional[dict] = None
    class_imbalance: bool = False
    data_quality_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ── Helpers ───────────────────────────────────────────────────────────────────
TARGET_KEYWORDS = {
    "target", "label", "class", "output", "result", "outcome", "prediction",
    "y", "response", "dependent", "disease", "diagnosis", "fraud", "churn",
    "price", "cost", "sales", "revenue", "demand", "default", "survived",
    "status", "grade", "score", "rating", "category",
}

ID_KEYWORDS = {"id", "_id", "uuid", "guid", "key", "index", "idx", "code", "no", "num", "number"}


def _detect_semantic_type(series: pd.Series, col_name: str) -> str:
    """Classify a column into a semantic type."""
    col_lower = col_name.lower()

    if series.nunique() == 1:
        return "constant"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Try parsing as datetime
    if series.dtype == object:
        sample = series.dropna().head(10)
        try:
            pd.to_datetime(sample, infer_datetime_format=True)
            if series.str.match(r"\d{4}[-/]\d{2}[-/]\d{2}").mean() > 0.5:
                return "datetime"
        except Exception:
            pass

    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        n_total = len(series.dropna())
        unique_ratio = n_unique / max(n_total, 1)

        # Likely an ID column
        if unique_ratio > 0.85:
            for kw in ID_KEYWORDS:
                if kw in col_lower:
                    return "id"
            if n_unique == n_total and n_total > 50:
                return "id"

        # Binary numerical → boolean
        if set(series.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            return "boolean"

        return "numerical"

    # Object dtype
    n_unique = series.nunique()
    n_total = len(series.dropna())
    unique_ratio = n_unique / max(n_total, 1)

    # Likely an ID column
    if unique_ratio > 0.85:
        for kw in ID_KEYWORDS:
            if kw in col_lower:
                return "id"


    # Long text
    avg_len = series.dropna().astype(str).str.len().mean()
    if avg_len > 50 and n_unique > 50:
        return "text"

    # Boolean strings
    unique_vals = set(series.dropna().str.lower().unique())
    if unique_vals.issubset({"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}):
        return "boolean"

    return "categorical"


def _near_zero_variance(series: pd.Series, threshold: float = 0.01) -> bool:
    """Return True if the column has near-zero variance."""
    if not pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if len(clean) == 0:
            return True
        top_freq = float(clean.value_counts(normalize=True).iloc[0])
        return bool(top_freq > (1 - threshold))

    clean = series.dropna()
    if len(clean) <= 1:
        return True
    std_val = float(clean.std())
    mean_val = float(clean.mean())
    if std_val == 0.0:
        return True
    return bool((std_val / (abs(mean_val) + 1e-9)) < threshold)



def _top_correlations(df: pd.DataFrame, top_n: int = 20) -> dict:
    """Return top N absolute correlations as a dict of {col_pair: value}."""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 2:
        return {}
    corr = df[num_cols].corr().abs()
    pairs = {}
    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i + 1:]:
            val = corr.loc[c1, c2]
            if not np.isnan(val):
                pairs[f"{c1}|{c2}"] = round(float(val), 4)
    sorted_pairs = dict(sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:top_n])
    return sorted_pairs


def _suggest_target(df: pd.DataFrame, columns: list[ColumnProfile]) -> tuple[Optional[str], float]:
    """
    Heuristically suggest a target column and a confidence score [0, 1].
    Priority: keyword match + last column + low unique count.
    """
    scores: dict[str, float] = {}
    n_rows = len(df)

    for col in columns:
        if col.semantic_type in ("id", "constant", "text"):
            continue
        score = 0.0
        col_lower = col.name.lower()

        # Keyword match (strong signal)
        for kw in TARGET_KEYWORDS:
            if kw == col_lower or col_lower.endswith(f"_{kw}") or col_lower.startswith(f"{kw}_"):
                score += 0.5
                break
            elif kw in col_lower:
                score += 0.3
                break

        # Last column is often the target
        if col.name == df.columns[-1]:
            score += 0.2

        # Low unique count relative to rows (categorical target)
        if col.semantic_type == "categorical":
            u_ratio = col.unique_count / max(n_rows, 1)
            if u_ratio < 0.05:
                score += 0.2
            elif u_ratio < 0.2:
                score += 0.1

        # Binary numerical
        if col.semantic_type in ("boolean", "numerical") and col.unique_count == 2:
            score += 0.15

        # Penalize if looks like feature
        if col.semantic_type == "numerical" and col.unique_count > 50:
            score -= 0.1

        if score > 0:
            scores[col.name] = min(score, 1.0)

    if not scores:
        return None, 0.0

    best = max(scores, key=lambda k: scores[k])
    return best, scores[best]


def _detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """Detect classification vs regression based on target column properties."""
    series = df[target_col].dropna()
    n_unique = series.nunique()

    if pd.api.types.is_numeric_dtype(series):
        # If wide range, large mean, or high unique count -> regression
        val_range = float(series.max() - series.min()) if len(series) > 0 else 0
        mean_val = float(series.mean()) if len(series) > 0 else 0
        if val_range > 100 or mean_val > 100 or n_unique > 20:
            return "regression"
        return "classification"

    return "classification"



def _leakage_risk(col_name: str, target_col: Optional[str], df: pd.DataFrame) -> str:
    """Heuristic leakage risk for a column."""
    if target_col is None:
        return "none"
    col_lower = col_name.lower()
    leakage_keywords = {"final", "result", "outcome", "post", "after", "end", "last"}
    for kw in leakage_keywords:
        if kw in col_lower:
            return "medium"

    # High correlation with target (numerical target)
    if pd.api.types.is_numeric_dtype(df[target_col]) and pd.api.types.is_numeric_dtype(df[col_name]):
        try:
            corr = abs(df[col_name].corr(df[target_col]))
            if corr > 0.98:
                return "high"
            elif corr > 0.9:
                return "medium"
        except Exception:
            pass
    return "none"


# ── Main profiling function ────────────────────────────────────────────────────
def profile_dataframe(df: pd.DataFrame, target_col: Optional[str] = None) -> DatasetProfile:
    """
    Profile a DataFrame and return a structured DatasetProfile.
    This is a pure Python/Pandas function — no LLM involved.
    """
    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    total_missing = int(df.isnull().sum().sum())
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3)

    columns: list[ColumnProfile] = []
    numerical_cols, categorical_cols, boolean_cols, datetime_cols = [], [], [], []
    text_cols, id_cols, constant_cols, nzv_cols = [], [], [], []

    for col_name in df.columns:
        series = df[col_name]
        sem_type = _detect_semantic_type(series, col_name)
        null_count = int(series.isnull().sum())
        unique_count = int(series.nunique())
        sample = series.dropna().head(5).tolist()
        # Convert numpy types for JSON serialization
        sample = [x.item() if hasattr(x, "item") else x for x in sample]

        prof = ColumnProfile(
            name=col_name,
            dtype=str(series.dtype),
            semantic_type=sem_type,
            null_count=null_count,
            null_pct=round(null_count / max(n_rows, 1) * 100, 2),
            unique_count=unique_count,
            unique_pct=round(unique_count / max(n_rows, 1) * 100, 2),
            sample_values=sample,
            is_constant=(sem_type == "constant"),
            is_near_zero_variance=_near_zero_variance(series),
            is_likely_id=(sem_type == "id"),
            leakage_risk=_leakage_risk(col_name, target_col, df),
        )

        if sem_type == "numerical":
            clean = series.dropna()
            prof.mean = round(float(clean.mean()), 4) if len(clean) else None
            prof.median = round(float(clean.median()), 4) if len(clean) else None
            prof.std = round(float(clean.std()), 4) if len(clean) else None
            prof.min_val = round(float(clean.min()), 4) if len(clean) else None
            prof.max_val = round(float(clean.max()), 4) if len(clean) else None
            prof.q25 = round(float(clean.quantile(0.25)), 4) if len(clean) else None
            prof.q75 = round(float(clean.quantile(0.75)), 4) if len(clean) else None
            try:
                prof.skewness = round(float(stats.skew(clean)), 4)
                prof.kurtosis = round(float(stats.kurtosis(clean)), 4)
            except Exception:
                pass
            numerical_cols.append(col_name)

        elif sem_type in ("categorical", "text"):
            vc = series.value_counts().head(10).to_dict()
            prof.top_values = {str(k): int(v) for k, v in vc.items()}
            if sem_type == "categorical":
                categorical_cols.append(col_name)
            else:
                text_cols.append(col_name)

        elif sem_type == "boolean":
            boolean_cols.append(col_name)
        elif sem_type == "datetime":
            datetime_cols.append(col_name)
        elif sem_type == "id":
            id_cols.append(col_name)
        elif sem_type == "constant":
            constant_cols.append(col_name)

        if prof.is_near_zero_variance and sem_type not in ("constant", "id"):
            nzv_cols.append(col_name)

        columns.append(prof)

    # ── Target suggestion ─────────────────────────────────────────────────────
    suggested_target, confidence = _suggest_target(df, columns)
    if target_col:
        suggested_target = target_col
        confidence = 1.0

    # ── Problem type ──────────────────────────────────────────────────────────
    problem_type = None
    class_distribution = None
    class_imbalance = False
    if suggested_target and suggested_target in df.columns:
        problem_type = _detect_problem_type(df, suggested_target)
        if problem_type == "classification":
            vc = df[suggested_target].value_counts()
            class_distribution = {str(k): int(v) for k, v in vc.items()}
            max_cls = vc.max()
            min_cls = vc.min()
            class_imbalance = (max_cls / max(min_cls, 1)) > 3.0

        # Mark target as possible target
        for cp in columns:
            if cp.name == suggested_target:
                cp.is_possible_target = True

    # ── Correlations ──────────────────────────────────────────────────────────
    correlations = _top_correlations(df)

    # ── Data quality issues ───────────────────────────────────────────────────
    issues = []
    missing_pct = total_missing / max(n_rows * n_cols, 1) * 100
    if missing_pct > 5:
        issues.append(f"High missing value rate: {missing_pct:.1f}% of all cells")
    if duplicate_rows > 0:
        issues.append(f"{duplicate_rows} duplicate rows detected ({duplicate_rows/n_rows*100:.1f}%)")
    if constant_cols:
        issues.append(f"Constant columns detected: {constant_cols}")
    if id_cols:
        issues.append(f"Likely ID columns (should be dropped): {id_cols}")
    if class_imbalance:
        issues.append("Class imbalance detected in target column")
    leakage_cols = [cp.name for cp in columns if cp.leakage_risk in ("medium", "high") and cp.name != suggested_target]
    if leakage_cols:
        issues.append(f"Potential data leakage columns: {leakage_cols}")

    return DatasetProfile(
        row_count=n_rows,
        col_count=n_cols,
        duplicate_rows=duplicate_rows,
        duplicate_pct=round(duplicate_rows / max(n_rows, 1) * 100, 2),
        total_missing=total_missing,
        total_missing_pct=round(missing_pct, 2),
        memory_mb=memory_mb,
        columns=columns,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        boolean_columns=boolean_cols,
        datetime_columns=datetime_cols,
        text_columns=text_cols,
        id_columns=id_cols,
        constant_columns=constant_cols,
        near_zero_variance_columns=nzv_cols,
        correlation_matrix=correlations,
        target_column_suggestion=suggested_target,
        target_confidence=round(confidence, 3),
        problem_type_suggestion=problem_type,
        class_distribution=class_distribution,
        class_imbalance=class_imbalance,
        data_quality_issues=issues,
    )


def load_csv_safe(file_path: str, max_rows: Optional[int] = None) -> tuple[pd.DataFrame, str]:
    """
    Safely load a CSV file, detecting encoding automatically.
    Returns (dataframe, encoding_used).
    """
    encoding = "utf-8"
    try:
        import chardet
        with open(file_path, "rb") as f:
            raw = f.read(100_000)
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
    except Exception:
        encoding = "utf-8"

    try:
        df = pd.read_csv(file_path, encoding=encoding, nrows=max_rows, low_memory=False)
    except Exception:
        df = pd.read_csv(file_path, encoding="latin-1", nrows=max_rows, low_memory=False)
        encoding = "latin-1"

    # Attempt to parse datetime columns
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(20).astype(str)
            if sample.str.match(r"\d{4}[-/]\d{2}[-/]\d{2}").mean() > 0.7:
                try:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                except Exception:
                    pass

    return df, encoding
