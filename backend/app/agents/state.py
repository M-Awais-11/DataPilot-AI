"""
LangGraph State definition for the Autonomous Data Science workflow.
All agents read from and write to this shared state object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# ── Sub-models (Pydantic-style but plain dataclasses for speed) ───────────────

@dataclass
class ColumnStat:
    name: str
    dtype: str
    semantic_type: str
    null_pct: float
    unique_count: int
    is_constant: bool = False
    is_likely_id: bool = False
    is_possible_target: bool = False
    leakage_risk: str = "none"


@dataclass
class PreprocessingStepRecord:
    step_order: int
    step_type: str
    column_name: Optional[str]
    operation: str
    parameters: dict
    reason: str
    rows_affected: int = 0


@dataclass
class EDAInsight:
    chart_type: str
    columns: list[str]
    chart_path: str
    insight: str
    importance: str = "medium"   # low | medium | high


@dataclass
class FeatureOperation:
    operation: str
    columns: list[str]
    new_columns: list[str]
    reason: str
    parameters: dict = field(default_factory=dict)


@dataclass
class TrainedModelResult:
    model_id: str
    algorithm: str
    hyperparameters: dict
    artifact_path: str
    metrics: dict[str, dict[str, float]]   # split -> metric_name -> value
    cv_scores: dict[str, list[float]]
    training_time_seconds: float


@dataclass
class ModelEvaluation:
    model_id: str
    algorithm: str
    primary_metric: str
    primary_metric_value: float
    all_metrics: dict[str, float]
    overfitting_score: float     # train_score - val_score
    is_overfitting: bool


@dataclass
class SHAPResults:
    model_id: str
    summary_plot_path: str
    bar_plot_path: str
    feature_importance: list[dict]   # [{feature, shap_mean_abs, rank}]
    local_explanation_available: bool


@dataclass
class AgentRunRecord:
    agent_name: str
    status: str                   # pending | running | completed | failed | skipped
    reasoning: str = ""
    error: Optional[str] = None
    duration_ms: int = 0


# ── Main State TypedDict ───────────────────────────────────────────────────────
from typing import TypedDict


class DataScienceState(TypedDict, total=False):
    # ── Identity ──────────────────────────────────────────────────────────────
    experiment_id: str
    user_id: str
    dataset_id: str

    # ── Raw data references ───────────────────────────────────────────────────
    original_file_path: str
    working_file_path: str       # current working version (post-cleaning)
    df_shape: tuple[int, int]

    # ── Dataset profile (from profiling tool) ─────────────────────────────────
    profile_json: dict           # full profile dict from DatasetProfile.to_dict()
    numerical_columns: list[str]
    categorical_columns: list[str]
    boolean_columns: list[str]
    datetime_columns: list[str]
    id_columns: list[str]
    constant_columns: list[str]
    target_column: Optional[str]
    target_confirmed: bool
    target_confidence: float
    problem_type: Optional[str]  # classification | regression | clustering
    class_distribution: Optional[dict]
    class_imbalance: bool
    data_quality_issues: list[str]

    # ── Cleaning ──────────────────────────────────────────────────────────────
    cleaning_log: list[PreprocessingStepRecord]
    cleaned_file_path: Optional[str]
    leakage_warnings: list[str]
    dropped_columns: list[str]

    # ── EDA ───────────────────────────────────────────────────────────────────
    eda_insights: list[EDAInsight]
    visualization_paths: list[str]
    eda_summary: str              # human-readable LLM summary

    # ── Feature Engineering ───────────────────────────────────────────────────
    feature_engineering_log: list[FeatureOperation]
    selected_features: list[str]
    dropped_features: list[str]
    feature_engineered_file_path: Optional[str]

    # ── Model Selection ───────────────────────────────────────────────────────
    candidate_models: list[str]
    model_selection_reason: str

    # ── Training ─────────────────────────────────────────────────────────────
    trained_models: list[TrainedModelResult]
    best_model_id: Optional[str]
    best_model_name: Optional[str]
    primary_metric: str

    # ── Evaluation ────────────────────────────────────────────────────────────
    evaluation_results: dict[str, ModelEvaluation]
    model_comparison_table: list[dict]
    overfitting_detected: bool
    overfitting_details: Optional[str]

    # ── Explainability ────────────────────────────────────────────────────────
    shap_results: Optional[SHAPResults]
    feature_importance: list[dict]

    # ── Report ────────────────────────────────────────────────────────────────
    report_path: Optional[str]

    # ── Workflow control ──────────────────────────────────────────────────────
    current_agent: str
    next_agent: Optional[str]
    agent_history: list[AgentRunRecord]
    error: Optional[str]
    should_stop: bool
    iteration_count: int
    max_iterations: int
    user_commands: list[str]         # queued interactive commands

    # ── LangGraph message history ─────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
