"""Pydantic schemas for Experiment endpoints."""
from datetime import datetime
from uuid import UUID
from typing import Any, Optional, Union
from pydantic import BaseModel, Field


class ExperimentCreateRequest(BaseModel):
    dataset_id: str
    name: str = Field(min_length=1, max_length=255)
    description: Optional[str] = None
    target_column: Optional[str] = None
    primary_metric: Optional[str] = None
    commands: Optional[list[str]] = Field(default_factory=list)


class ExperimentResponse(BaseModel):
    id: Union[UUID, str]
    dataset_id: Union[UUID, str]
    name: str
    description: Optional[str] = None
    target_column: Optional[str] = None
    problem_type: Optional[str] = None
    primary_metric: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class AgentRunResponse(BaseModel):
    id: Union[UUID, str]
    agent_name: str
    status: str
    reasoning: Optional[str] = None
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class PreprocessingStepResponse(BaseModel):
    id: Union[UUID, str]
    step_order: int
    step_type: Optional[str] = None
    column_name: Optional[str] = None
    operation: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None
    reason: Optional[str] = None
    rows_affected: Optional[int] = None

    model_config = {"from_attributes": True}


class InteractiveCommandRequest(BaseModel):
    command: str = Field(min_length=1, max_length=1000)
