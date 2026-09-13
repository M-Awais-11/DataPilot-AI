"""Pydantic schemas for dataset-related endpoints."""
from datetime import datetime
from uuid import UUID
from typing import Any, Optional, Union
from pydantic import BaseModel


class DatasetUploadResponse(BaseModel):
    id: Union[UUID, str]
    name: str
    original_name: str
    file_size: Optional[int]
    row_count: Optional[int]
    col_count: Optional[int]
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetListItem(BaseModel):
    id: Union[UUID, str]
    name: str
    original_name: str
    row_count: Optional[int]
    col_count: Optional[int]
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetProfileResponse(BaseModel):
    id: Union[UUID, str]
    dataset_id: Union[UUID, str]
    profile_json: dict
    target_column: Optional[str]
    problem_type: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetPreviewResponse(BaseModel):
    columns: list[str]
    dtypes: dict[str, str]
    rows: list[dict[str, Any]]
    total_rows_in_preview: int
