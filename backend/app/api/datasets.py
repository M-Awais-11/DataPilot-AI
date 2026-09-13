"""Dataset API router — upload, list, detail, profile, preview, delete."""
from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.models.user import User
from app.schemas.dataset import (
    DatasetUploadResponse,
    DatasetListItem,
    DatasetProfileResponse,
    DatasetPreviewResponse,
)
from app.services.auth_service import get_current_user
from app.services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.post("/upload", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a CSV dataset. The system will immediately profile it."""
    service = DatasetService(db)
    dataset = await service.upload_dataset(file, current_user)
    return dataset


@router.get("", response_model=list[DatasetListItem])
async def list_datasets(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all datasets owned by the current user."""
    service = DatasetService(db)
    return await service.list_datasets(current_user)


@router.get("/{dataset_id}", response_model=DatasetUploadResponse)
async def get_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get dataset details. Returns 404 if not found or not owned by current user."""
    service = DatasetService(db)
    return await service.get_dataset(dataset_id, current_user)


@router.get("/{dataset_id}/profile", response_model=DatasetProfileResponse)
async def get_dataset_profile(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the structured profile JSON for a dataset."""
    service = DatasetService(db)
    return await service.get_dataset_profile(dataset_id, current_user)


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
async def get_dataset_preview(
    dataset_id: str,
    n_rows: int = Query(default=100, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the first N rows of the dataset for preview."""
    service = DatasetService(db)
    return await service.get_preview(dataset_id, current_user, n_rows)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a dataset and all its associated files."""
    service = DatasetService(db)
    await service.delete_dataset(dataset_id, current_user)
