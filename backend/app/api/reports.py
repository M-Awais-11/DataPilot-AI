"""Reports API router — download and inspect generated PDF reports."""
import os
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.connection import get_db
from app.models.user import User
from app.models.experiment import Experiment
from app.models.model import Report
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/{experiment_id}")
async def get_report_metadata(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get report metadata for an experiment."""
    # Verify experiment ownership
    exp_res = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id, Experiment.user_id == current_user.id)
    )
    if not exp_res.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")

    rep_res = await db.execute(select(Report).where(Report.experiment_id == experiment_id))
    report = rep_res.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report has not been generated yet.")

    return {
        "id": str(report.id),
        "experiment_id": str(report.experiment_id),
        "file_path": report.file_path,
        "generated_at": report.generated_at,
        "available": os.path.exists(report.file_path),
    }


@router.get("/{experiment_id}/download")
async def download_report(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Download the executive PDF report."""
    # Verify experiment ownership
    exp_res = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id, Experiment.user_id == current_user.id)
    )
    if not exp_res.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")

    rep_res = await db.execute(select(Report).where(Report.experiment_id == experiment_id))
    report = rep_res.scalar_one_or_none()
    if not report or not os.path.exists(report.file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report file not found")

    return FileResponse(
        path=report.file_path,
        media_type="application/pdf",
        filename=f"Data_Scientist_Report_{experiment_id[:8]}.pdf",
    )
