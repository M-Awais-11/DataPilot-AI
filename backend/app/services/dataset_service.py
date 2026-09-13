"""Dataset service — file upload, storage, profiling, retrieval."""
import os
import uuid
import shutil
import asyncio
from pathlib import Path
from typing import Optional
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete as sql_delete

from app.config.settings import settings
from app.models.dataset import Dataset, DatasetProfile as DatasetProfileModel, DatasetVersion
from app.models.experiment import Experiment, AgentRun, PreprocessingStep, ProgressEvent
from app.models.model import TrainedModel, ModelMetric, FeatureImportance, Prediction, Report
from app.models.chat import ChatMessage
from app.models.user import User
from app.tools.profiling import profile_dataframe, load_csv_safe


ALLOWED_EXTENSIONS = {".csv"}
MAX_COLUMNS = 500


class DatasetService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def upload_dataset(self, file: UploadFile, user: User) -> Dataset:
        """Validate, store, and profile an uploaded CSV file."""
        # ── Validate extension
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Only CSV files are accepted. Got: {suffix or 'unknown'}",
            )

        # ── Read content and validate size
        content = await file.read()
        if len(content) > settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum of {settings.max_upload_size_mb} MB",
            )
        if len(content) == 0:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty file uploaded")

        # ── Create user-specific storage directory
        user_dir = Path(settings.upload_dir) / str(user.id)
        user_dir.mkdir(parents=True, exist_ok=True)

        dataset_id = uuid.uuid4()
        file_name = f"{dataset_id}{suffix}"
        file_path = user_dir / file_name

        # ── Save original file
        with open(file_path, "wb") as f:
            f.write(content)

        # ── Create dataset record
        dataset = Dataset(
            id=dataset_id,
            user_id=user.id,
            name=Path(file.filename or file_name).stem,
            original_name=file.filename or file_name,
            file_path=str(file_path),
            file_size=len(content),
            status="profiling",
        )
        self.db.add(dataset)
        await self.db.flush()

        # ── Profile synchronously (fast for small files)
        try:
            df, encoding = load_csv_safe(str(file_path))
            if df.shape[1] > MAX_COLUMNS:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Dataset has too many columns ({df.shape[1]}). Maximum is {MAX_COLUMNS}.",
                )

            profile = profile_dataframe(df)
            dataset.row_count = profile.row_count
            dataset.col_count = profile.col_count
            dataset.status = "ready"

            # Store profile in DB
            profile_record = DatasetProfileModel(
                dataset_id=dataset_id,
                profile_json=profile.to_dict(),
                target_column=profile.target_column_suggestion,
                problem_type=profile.problem_type_suggestion,
            )
            self.db.add(profile_record)

            # Create version 1 (the original)
            version = DatasetVersion(
                dataset_id=dataset_id,
                version_number=1,
                file_path=str(file_path),
                transformations={"description": "Original uploaded file", "steps": []},
            )
            self.db.add(version)

        except HTTPException:
            raise
        except Exception as e:
            dataset.status = "error"
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to parse CSV: {str(e)}",
            )

        return dataset

    async def list_datasets(self, user: User) -> list[Dataset]:
        result = await self.db.execute(
            select(Dataset).where(Dataset.user_id == user.id).order_by(Dataset.created_at.desc())
        )
        return result.scalars().all()

    async def get_dataset(self, dataset_id: str, user: User) -> Dataset:
        result = await self.db.execute(
            select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == user.id)
        )
        dataset = result.scalar_one_or_none()
        if not dataset:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
        return dataset

    async def get_dataset_profile(self, dataset_id: str, user: User) -> DatasetProfileModel:
        # Ensure user owns the dataset
        await self.get_dataset(dataset_id, user)
        result = await self.db.execute(
            select(DatasetProfileModel).where(DatasetProfileModel.dataset_id == dataset_id)
        )
        profile = result.scalar_one_or_none()
        if not profile:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not yet generated")
        return profile

    async def get_preview(self, dataset_id: str, user: User, n_rows: int = 100) -> dict:
        dataset = await self.get_dataset(dataset_id, user)
        try:
            df, _ = load_csv_safe(dataset.file_path, max_rows=n_rows)
            return {
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "rows": df.head(n_rows).fillna("").to_dict(orient="records"),
                "total_rows_in_preview": len(df),
            }
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def delete_dataset(self, dataset_id: str, user: User) -> None:
        dataset = await self.get_dataset(dataset_id, user)

        # 1. Find all experiments that reference this dataset
        exp_res = await self.db.execute(
            select(Experiment).where(Experiment.dataset_id == dataset.id)
        )
        experiments = exp_res.scalars().all()

        for exp in experiments:
            # Delete model files and DB records for each experiment
            models_res = await self.db.execute(
                select(TrainedModel).where(TrainedModel.experiment_id == exp.id)
            )
            models = models_res.scalars().all()
            for m in models:
                if m.artifact_path and os.path.exists(m.artifact_path):
                    try:
                        os.remove(m.artifact_path)
                    except OSError:
                        pass
                await self.db.execute(sql_delete(Prediction).where(Prediction.model_id == m.id))
                await self.db.execute(sql_delete(ModelMetric).where(ModelMetric.model_id == m.id))
                await self.db.execute(sql_delete(FeatureImportance).where(FeatureImportance.model_id == m.id))

            await self.db.execute(sql_delete(TrainedModel).where(TrainedModel.experiment_id == exp.id))

            # Delete report files and record
            report_res = await self.db.execute(select(Report).where(Report.experiment_id == exp.id))
            report = report_res.scalar_one_or_none()
            if report:
                if report.file_path and os.path.exists(report.file_path):
                    try:
                        os.remove(report.file_path)
                    except OSError:
                        pass
                await self.db.delete(report)

            # Delete experiment charts and models directories
            charts_dir = Path(settings.upload_dir) / str(user.id) / "charts" / str(exp.id)
            if charts_dir.exists():
                shutil.rmtree(charts_dir, ignore_errors=True)

            exp_models_dir = Path(settings.upload_dir) / str(user.id) / "models" / str(exp.id)
            if exp_models_dir.exists():
                shutil.rmtree(exp_models_dir, ignore_errors=True)

            # Delete runs, steps, events, messages
            await self.db.execute(sql_delete(ChatMessage).where(ChatMessage.experiment_id == exp.id))
            await self.db.execute(sql_delete(ProgressEvent).where(ProgressEvent.experiment_id == exp.id))
            await self.db.execute(sql_delete(AgentRun).where(AgentRun.experiment_id == exp.id))
            await self.db.execute(sql_delete(PreprocessingStep).where(PreprocessingStep.experiment_id == exp.id))

            # Delete the experiment itself
            await self.db.delete(exp)

        # 2. Delete dataset profiles and versions
        await self.db.execute(sql_delete(DatasetProfileModel).where(DatasetProfileModel.dataset_id == dataset.id))
        await self.db.execute(sql_delete(DatasetVersion).where(DatasetVersion.dataset_id == dataset.id))

        # 3. Remove physical files on disk
        try:
            if dataset.file_path and os.path.exists(dataset.file_path):
                os.remove(dataset.file_path)
            user_dir = Path(settings.upload_dir) / str(user.id)
            for f in user_dir.glob(f"{dataset.id}*"):
                f.unlink(missing_ok=True)
        except Exception:
            pass

        # 4. Delete the dataset record and commit transaction
        await self.db.delete(dataset)
        await self.db.commit()

