import os
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
from fastapi import HTTPException

from app.api.models import delete_model, list_models
from app.models.model import TrainedModel
from app.models.user import User


@pytest.mark.asyncio
async def test_delete_model_removes_physical_file_and_db_records(tmp_path):
    # Create dummy artifact file on disk
    artifact_file = tmp_path / "test_model_v1.pkl"
    artifact_file.write_bytes(b"dummy binary model data")
    assert artifact_file.exists()

    model_id = str(uuid.uuid4())
    user_id = uuid.uuid4()
    exp_id = uuid.uuid4()

    mock_user = MagicMock(spec=User)
    mock_user.id = user_id

    mock_model = MagicMock(spec=TrainedModel)
    mock_model.id = uuid.UUID(model_id)
    mock_model.experiment_id = exp_id
    mock_model.algorithm = "RandomForestClassifier"
    mock_model.artifact_path = str(artifact_file)
    mock_model.is_best = False

    # Mock DB session
    mock_db = AsyncMock()
    mock_execute_res = MagicMock()
    mock_execute_res.scalar_one_or_none.return_value = mock_model
    mock_db.execute.return_value = mock_execute_res

    # Call delete_model
    result = await delete_model(model_id=model_id, current_user=mock_user, db=mock_db)

    # 1. Assert physical file was deleted from disk
    assert not os.path.exists(str(artifact_file))
    assert not artifact_file.exists()

    # 2. Assert DB delete was called
    assert mock_db.delete.called
    assert mock_db.commit.called

    # 3. Assert return payload
    assert result["status"] == "deleted"
    assert result["model_id"] == model_id
    assert result["deleted_artifact"] == str(artifact_file)


@pytest.mark.asyncio
async def test_delete_model_not_found():
    mock_user = MagicMock(spec=User)
    mock_user.id = uuid.uuid4()

    mock_db = AsyncMock()
    mock_execute_res = MagicMock()
    mock_execute_res.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_execute_res

    with pytest.raises(HTTPException) as exc_info:
        await delete_model(model_id=str(uuid.uuid4()), current_user=mock_user, db=mock_db)

    assert exc_info.value.status_code == 404
