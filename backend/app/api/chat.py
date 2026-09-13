"""Chat API router — interactive queries and agent commands."""
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.connection import get_db
from app.models.user import User
from app.models.experiment import Experiment
from app.models.chat import ChatMessage
from app.schemas.chat import ChatMessageRequest, ChatMessageResponse
from app.services.auth_service import get_current_user
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/{experiment_id}", response_model=ChatMessageResponse)
async def send_chat_message(
    experiment_id: str,
    data: ChatMessageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Chat with the AI Data Scientist about an experiment, or send interactive commands."""
    exp_res = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id, Experiment.user_id == current_user.id)
    )
    if not exp_res.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")

    chat_svc = ChatService(db)
    response_text, context_used = await chat_svc.handle_chat(
        experiment_id=experiment_id,
        user_message=data.message,
        user=current_user,
    )

    # Return the assistant's reply
    latest_msg = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.experiment_id == experiment_id, ChatMessage.role == "assistant")
        .order_by(ChatMessage.created_at.desc())
    )
    assistant_msg = latest_msg.scalars().first()

    return assistant_msg


@router.get("/{experiment_id}/history", response_model=list[ChatMessageResponse])
async def get_chat_history(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve full conversation history for an experiment."""
    exp_res = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id, Experiment.user_id == current_user.id)
    )
    if not exp_res.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")

    msgs = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.experiment_id == experiment_id)
        .order_by(ChatMessage.created_at.asc())
    )
    return msgs.scalars().all()
