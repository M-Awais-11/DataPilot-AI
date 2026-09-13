"""Pydantic schemas for AI Data Scientist Chatbot endpoints."""
from datetime import datetime
from uuid import UUID
from typing import Any, Optional, Union
from pydantic import BaseModel, Field


class ChatMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatMessageResponse(BaseModel):
    id: Union[UUID, str]
    role: str
    content: str
    context_used: Optional[dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}
