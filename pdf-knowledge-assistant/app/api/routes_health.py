"""Health and status check endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.config import Settings, get_settings


router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    app_version: str
    llm_provider: str
    llm_model: str
    embedding_model: str
    vector_store: str


@router.get("/health", response_model=HealthResponse)
def get_health(settings: Settings = Depends(get_settings)):
    """Returns application health status and current engine configuration."""
    return HealthResponse(
        status="ok",
        app_version="1.0.0",
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        vector_store=settings.vector_store,
    )

