"""Query and Question-Answering Endpoints."""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.rag.pipeline import RAGResponse, SourceCitation
from app.state import AppState, get_app_state


logger = logging.getLogger(__name__)
router = APIRouter(tags=["Query"])


class ChatMessage(BaseModel):
    role: str = Field(description="'user' or 'assistant'")
    content: str = Field(description="Message body")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question to answer")
    top_k: Optional[int] = Field(default=None, ge=1, le=50, description="Override top-K retrieved chunks")
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override minimum similarity threshold"
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional filter to search only within a specific document"
    )
    conversation_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Previous conversation turns for context-aware query rewriting"
    )


@router.post("/query", response_model=RAGResponse)
def ask_question(
    request: QueryRequest,
    state: AppState = Depends(get_app_state),
):
    """Executes the full RAG pipeline: query rewriting, semantic retrieval, reranking, and grounded generation."""
    history_dicts = None
    if request.conversation_history:
        history_dicts = [{"role": m.role, "content": m.content} for m in request.conversation_history]

    try:
        response = state.pipeline.answer_question(
            question=request.question,
            conversation_history=history_dicts,
            document_id_filter=request.document_id,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
        )
        return response

    except Exception as e:
        logger.error("Query processing failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to process query: {e}")

