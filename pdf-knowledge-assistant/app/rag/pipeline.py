"""End-to-End RAG Pipeline Orchestrator.

Coordinates query rewriting, dense retrieval, reranking, anti-hallucination prompt construction,
LLM generation, and source citation extraction.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.generation.llm import BaseLLMService, LLMError
from app.generation.prompts import (
    SYSTEM_RAG_PROMPT,
    build_context_string,
    build_rag_prompt,
)
from app.retrieval.retriever import DocumentRetriever, RetrievalResult
from app.vectorstore.base import SearchResult


logger = logging.getLogger(__name__)


class SourceCitation(BaseModel):
    """Structured citation metadata for evidence attribution."""

    filename: str = Field(description="Name of the source PDF document")
    page: int = Field(description="1-indexed page number where content was located")
    section: str = Field(default="General", description="Detected section heading")
    chunk_id: str = Field(description="Unique identifier of the matching chunk")
    score: float = Field(description="Relevance similarity score")
    snippet: str = Field(description="Text excerpt from the matching chunk")


class RAGResponse(BaseModel):
    """Complete response returned by the RAG system."""

    question: str
    rewritten_question: Optional[str] = None
    answer: str
    sources: List[SourceCitation] = Field(default_factory=list)
    has_sufficient_context: bool = True
    retrieval_count: int = 0


class RAGPipeline:
    """Production-grade RAG pipeline orchestrating retrieval, grounding, and answer synthesis."""

    INSUFFICIENT_CONTEXT_MESSAGE = (
        "I couldn't find enough relevant information in the uploaded documents to answer this question."
    )

    def __init__(
        self,
        retriever: DocumentRetriever,
        llm_service: BaseLLMService,
        reranker: Optional[Any] = None,
        query_rewriter: Optional[Any] = None,
    ):
        self.retriever = retriever
        self.llm_service = llm_service
        self.reranker = reranker
        self.query_rewriter = query_rewriter

    def answer_question(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        document_id_filter: Optional[str] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> RAGResponse:
        """Executes the complete RAG query lifecycle.

        Args:
            question: The user's input question.
            conversation_history: Optional list of previous chat turns [{"role": "user"|"assistant", "content": "..."}].
            document_id_filter: Optional filter restricting search to a specific document ID.
            top_k: Number of chunks to feed to generation.
            similarity_threshold: Score threshold below which chunks are discarded.

        Returns:
            RAGResponse containing generated answer and verified source citations.
        """
        logger.info("RAG Query received: '%s'", question)
        effective_query = question.strip()
        rewritten_query: Optional[str] = None

        # 1. Query Rewriting (if history is present and query rewriter is configured)
        if self.query_rewriter and conversation_history:
            try:
                rewritten_query = self.query_rewriter.rewrite(question, conversation_history)
                if rewritten_query and rewritten_query != question:
                    logger.info("Rewrote query from '%s' to '%s'", question, rewritten_query)
                    effective_query = rewritten_query
            except Exception as e:
                logger.warning("Query rewriting failed; falling back to original question: %s", e)

        # 2. Vector Retrieval
        filter_dict = {"document_id": document_id_filter} if document_id_filter else None
        retrieval_result: RetrievalResult = self.retriever.retrieve(
            query=effective_query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter_dict=filter_dict,
        )

        # 3. Optional Reranking
        candidate_chunks: List[SearchResult] = retrieval_result.chunks
        if self.reranker and candidate_chunks:
            try:
                candidate_chunks = self.reranker.rerank(
                    query=effective_query,
                    candidates=candidate_chunks,
                    top_n=top_k or self.retriever.default_top_k,
                )
                logger.info("Cross-Encoder reranking applied to %d candidates", len(candidate_chunks))
            except Exception as e:
                logger.warning("Reranking failed; continuing with dense retrieval order: %s", e)

        # 4. Context Sufficiency Check
        if not candidate_chunks or not retrieval_result.has_sufficient_context:
            logger.info("Insufficient context found for query '%s'", effective_query)
            return RAGResponse(
                question=question,
                rewritten_question=rewritten_query,
                answer=self.INSUFFICIENT_CONTEXT_MESSAGE,
                sources=[],
                has_sufficient_context=False,
                retrieval_count=0,
            )

        # 5. Context Construction
        context_str = build_context_string(candidate_chunks)
        prompt = build_rag_prompt(question=effective_query, context=context_str)

        # 6. LLM Generation
        try:
            raw_answer = self.llm_service.generate(
                prompt=prompt,
                system_instruction=SYSTEM_RAG_PROMPT,
                temperature=0.0,
            )
        except LLMError as e:
            logger.error("LLM Generation error: %s", e)
            return RAGResponse(
                question=question,
                rewritten_question=rewritten_query,
                answer=f"Error generating answer: {e}",
                sources=[],
                has_sufficient_context=False,
                retrieval_count=len(candidate_chunks),
            )

        # 7. Build Source Citations
        citations: List[SourceCitation] = []
        for res in candidate_chunks:
            c = res.chunk
            # Create a 200-character snippet for previewing
            snippet = c.text[:200] + "..." if len(c.text) > 200 else c.text
            citations.append(
                SourceCitation(
                    filename=c.filename,
                    page=c.page,
                    section=c.section,
                    chunk_id=c.chunk_id,
                    score=round(res.score, 4),
                    snippet=snippet,
                )
            )

        return RAGResponse(
            question=question,
            rewritten_question=rewritten_query,
            answer=raw_answer,
            sources=citations,
            has_sufficient_context=True,
            retrieval_count=len(candidate_chunks),
        )

