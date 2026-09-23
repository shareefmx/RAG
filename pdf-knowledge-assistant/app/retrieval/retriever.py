"""Document Retriever Module.

Orchestrates query embedding, vector database search, similarity threshold gating,
and metadata filtering to prevent irrelevant chunks from polluting LLM context.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

from app.embeddings.embedding_service import EmbeddingService, get_embedding_service
from app.vectorstore.base import BaseVectorStore, SearchResult


logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Encapsulates the output of a retrieval request."""

    query: str
    chunks: List[SearchResult] = field(default_factory=list)
    has_sufficient_context: bool = False
    best_score: float = 0.0
    all_candidates: List[SearchResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "has_sufficient_context": self.has_sufficient_context,
            "best_score": round(self.best_score, 4),
            "chunks_count": len(self.chunks),
            "chunks": [c.to_dict() for c in self.chunks],
        }


class DocumentRetriever:
    """Coordinates semantic retrieval with score thresholding and metadata filters."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_service: Optional[EmbeddingService] = None,
        similarity_threshold: float = 0.35,
        default_top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service or get_embedding_service()
        self.similarity_threshold = similarity_threshold
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """Retrieves semantically relevant document chunks for a given query.

        Args:
            query: Natural language question or search query.
            top_k: Maximum number of chunks to return (defaults to self.default_top_k).
            similarity_threshold: Minimum cosine similarity score required.
            filter_dict: Optional metadata filter, e.g. {"document_id": "doc_123"}.

        Returns:
            RetrievalResult containing vetted chunks and sufficiency status.
        """
        k = top_k if top_k is not None else self.default_top_k
        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold

        clean_query = query.strip()
        if not clean_query:
            logger.warning("Empty query passed to retriever.")
            return RetrievalResult(query=query, has_sufficient_context=False)

        if self.vector_store.total_chunks() == 0:
            logger.info("Retriever called but vector store is empty.")
            return RetrievalResult(query=clean_query, has_sufficient_context=False)

        # 1. Embed query
        query_vec = self.embedding_service.embed_query(clean_query)

        # 2. Similarity search
        candidates = self.vector_store.similarity_search(query_vec, k=k, filter_dict=filter_dict)

        if not candidates:
            return RetrievalResult(query=clean_query, has_sufficient_context=False)

        best_score = candidates[0].score

        # 3. Apply similarity threshold
        qualified_chunks = [c for c in candidates if c.score >= threshold]

        has_context = len(qualified_chunks) > 0

        logger.info(
            "Retrieved %d candidates (best score: %.3f). %d chunks passed threshold (%.2f). Sufficient: %s",
            len(candidates),
            best_score,
            len(qualified_chunks),
            threshold,
            has_context,
        )

        return RetrievalResult(
            query=clean_query,
            chunks=qualified_chunks,
            has_sufficient_context=has_context,
            best_score=best_score,
            all_candidates=candidates,
        )

