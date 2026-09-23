"""Document Retriever Module.

Orchestrates dense vector similarity search, BM25 lexical keyword retrieval,
Reciprocal Rank Fusion (RRF), and intelligent out-of-domain rejection gating.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

from app.embeddings.embedding_service import EmbeddingService, get_embedding_service
from app.retrieval.bm25 import BM25Index
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
    """Coordinates hybrid semantic (dense) + lexical (BM25) retrieval."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_service: Optional[EmbeddingService] = None,
        similarity_threshold: float = 0.05,
        default_top_k: int = 5,
        use_hybrid: bool = True,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service or get_embedding_service()
        self.similarity_threshold = similarity_threshold
        self.default_top_k = default_top_k
        self.use_hybrid = use_hybrid
        self.rrf_k = rrf_k
        self.bm25_index = BM25Index()
        self._last_chunk_count: int = -1

    def _sync_bm25(self) -> None:
        """Synchronizes the BM25 index with the current chunks in the vector store."""
        current_count = self.vector_store.total_chunks()
        if current_count != self._last_chunk_count:
            all_chunks = self.vector_store.get_all_chunks()
            self.bm25_index.index_chunks(all_chunks)
            self._last_chunk_count = current_count
            logger.debug("BM25 index synchronized with %d chunks", len(all_chunks))

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """Retrieves semantically and lexically relevant chunks for a given query.

        Args:
            query: Natural language question or search query.
            top_k: Maximum number of chunks to return (defaults to self.default_top_k).
            similarity_threshold: Minimum score threshold required for relevance gating.
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

        # 1. Dense Vector Retrieval
        query_vec = self.embedding_service.embed_query(clean_query)
        dense_candidates = self.vector_store.similarity_search(query_vec, k=max(k * 2, 10), filter_dict=filter_dict)

        dense_best_score = dense_candidates[0].score if dense_candidates else 0.0

        # 2. Lexical BM25 Retrieval (if enabled)
        bm25_candidates: List[SearchResult] = []
        if self.use_hybrid:
            self._sync_bm25()
            bm25_candidates = self.bm25_index.search(clean_query, k=max(k * 2, 10), filter_dict=filter_dict)

        bm25_best_score = bm25_candidates[0].score if bm25_candidates else 0.0

        # 3. Reciprocal Rank Fusion (RRF)
        fused_chunks: List[SearchResult] = []
        if self.use_hybrid and (dense_candidates or bm25_candidates):
            rrf_scores: Dict[str, float] = {}
            chunk_map: Dict[str, SearchResult] = {}
            score_map: Dict[str, float] = {}

            # Dense ranks
            for rank, item in enumerate(dense_candidates, start=1):
                cid = item.chunk.chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
                chunk_map[cid] = item
                score_map[cid] = max(score_map.get(cid, 0.0), item.score)

            # BM25 ranks
            for rank, item in enumerate(bm25_candidates, start=1):
                cid = item.chunk.chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
                if cid not in chunk_map:
                    chunk_map[cid] = item
                    score_map[cid] = min(0.35, item.score / 10.0)
                else:
                    # Boost dense score slightly if confirmed by lexical match
                    score_map[cid] = min(1.0, score_map[cid] + 0.08)

            # Sort by RRF score descending
            sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
            for cid in sorted_cids:
                orig_item = chunk_map[cid]
                fused_chunks.append(SearchResult(chunk=orig_item.chunk, score=round(score_map[cid], 4)))
        else:
            fused_chunks = dense_candidates

        if not fused_chunks:
            return RetrievalResult(query=clean_query, has_sufficient_context=False)

        best_score = fused_chunks[0].score

        # 4. Out-of-Domain / Context Sufficiency Gate:
        # A query is considered relevant if dense similarity is at or above threshold OR BM25 found strong lexical evidence
        has_dense_signal = dense_best_score >= threshold
        has_bm25_signal = bm25_best_score >= 0.5
        has_context = has_dense_signal or has_bm25_signal

        # For qualified queries, provide the top k candidate chunks
        final_chunks = fused_chunks[:k] if has_context else []

        logger.info(
            "Hybrid Retrieval for '%s': Dense best=%.3f, BM25 best=%.3f, Combined best=%.3f. Sufficient: %s, Chunks: %d",
            clean_query,
            dense_best_score,
            bm25_best_score,
            best_score,
            has_context,
            len(final_chunks),
        )

        return RetrievalResult(
            query=clean_query,
            chunks=final_chunks,
            has_sufficient_context=has_context,
            best_score=best_score,
            all_candidates=fused_chunks,
        )
