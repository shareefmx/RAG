"""Cross-Encoder Reranker Module.

Performs two-stage retrieval refinement using full cross-attention between
the user query and candidate document chunks.
"""

import logging
from typing import List, Optional

from app.vectorstore.base import SearchResult


logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-scores candidate chunks using a Transformer Cross-Encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy-loads the CrossEncoder model upon first use."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading CrossEncoder reranker model: '%s'", self.model_name)
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                logger.error("Failed to load CrossEncoder model '%s': %s", self.model_name, e)
                raise
        return self._model

    def rerank(self, query: str, candidates: List[SearchResult], top_n: int = 5) -> List[SearchResult]:
        """Re-scores and sorts candidate chunks by cross-encoder relevance.

        Args:
            query: The user query string.
            candidates: List of candidate SearchResult objects from vector retrieval.
            top_n: Number of highest-ranked chunks to retain.

        Returns:
            Reordered list of SearchResult objects with updated scores.
        """
        if not candidates:
            return []

        # Pair query with each candidate's text
        pairs = [[query, res.chunk.text] for res in candidates]

        scores = self.model.predict(pairs)

        # Build new SearchResult list with cross-encoder scores
        scored_candidates: List[SearchResult] = []
        for res, ce_score in zip(candidates, scores):
            # Normalize cross-encoder logit/score for consistency
            scored_candidates.append(
                SearchResult(chunk=res.chunk, score=float(ce_score))
            )

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            "Reranked %d candidates to top %d (Highest score: %.4f)",
            len(candidates),
            min(top_n, len(scored_candidates)),
            scored_candidates[0].score if scored_candidates else 0.0,
        )

        return scored_candidates[:top_n]

