"""Vector Embedding Service Module.

Generates dense semantic vector embeddings for document chunks and user queries
using Hugging Face sentence-transformers. Supports batch processing, L2 normalization,
and configurable model backends.
"""

from functools import lru_cache
import logging
from typing import List, Optional, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


logger = logging.getLogger(__name__)


class EmbeddingService:
    """Encapsulates sentence-transformers model loading, batch encoding, and normalization."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model = None

    @property
    def model(self):
        """Lazy-loads the SentenceTransformer model on first invocation."""
        if self._model is None:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is not installed. Please install it with: pip install sentence-transformers"
                )
            logger.info("Loading embedding model: '%s'", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Returns the dimensionality of the generated embedding vectors (e.g., 384 for all-MiniLM-L6-v2)."""
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generates normalized vector embeddings for a list of document chunk texts.

        Args:
            texts: List of chunk text strings.

        Returns:
            np.ndarray of shape (len(texts), dimension), dtype float32.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        logger.debug("Embedding %d document chunks in batches of %d", len(texts), self.batch_size)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generates a single normalized vector embedding for a user search query.

        Args:
            query: The user query string.

        Returns:
            np.ndarray of shape (dimension,), dtype float32.
        """
        if not query.strip():
            raise ValueError("Query string cannot be empty for embedding.")

        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)

    def compute_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Computes cosine similarity scores between query and document vectors.

        Because embeddings are L2 normalized, cosine similarity is equivalent to dot product.
        """
        if doc_embeddings.ndim == 1:
            doc_embeddings = doc_embeddings.reshape(1, -1)
        return np.dot(doc_embeddings, query_embedding)


# Global cached singleton instance for the default model
_DEFAULT_EMBEDDER = None


def get_embedding_service(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingService:
    """Returns or initializes a cached EmbeddingService instance."""
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None or _DEFAULT_EMBEDDER.model_name != model_name:
        _DEFAULT_EMBEDDER = EmbeddingService(model_name=model_name)
    return _DEFAULT_EMBEDDER
