"""Vector Store Base Interface and Common Types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from app.ingestion.chunker import DocumentChunk


@dataclass
class SearchResult:
    """Represents a matched chunk returned from vector similarity search."""

    chunk: DocumentChunk
    score: float  # Cosine similarity score in range [-1.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to a dictionary."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": round(self.score, 4),
        }


class BaseVectorStore(ABC):
    """Abstract interface defining standard vector database operations."""

    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Adds document chunks and their corresponding embedding vectors to the index.

        Args:
            chunks: List of DocumentChunk objects.
            embeddings: 2D numpy array of shape (N, dimension) containing float32 vectors.
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Performs nearest-neighbor search for a query embedding.

        Args:
            query_embedding: 1D numpy array of shape (dimension,) or 2D of shape (1, dimension).
            k: Maximum number of candidate results to return.
            filter_dict: Optional metadata key-value filters (e.g. {"document_id": "doc_123"}).

        Returns:
            List of SearchResult objects ordered by descending similarity score.
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Removes all chunks associated with a specific document_id.

        Returns:
            Count of deleted chunks.
        """
        pass

    @abstractmethod
    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns metadata summaries of all currently indexed documents."""
        pass

    @abstractmethod
    def total_chunks(self) -> int:
        """Returns the total number of indexed chunks."""
        pass

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Persists the vector index and associated metadata to disk."""
        pass

    @abstractmethod
    def load(self, directory: Path) -> None:
        """Loads a persisted vector index and metadata from disk."""
        pass

