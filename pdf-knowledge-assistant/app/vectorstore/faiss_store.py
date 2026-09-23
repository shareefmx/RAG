"""FAISS Vector Store Implementation.

Provides in-process, high-performance vector indexing using FAISS IndexFlatIP.
Stores chunk metadata, supports cosine similarity scoring, metadata filtering,
document deletion, and disk persistence.
"""

import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional
import numpy as np

import faiss

from app.ingestion.chunker import DocumentChunk
from app.vectorstore.base import BaseVectorStore, SearchResult


logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """FAISS-backed vector database with metadata association."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._lock = threading.RLock()
        self.index: faiss.Index = faiss.IndexFlatIP(self.dimension)
        # Mapping from sequential internal index id (int) to DocumentChunk
        self._chunks: List[DocumentChunk] = []

    def total_chunks(self) -> int:
        """Returns current total count of indexed chunks."""
        with self._lock:
            return len(self._chunks)

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Adds chunks and their float32 normalized embeddings to the FAISS index."""
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match index dimension ({self.dimension})."
            )

        # Ensure embeddings are float32 and contiguous
        embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)

        with self._lock:
            self.index.add(embeddings_f32)
            self._chunks.extend(chunks)
            logger.info(
                "Added %d chunks to FAISS index. Total indexed chunks: %d",
                len(chunks),
                len(self._chunks),
            )

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Searches for top-k chunks closest to query_embedding with optional metadata filtering."""
        with self._lock:
            total = len(self._chunks)
            if total == 0:
                return []

            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            query_f32 = np.ascontiguousarray(query_embedding, dtype=np.float32)

            # If filtering is required, search more candidates to ensure top-k valid matches
            search_k = min(total, k * 4 if filter_dict else k)
            scores, indices = self.index.search(query_f32, search_k)

            results: List[SearchResult] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._chunks):
                    continue

                chunk = self._chunks[idx]

                # Apply metadata filtering if specified
                if filter_dict:
                    match = True
                    for key, expected_val in filter_dict.items():
                        if getattr(chunk, key, None) != expected_val:
                            match = False
                            break
                    if not match:
                        continue

                results.append(SearchResult(chunk=chunk, score=float(score)))

                if len(results) >= k:
                    break

            return results

    def delete_document(self, document_id: str) -> int:
        """Removes all chunks belonging to a document and rebuilds the index."""
        with self._lock:
            remaining_chunks = [c for c in self._chunks if c.document_id != document_id]
            deleted_count = len(self._chunks) - len(remaining_chunks)

            if deleted_count == 0:
                logger.info("No chunks found with document_id '%s' to delete.", document_id)
                return 0

            # Rebuild FAISS index from remaining chunks
            new_index = faiss.IndexFlatIP(self.dimension)
            if remaining_chunks:
                # Extract embeddings of remaining chunks by index
                # (FAISS IndexFlatIP allows reconstructing vectors)
                remaining_indices = [i for i, c in enumerate(self._chunks) if c.document_id != document_id]
                vectors = np.empty((len(remaining_indices), self.dimension), dtype=np.float32)
                for new_idx, old_idx in enumerate(remaining_indices):
                    vectors[new_idx] = self.index.reconstruct(int(old_idx))
                new_index.add(vectors)

            self.index = new_index
            self._chunks = remaining_chunks
            logger.info("Deleted %d chunks for document '%s'. Remaining chunks: %d", deleted_count, document_id, len(self._chunks))
            return deleted_count

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns unique document summaries currently indexed."""
        with self._lock:
            docs: Dict[str, Dict[str, Any]] = {}
            for chunk in self._chunks:
                doc_id = chunk.document_id
                if doc_id not in docs:
                    docs[doc_id] = {
                        "document_id": doc_id,
                        "filename": chunk.filename,
                        "chunk_count": 0,
                        "pages": set(),
                        "sections": set(),
                    }
                docs[doc_id]["chunk_count"] += 1
                docs[doc_id]["pages"].add(chunk.page)
                if chunk.section:
                    docs[doc_id]["sections"].add(chunk.section)

            return [
                {
                    "document_id": d["document_id"],
                    "filename": d["filename"],
                    "chunk_count": d["chunk_count"],
                    "page_count": len(d["pages"]),
                    "sections": sorted(list(d["sections"])),
                }
                for d in docs.values()
            ]

    def save(self, directory: Path) -> None:
        """Persists the FAISS index binary and chunks metadata to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with self._lock:
            index_file = directory / "index.faiss"
            metadata_file = directory / "metadata.json"

            faiss.write_index(self.index, str(index_file))

            serialized_chunks = [chunk.to_dict() for chunk in self._chunks]
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(serialized_chunks, f, indent=2, ensure_ascii=False)

            logger.info("Saved FAISS index (%d chunks) to '%s'", len(self._chunks), directory)

    def load(self, directory: Path) -> None:
        """Loads a persisted FAISS index and chunks metadata from disk."""
        directory = Path(directory)
        index_file = directory / "index.faiss"
        metadata_file = directory / "metadata.json"

        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"FAISS index files not found in '{directory}'")

        with self._lock:
            loaded_index = faiss.read_index(str(index_file))
            with open(metadata_file, "r", encoding="utf-8") as f:
                raw_chunks = json.load(f)

            loaded_chunks = [
                DocumentChunk(
                    chunk_id=c["chunk_id"],
                    document_id=c["document_id"],
                    filename=c["filename"],
                    page=c["page"],
                    section=c["section"],
                    chunk_index=c["chunk_index"],
                    text=c["text"],
                    metadata=c.get("metadata", {}),
                )
                for c in raw_chunks
            ]

            self.index = loaded_index
            self._chunks = loaded_chunks
            self.dimension = loaded_index.d
            logger.info("Loaded FAISS index (%d chunks) from '%s'", len(self._chunks), directory)

