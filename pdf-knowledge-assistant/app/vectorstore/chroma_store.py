"""Chroma Vector Store Implementation.

Implements BaseVectorStore using ChromaDB with metadata filtering and persistence.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None  # type: ignore

from app.ingestion.chunker import DocumentChunk
from app.vectorstore.base import BaseVectorStore, SearchResult


logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-backed vector database implementation."""

    def __init__(
        self,
        collection_name: str = "pdf_knowledge",
        persist_directory: Optional[Path] = None,
    ):
        if chromadb is None:
            raise ImportError("chromadb is not installed. Please install it with: pip install chromadb")

        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory) if persist_directory else None

        if self.persist_directory:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        else:
            self.client = chromadb.EphemeralClient()

        # Chroma handles cosine similarity when metadata is set to cosine space
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Initialized ChromaVectorStore (Collection: '%s')", self.collection_name)

    def total_chunks(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        embeddings_list = embeddings.tolist()
        metadatas = [
            {
                "document_id": c.document_id,
                "filename": c.filename,
                "page": c.page,
                "section": c.section,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Added %d chunks to Chroma collection '%s'", len(chunks), self.collection_name)

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if self.collection.count() == 0:
            return []

        if query_embedding.ndim == 2:
            query_list = query_embedding[0].tolist()
        else:
            query_list = query_embedding.tolist()

        where_clause = filter_dict if filter_dict else None

        response = self.collection.query(
            query_embeddings=[query_list],
            n_results=min(k, self.collection.count()),
            where=where_clause,
        )

        results: List[SearchResult] = []
        ids = response.get("ids", [[]])[0]
        distances = response.get("distances", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]

        for chunk_id, distance, doc_text, meta in zip(ids, distances, documents, metadatas):
            # Chroma cosine distance = 1 - cosine_similarity. Convert to cosine similarity:
            similarity = 1.0 - float(distance)
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=meta.get("document_id", ""),
                filename=meta.get("filename", ""),
                page=int(meta.get("page", 1)),
                section=meta.get("section", "General"),
                chunk_index=int(meta.get("chunk_index", 1)),
                text=doc_text,
                metadata=meta,
            )
            results.append(SearchResult(chunk=chunk, score=similarity))

        return results

    def delete_document(self, document_id: str) -> int:
        initial = self.collection.count()
        self.collection.delete(where={"document_id": document_id})
        after = self.collection.count()
        deleted = initial - after
        logger.info("Deleted %d chunks for document '%s' from Chroma", deleted, document_id)
        return deleted

    def list_documents(self) -> List[Dict[str, Any]]:
        count = self.collection.count()
        if count == 0:
            return []

        data = self.collection.get(include=["metadatas"])
        docs: Dict[str, Dict[str, Any]] = {}
        for meta in data.get("metadatas", []):
            doc_id = meta.get("document_id")
            if not doc_id:
                continue
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta.get("filename", "unknown"),
                    "chunk_count": 0,
                    "pages": set(),
                    "sections": set(),
                }
            docs[doc_id]["chunk_count"] += 1
            docs[doc_id]["pages"].add(meta.get("page", 1))
            if meta.get("section"):
                docs[doc_id]["sections"].add(meta.get("section"))

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
        # Chroma with PersistentClient automatically syncs to persist_directory
        logger.info("Chroma persistence active at '%s'", self.persist_directory)

    def load(self, directory: Path) -> None:
        self.persist_directory = Path(directory)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

