"""Application Shared State and Service Dependency Provider.

Manages singletons for the vector store, embedding service, and RAG pipeline.
"""

from functools import lru_cache
import logging
from pathlib import Path
from typing import Optional

from app.config import Settings, get_settings
from app.embeddings.embedding_service import EmbeddingService, get_embedding_service
from app.generation.llm import LLMServiceFactory
from app.ingestion.chunker import RecursiveChunker
from app.ingestion.pdf_loader import PDFLoader
from app.rag.pipeline import RAGPipeline
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import DocumentRetriever
from app.vectorstore.base import BaseVectorStore
from app.vectorstore.chroma_store import ChromaVectorStore
from app.vectorstore.faiss_store import FAISSVectorStore


logger = logging.getLogger(__name__)


class AppState:
    """Encapsulates active services and indices across the application lifetime."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.pdf_loader = PDFLoader()
        self.chunker = RecursiveChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        # 1. Embedding Service
        self.embedding_service = EmbeddingService(
            model_name=self.settings.embedding_model
        )

        # 2. Vector Database
        self.vector_store: BaseVectorStore = self._init_vector_store()

        # 3. Retriever
        self.retriever = DocumentRetriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            similarity_threshold=self.settings.similarity_threshold,
            default_top_k=self.settings.top_k,
        )

        # 4. LLM Service
        self.llm_service = LLMServiceFactory.create(
            provider=self.settings.llm_provider,
            gemini_api_key=self.settings.gemini_api_key,
            nvidia_api_key=self.settings.nvidia_api_key,
            openrouter_api_key=self.settings.openrouter_api_key,
            model_name=self.settings.llm_model,
            nvidia_model=self.settings.nvidia_model,
        )

        # 5. Optional Reranker
        self.reranker = None
        if self.settings.use_reranker:
            try:
                self.reranker = CrossEncoderReranker(model_name=self.settings.reranker_model)
            except Exception as e:
                logger.warning("Could not initialize reranker: %s", e)

        # 6. Optional Query Rewriter
        self.query_rewriter = None
        if self.settings.enable_query_rewriting:
            self.query_rewriter = QueryRewriter(llm_service=self.llm_service)

        # 7. RAG Pipeline
        self.pipeline = RAGPipeline(
            retriever=self.retriever,
            llm_service=self.llm_service,
            reranker=self.reranker,
            query_rewriter=self.query_rewriter,
        )

    def _init_vector_store(self) -> BaseVectorStore:
        store_type = self.settings.vector_store.lower()
        store_dir = self.settings.get_vectorstore_path()

        if store_type == "chroma":
            logger.info("Initializing Chroma vector store at '%s'", store_dir)
            return ChromaVectorStore(persist_directory=store_dir)
        else:
            logger.info("Initializing FAISS vector store with dimension %d", self.embedding_service.dimension)
            faiss_store = FAISSVectorStore(dimension=self.embedding_service.dimension)
            # Attempt to restore if persisted files exist
            if (store_dir / "index.faiss").exists() and (store_dir / "metadata.json").exists():
                try:
                    faiss_store.load(store_dir)
                    logger.info("Restored existing FAISS index with %d chunks", faiss_store.total_chunks())
                except Exception as e:
                    logger.warning("Could not restore FAISS index: %s", e)
            return faiss_store

    def index_pdf_file(self, file_path: Path) -> dict:
        """Processes a PDF file through extraction, cleaning, chunking, and embedding into the vector store."""
        # Clean up existing index for the same filename if previously indexed
        for doc in self.vector_store.list_documents():
            if doc["filename"] == file_path.name:
                logger.info("Replacing existing index for '%s' (ID: %s)", file_path.name, doc["document_id"])
                self.vector_store.delete_document(doc["document_id"])

        pages = self.pdf_loader.load(file_path)
        chunks = self.chunker.chunk_documents(pages)

        if not chunks:
            raise ValueError(f"No valid text chunks generated from '{file_path.name}'.")

        # Generate embeddings in batch
        texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_documents(texts)

        # Store in vector database
        self.vector_store.add_chunks(chunks, embeddings)

        # Persist index
        try:
            self.vector_store.save(self.settings.get_vectorstore_path())
        except Exception as e:
            logger.warning("Could not auto-save vector store: %s", e)

        return {
            "document_id": pages[0].document_id,
            "filename": file_path.name,
            "pages_extracted": len(pages),
            "chunks_created": len(chunks),
            "total_indexed_chunks": self.vector_store.total_chunks(),
        }

    def reload_vector_store_if_needed(self) -> None:
        """Reloads FAISS vector store from disk if persisted files exist and in-memory is empty."""
        store_dir = self.settings.get_vectorstore_path()
        index_file = store_dir / "index.faiss"
        meta_file = store_dir / "metadata.json"
        if index_file.exists() and meta_file.exists():
            try:
                self.vector_store.load(store_dir)
                logger.info("Synchronized FAISS index from disk (%d chunks)", self.vector_store.total_chunks())
            except Exception as e:
                logger.warning("Could not sync FAISS index from disk: %s", e)


# Global app state singleton
_APP_STATE: Optional[AppState] = None


def get_app_state() -> AppState:
    """Returns application state singleton."""
    global _APP_STATE
    if _APP_STATE is None:
        _APP_STATE = AppState()
    return _APP_STATE
