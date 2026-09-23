"""Application Configuration Module.

Loads and validates settings from environment variables and .env files using Pydantic.
Never hardcodes secrets. Provides structured directory path resolution.
"""

from functools import lru_cache
import logging
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root path (directory containing app/, data/, tests/, etc.)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Global configuration settings for the PDF Knowledge Assistant."""

    # LLM Settings
    llm_provider: Literal["gemini", "openrouter"] = Field(
        default="gemini",
        description="Active LLM provider backend ('gemini' or 'openrouter')"
    )
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key (https://aistudio.google.com/app/apikey)"
    )
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key (https://openrouter.ai/keys)"
    )
    llm_model: str = Field(
        default="gemini-3.5-flash-lite",
        description="Model name to use with the selected provider"
    )

    # Embedding Settings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Hugging Face sentence-transformers model identifier"
    )

    # Vector Database Settings
    vector_store: Literal["faiss", "chroma"] = Field(
        default="faiss",
        description="Vector database backend ('faiss' or 'chroma')"
    )

    # Document Chunking Settings
    chunk_size: int = Field(
        default=800,
        ge=100,
        le=4000,
        description="Target character length for document chunks"
    )
    chunk_overlap: int = Field(
        default=150,
        ge=0,
        le=1000,
        description="Character overlap between consecutive chunks"
    )

    # Retrieval Settings
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of final context chunks to feed into the generation prompt"
    )
    candidate_top_k: int = Field(
        default=15,
        ge=1,
        le=100,
        description="Number of candidate chunks retrieved before reranking"
    )
    similarity_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold required for out-of-domain rejection gating"
    )

    # Advanced Retrieval: Reranker
    use_reranker: bool = Field(
        default=False,
        description="Whether to execute cross-encoder reranking on candidate chunks"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Hugging Face cross-encoder model identifier"
    )

    # Query Rewriting
    enable_query_rewriting: bool = Field(
        default=True,
        description="Whether to rewrite conversational follow-up questions using chat history"
    )

    # Storage Paths (relative to PROJECT_ROOT or absolute)
    upload_dir: str = Field(
        default="data/uploads",
        description="Path for temporarily staging uploaded PDF documents"
    )
    processed_dir: str = Field(
        default="data/processed",
        description="Path for serialized processed chunks and metadata"
    )
    vectorstore_dir: str = Field(
        default="data/vectorstore",
        description="Path where vector indices and document stores are persisted"
    )

    # Server Settings
    app_host: str = Field(default="0.0.0.0", description="Host to bind FastAPI server")
    app_port: int = Field(default=8000, description="Port to bind FastAPI server")
    log_level: str = Field(default="INFO", description="Application log level")

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_upload_path(self) -> Path:
        """Returns resolved upload directory path and ensures it exists."""
        path = (PROJECT_ROOT / self.upload_dir).resolve() if not Path(self.upload_dir).is_absolute() else Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_processed_path(self) -> Path:
        """Returns resolved processed data directory path and ensures it exists."""
        path = (PROJECT_ROOT / self.processed_dir).resolve() if not Path(self.processed_dir).is_absolute() else Path(self.processed_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_vectorstore_path(self) -> Path:
        """Returns resolved vector store directory path and ensures it exists."""
        path = (PROJECT_ROOT / self.vectorstore_dir).resolve() if not Path(self.vectorstore_dir).is_absolute() else Path(self.vectorstore_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache()
def get_settings() -> Settings:
    """Returns cached application settings instance."""
    return Settings()


def setup_logging(level: str = "INFO") -> None:
    """Configures structured application logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence noisy third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

