"""FastAPI Application Main Entrypoint.

Exposes REST APIs for health checking, document upload/indexing, and RAG queries.
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_health import router as health_router
from app.api.routes_query import router as query_router
from app.api.routes_upload import router as upload_router
from app.config import get_settings, setup_logging
from app.state import get_app_state


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context: initialize logging and warm up services on startup."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Initializing PDF Knowledge Assistant API server...")
    # Trigger lazy singleton initialization
    state = get_app_state()
    logger.info("Services ready. Vector store contains %d chunks.", state.vector_store.total_chunks())
    yield
    logger.info("Shutting down PDF Knowledge Assistant API server.")


app = FastAPI(
    title="📚 PDF Knowledge Assistant API",
    description="Production-grade Retrieval-Augmented Generation (RAG) backend for PDF documents.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend and development origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(health_router)
app.include_router(upload_router)
app.include_router(query_router)


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )

