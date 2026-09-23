"""Document upload, indexing, and management endpoints."""

import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.ingestion.pdf_loader import InvalidPDFError, EmptyPDFError
from app.state import AppState, get_app_state


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


class IndexResponse(BaseModel):
    document_id: str
    filename: str
    pages_extracted: int
    chunks_created: int
    total_indexed_chunks: int


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    page_count: int
    sections: List[str]


class DeleteResponse(BaseModel):
    document_id: str
    deleted_chunks: int
    message: str


@router.post("/upload", response_model=IndexResponse)
async def upload_and_index_document(
    file: UploadFile = File(...),
    state: AppState = Depends(get_app_state),
):
    """Uploads a PDF document and immediately indexes it into the vector database."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF documents (.pdf) are supported."
        )

    upload_dir = state.settings.get_upload_path()
    safe_filename = Path(file.filename).name
    destination = upload_dir / safe_filename

    # Save uploaded bytes to disk safely
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error("Failed to save uploaded file '%s': %s", safe_filename, e)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Process and index the document
    try:
        result = state.index_pdf_file(destination)
        return IndexResponse(**result)

    except (InvalidPDFError, EmptyPDFError) as e:
        if destination.exists():
            destination.unlink()
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error("Indexing failed for '%s': %s", safe_filename, e)
        raise HTTPException(status_code=500, detail=f"Document indexing failed: {e}")


@router.get("", response_model=List[DocumentSummary])
def list_indexed_documents(state: AppState = Depends(get_app_state)):
    """Lists all documents currently indexed in the vector store with page and section statistics."""
    docs = state.vector_store.list_documents()
    return [DocumentSummary(**d) for d in docs]


@router.delete("/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str, state: AppState = Depends(get_app_state)):
    """Deletes all chunks associated with a specific document ID from the vector database."""
    deleted_count = state.vector_store.delete_document(document_id)
    if deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Document ID '{document_id}' not found in index."
        )

    # Persist updated vector store
    state.vector_store.save(state.settings.get_vectorstore_path())

    return DeleteResponse(
        document_id=document_id,
        deleted_chunks=deleted_count,
        message=f"Successfully removed document '{document_id}' ({deleted_count} chunks deleted)."
    )

