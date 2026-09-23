"""PDF Document Extraction Module.

Uses PyMuPDF (fitz) for fast, robust text and page-level extraction.
Preserves page numbers, generates unique document IDs, and validates file integrity.
"""

from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import pymupdf as fitz


logger = logging.getLogger(__name__)


class PDFLoadingError(Exception):
    """Base exception for PDF loading errors."""
    pass


class InvalidPDFError(PDFLoadingError):
    """Raised when the uploaded file is not a valid PDF."""
    pass


class EmptyPDFError(PDFLoadingError):
    """Raised when the PDF contains no extractable text."""
    pass


@dataclass
class DocumentPage:
    """Represents text extracted from a single page of a PDF document."""

    document_id: str
    filename: str
    page_number: int  # 1-indexed for human readability and citations
    total_pages: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert page representation to a serializable dictionary."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "total_pages": self.total_pages,
            "text": self.text,
            "metadata": self.metadata,
        }


class PDFLoader:
    """Extracts text and page-level metadata from PDF files using PyMuPDF."""

    def __init__(self, max_file_size_mb: int = 50):
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def validate_file(self, file_path: Path) -> None:
        """Validates that the file exists, has a valid size, and starts with PDF magic bytes."""
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {file_path}")

        file_size = file_path.stat().st_size
        if file_size == 0:
            raise InvalidPDFError(f"PDF file '{file_path.name}' is empty (0 bytes).")

        if file_size > self.max_file_size_bytes:
            raise InvalidPDFError(
                f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed limit ({self.max_file_size_bytes / (1024*1024):.0f} MB)."
            )

        # Check PDF magic bytes (%PDF-)
        with open(file_path, "rb") as f:
            header = f.read(5)
            if header != b"%PDF-":
                raise InvalidPDFError(f"File '{file_path.name}' is not a valid PDF document (missing %PDF- header).")

    @staticmethod
    def generate_document_id(file_path: Path) -> str:
        """Generates a stable, collision-free unique document ID using UUID4 and path hash."""
        random_suffix = uuid.uuid4().hex[:8]
        path_hash = hashlib.sha256(file_path.name.encode("utf-8")).hexdigest()[:6]
        return f"doc_{path_hash}_{random_suffix}"

    def load(self, file_path: str | Path, document_id: Optional[str] = None) -> List[DocumentPage]:
        """Loads and extracts text page-by-page from a PDF file.

        Args:
            file_path: Path to the target PDF file.
            document_id: Optional unique identifier. If not provided, one is generated.

        Returns:
            List of DocumentPage objects containing page text and citation metadata.

        Raises:
            InvalidPDFError: When the file is corrupt or not a valid PDF.
            EmptyPDFError: When no extractable text is found across any page.
        """
        path = Path(file_path).resolve()
        self.validate_file(path)

        doc_id = document_id or self.generate_document_id(path)
        filename = path.name

        logger.info("Starting text extraction for document '%s' (ID: %s)", filename, doc_id)

        try:
            doc = fitz.open(str(path))
        except Exception as e:
            logger.error("PyMuPDF failed to open PDF '%s': %s", filename, str(e))
            raise InvalidPDFError(f"Failed to open PDF document: {e}") from e

        try:
            total_pages = len(doc)
            if total_pages == 0:
                raise InvalidPDFError(f"PDF document '{filename}' contains 0 pages.")

            # Extract document-level metadata from PDF dictionary
            pdf_metadata = doc.metadata or {}
            cleaned_doc_metadata = {
                "title": pdf_metadata.get("title") or filename,
                "author": pdf_metadata.get("author") or "Unknown",
                "subject": pdf_metadata.get("subject") or "",
                "creator": pdf_metadata.get("creator") or "",
                "producer": pdf_metadata.get("producer") or "",
                "creation_date": pdf_metadata.get("creationDate") or "",
            }

            pages: List[DocumentPage] = []
            total_char_count = 0

            for page_index in range(total_pages):
                page = doc[page_index]
                # Page numbers are 1-indexed for standard human document citation
                page_number = page_index + 1

                # Extract text using PyMuPDF layout-preserving text mode
                page_text = page.get_text("text") or ""
                total_char_count += len(page_text.strip())

                page_meta = {
                    **cleaned_doc_metadata,
                    "page_width": page.rect.width,
                    "page_height": page.rect.height,
                }

                pages.append(
                    DocumentPage(
                        document_id=doc_id,
                        filename=filename,
                        page_number=page_number,
                        total_pages=total_pages,
                        text=page_text,
                        metadata=page_meta,
                    )
                )

            if total_char_count == 0:
                logger.warning("PDF document '%s' contains %d pages but 0 extractable text characters.", filename, total_pages)
                raise EmptyPDFError(
                    f"No extractable text was found in '{filename}'. "
                    "The document may be scanned or image-only without an embedded text layer."
                )

            logger.info(
                "Successfully extracted %d pages (%d total characters) from '%s'",
                total_pages,
                total_char_count,
                filename,
            )
            return pages

        finally:
            doc.close()

