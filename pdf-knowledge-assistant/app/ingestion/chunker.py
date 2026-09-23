"""Recursive Text Chunker for Document Ingestion.

Splits text into semantically cohesive chunks using hierarchical separators,
retains sliding overlap to preserve boundary context, and generates unique chunk IDs.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

from app.ingestion.metadata import SectionDetector
from app.ingestion.pdf_loader import DocumentPage
from app.ingestion.text_cleaner import TextCleaner


logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents an atomic, searchable text segment with citation metadata."""

    chunk_id: str
    document_id: str
    filename: str
    page: int
    section: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk into a dictionary representation for serialization and API responses."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page": self.page,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "metadata": self.metadata,
        }


class RecursiveChunker:
    """Recursively splits text on natural boundaries (paragraphs, lines, sentences, words)."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.cleaner = TextCleaner()

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text using the first matching separator."""
        final_chunks: List[str] = []

        if not separators:
            return [text] if text else []

        separator = separators[0]
        new_separators = separators[1:]

        # Split text by current separator
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            # Reattach the separator unless it's empty string or newline
            sub_text = s if separator in ("\n\n", "\n", "") else s + separator
            if len(sub_text) <= self.chunk_size:
                good_splits.append(sub_text)
            else:
                # If chunk is still larger than chunk_size, split further with remaining separators
                if new_separators:
                    deeper_splits = self._split_text(sub_text, new_separators)
                    good_splits.extend(deeper_splits)
                else:
                    # Hard character split if no separators left
                    for i in range(0, len(sub_text), self.chunk_size - self.chunk_overlap):
                        good_splits.append(sub_text[i : i + self.chunk_size])

        # Merge small splits up to chunk_size with chunk_overlap
        current_chunk: List[str] = []
        current_len = 0

        for split in good_splits:
            split_len = len(split)
            if current_len + split_len > self.chunk_size and current_chunk:
                merged = "".join(current_chunk).strip()
                if merged:
                    final_chunks.append(merged)

                # Keep overlap from the end of the current chunk
                overlap_text = ""
                for item in reversed(current_chunk):
                    if len(overlap_text) + len(item) <= self.chunk_overlap:
                        overlap_text = item + overlap_text
                    else:
                        break

                current_chunk = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)

            current_chunk.append(split)
            current_len += split_len

        if current_chunk:
            merged = "".join(current_chunk).strip()
            if merged:
                final_chunks.append(merged)

        return final_chunks

    def chunk_page(
        self,
        page: DocumentPage,
        start_chunk_index: int = 1,
        active_section: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Splits a single document page into chunks, retaining page and section context."""
        cleaned_text = self.cleaner.clean(page.text)
        if not cleaned_text:
            return []

        # Detect section if not already carried over or if a new heading appears
        detected_section = SectionDetector.extract_sections_from_page(
            cleaned_text,
            default_section=active_section or "General"
        )

        raw_chunks = self._split_text(cleaned_text, self.separators)
        chunks: List[DocumentChunk] = []

        for i, text_segment in enumerate(raw_chunks):
            chunk_idx = start_chunk_index + i
            chunk_id = f"{page.document_id}_chunk_{chunk_idx:03d}"

            # Check if this specific chunk contains a section heading
            chunk_section = SectionDetector.extract_sections_from_page(
                text_segment,
                default_section=detected_section
            )

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=page.document_id,
                filename=page.filename,
                page=page.page_number,
                section=chunk_section,
                chunk_index=chunk_idx,
                text=text_segment,
                metadata={
                    **page.metadata,
                    "char_count": len(text_segment),
                    "word_count": len(text_segment.split()),
                    "total_pages": page.total_pages,
                },
            )
            chunks.append(chunk)

        return chunks

    def chunk_documents(self, pages: List[DocumentPage]) -> List[DocumentChunk]:
        """Processes a sequence of document pages into a global list of numbered chunks."""
        all_chunks: List[DocumentChunk] = []
        current_chunk_idx = 1
        current_section = "General"

        for page in pages:
            page_chunks = self.chunk_page(
                page,
                start_chunk_index=current_chunk_idx,
                active_section=current_section,
            )
            if page_chunks:
                current_section = page_chunks[-1].section
                current_chunk_idx += len(page_chunks)
                all_chunks.extend(page_chunks)

        logger.info(
            "Created %d chunks from %d pages for document ID '%s'",
            len(all_chunks),
            len(pages),
            pages[0].document_id if pages else "empty",
        )
        return all_chunks

