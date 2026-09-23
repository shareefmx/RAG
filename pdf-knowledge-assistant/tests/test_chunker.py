"""Unit tests for Document Chunker and Section Detection."""

import pytest

from app.ingestion.chunker import RecursiveChunker, DocumentChunk
from app.ingestion.metadata import SectionDetector
from app.ingestion.pdf_loader import DocumentPage


def test_section_detector_heuristics():
    assert SectionDetector.detect_heading("1. Introduction") == "1. Introduction"
    assert SectionDetector.detect_heading("Methodology") == "Methodology"
    assert SectionDetector.detect_heading("RESULTS AND DISCUSSION") == "RESULTS AND DISCUSSION"
    # Not a heading because it ends with a period and looks like a sentence
    assert SectionDetector.detect_heading("This is a complete sentence explaining findings.") is None


def test_recursive_chunker_basic():
    text = (
        "Introduction\n\n"
        "Retrieval-augmented generation (RAG) is an architectural pattern for question answering.\n"
        "It combines dense vector retrieval with LLM synthesis.\n\n"
        "Methodology\n\n"
        "We evaluate chunk sizes ranging from 400 to 1200 characters.\n"
        "Sentence embeddings are produced via MiniLM."
    )
    page = DocumentPage(
        document_id="doc_test_123",
        filename="test.pdf",
        page_number=1,
        total_pages=1,
        text=text,
    )

    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=40)
    chunks = chunker.chunk_page(page)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)
        assert chunk.document_id == "doc_test_123"
        assert chunk.page == 1
        assert chunk.filename == "test.pdf"
        assert chunk.chunk_id.startswith("doc_test_123_chunk_")
        assert len(chunk.text) <= 250  # roughly within chunk size target


def test_recursive_chunker_invalid_overlap():
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=500, chunk_overlap=500)

    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=500, chunk_overlap=600)


def test_chunk_documents_continuous_indexing():
    page1 = DocumentPage(
        document_id="doc_abc",
        filename="multipage.pdf",
        page_number=1,
        total_pages=2,
        text="Page 1: Introduction to AI systems and vector databases.",
    )
    page2 = DocumentPage(
        document_id="doc_abc",
        filename="multipage.pdf",
        page_number=2,
        total_pages=2,
        text="Page 2: Evaluation results and hit rate benchmarking.",
    )

    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    all_chunks = chunker.chunk_documents([page1, page2])

    assert len(all_chunks) == 2
    assert all_chunks[0].chunk_index == 1
    assert all_chunks[0].chunk_id == "doc_abc_chunk_001"
    assert all_chunks[0].page == 1

    assert all_chunks[1].chunk_index == 2
    assert all_chunks[1].chunk_id == "doc_abc_chunk_002"
    assert all_chunks[1].page == 2
