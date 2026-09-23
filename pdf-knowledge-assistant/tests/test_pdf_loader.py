"""Unit tests for PDFLoader."""

from pathlib import Path
import pymupdf as fitz
import pytest

from app.ingestion.pdf_loader import (
    PDFLoader,
    DocumentPage,
    InvalidPDFError,
    EmptyPDFError,
)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Creates a temporary 2-page sample PDF with structured text."""
    pdf_path = tmp_path / "sample_research.pdf"
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page()
    page1.insert_text(
        (50, 72),
        "Introduction to Retrieval-Augmented Generation\n\n"
        "RAG combines parametric knowledge in neural models with non-parametric external retrieval.\n"
        "This prevents hallucinations and provides direct citations.",
    )

    # Page 2
    page2 = doc.new_page()
    page2.insert_text(
        (50, 72),
        "Methodology and Experimental Setup\n\n"
        "We tested FAISS vector indices with sentence-transformer embeddings.\n"
        "Evaluation demonstrated a 94% hit rate on domain queries.",
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    """Creates a temporary blank PDF with no text."""
    pdf_path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def corrupt_pdf(tmp_path: Path) -> Path:
    """Creates a corrupted non-PDF file."""
    bad_file = tmp_path / "fake.pdf"
    bad_file.write_text("This is not a PDF header.")
    return bad_file


def test_pdf_loader_success(sample_pdf: Path):
    loader = PDFLoader()
    pages = loader.load(sample_pdf)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[0].total_pages == 2
    assert "Retrieval-Augmented Generation" in pages[0].text
    assert pages[0].filename == "sample_research.pdf"
    assert pages[0].document_id.startswith("doc_")

    assert pages[1].page_number == 2
    assert "Methodology" in pages[1].text
    assert pages[1].document_id == pages[0].document_id


def test_pdf_loader_custom_doc_id(sample_pdf: Path):
    loader = PDFLoader()
    pages = loader.load(sample_pdf, document_id="doc_custom_test_999")
    assert all(p.document_id == "doc_custom_test_999" for p in pages)


def test_pdf_loader_empty_pdf(empty_pdf: Path):
    loader = PDFLoader()
    with pytest.raises(EmptyPDFError) as exc_info:
        loader.load(empty_pdf)
    assert "No extractable text was found" in str(exc_info.value)


def test_pdf_loader_corrupt_file(corrupt_pdf: Path):
    loader = PDFLoader()
    with pytest.raises(InvalidPDFError) as exc_info:
        loader.load(corrupt_pdf)
    assert "missing %PDF- header" in str(exc_info.value)


def test_pdf_loader_missing_file(tmp_path: Path):
    loader = PDFLoader()
    missing = tmp_path / "non_existent.pdf"
    with pytest.raises(FileNotFoundError):
        loader.load(missing)

