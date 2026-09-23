"""Unit and API integration tests for FastAPI endpoints."""

from pathlib import Path
from fastapi.testclient import TestClient
import pymupdf as fitz
import pytest

from app.main import app
from app.state import get_app_state


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture
def sample_pdf_bytes():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "FastAPI RAG Test Document\n\nThis API indexes PDF files and answers questions.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_api_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app_version"] == "1.0.0"
    assert "llm_provider" in data


def test_api_upload_and_query_flow(client: TestClient, sample_pdf_bytes: bytes):
    # 1. Upload & Index PDF
    files = {"file": ("api_test.pdf", sample_pdf_bytes, "application/pdf")}
    upload_res = client.post("/documents/upload", files=files)
    assert upload_res.status_code == 200
    index_data = upload_res.json()
    assert index_data["filename"] == "api_test.pdf"
    assert index_data["chunks_created"] >= 1
    doc_id = index_data["document_id"]

    # 2. List documents
    list_res = client.get("/documents")
    assert list_res.status_code == 200
    docs = list_res.json()
    assert any(d["document_id"] == doc_id for d in docs)

    # 3. Query
    query_payload = {
        "question": "What does this API do?",
        "top_k": 3,
        "document_id": doc_id,
    }
    query_res = client.post("/query", json=query_payload)
    assert query_res.status_code == 200
    res_data = query_res.json()
    assert "question" in res_data
    assert "answer" in res_data
    assert res_data["has_sufficient_context"] is True
    assert len(res_data["sources"]) >= 1

    # 4. Clean up / Delete document
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted_chunks"] >= 1

