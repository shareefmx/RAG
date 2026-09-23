"""Unit tests for FAISS and Chroma vector stores."""

from pathlib import Path
import numpy as np
import pytest

from app.ingestion.chunker import DocumentChunk
from app.vectorstore.faiss_store import FAISSVectorStore
from app.vectorstore.chroma_store import ChromaVectorStore


@pytest.fixture
def sample_chunks():
    c1 = DocumentChunk(
        chunk_id="doc1_chunk_001",
        document_id="doc1",
        filename="paper1.pdf",
        page=1,
        section="Introduction",
        chunk_index=1,
        text="Deep neural networks achieve high accuracy on computer vision tasks.",
    )
    c2 = DocumentChunk(
        chunk_id="doc1_chunk_002",
        document_id="doc1",
        filename="paper1.pdf",
        page=2,
        section="Methodology",
        chunk_index=2,
        text="We optimize using Adam with cosine learning rate schedule.",
    )
    c3 = DocumentChunk(
        chunk_id="doc2_chunk_001",
        document_id="doc2",
        filename="paper2.pdf",
        page=1,
        section="Abstract",
        chunk_index=1,
        text="Vector similarity search using FAISS enables scalable retrieval.",
    )
    return [c1, c2, c3]


@pytest.fixture
def synthetic_embeddings():
    # 3 normalized 4-dimensional synthetic vectors
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    return np.vstack([v1, v2, v3])


def test_faiss_search_and_metadata_filter(sample_chunks, synthetic_embeddings):
    store = FAISSVectorStore(dimension=4)
    store.add_chunks(sample_chunks, synthetic_embeddings)

    assert store.total_chunks() == 3

    # Query matching vector 1 exactly
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = store.similarity_search(query, k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "doc1_chunk_001"
    assert pytest.approx(results[0].score, abs=1e-4) == 1.0

    # Query with metadata filter for doc2 only
    filtered_results = store.similarity_search(query, k=2, filter_dict={"document_id": "doc2"})
    assert len(filtered_results) == 1
    assert filtered_results[0].chunk.document_id == "doc2"


def test_faiss_delete_and_save_load(tmp_path: Path, sample_chunks, synthetic_embeddings):
    store = FAISSVectorStore(dimension=4)
    store.add_chunks(sample_chunks, synthetic_embeddings)

    # Save to disk
    save_dir = tmp_path / "faiss_save"
    store.save(save_dir)

    # Load in a fresh store
    loaded_store = FAISSVectorStore(dimension=4)
    loaded_store.load(save_dir)
    assert loaded_store.total_chunks() == 3

    # Delete doc1
    deleted = loaded_store.delete_document("doc1")
    assert deleted == 2
    assert loaded_store.total_chunks() == 1

    docs = loaded_store.list_documents()
    assert len(docs) == 1
    assert docs[0]["document_id"] == "doc2"


def test_chroma_store_basic(sample_chunks, synthetic_embeddings):
    store = ChromaVectorStore(collection_name="test_col")
    store.add_chunks(sample_chunks, synthetic_embeddings)

    assert store.total_chunks() == 3

    query = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    results = store.similarity_search(query, k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "doc2_chunk_001"
    assert results[0].score > 0.95

