"""Unit tests for EmbeddingService."""

import numpy as np
import pytest

from app.embeddings.embedding_service import EmbeddingService


@pytest.fixture(scope="module")
def embedder():
    """Shared fixture for embedding service to avoid reloading the model per test."""
    return EmbeddingService(model_name="sentence-transformers/all-MiniLM-L6-v2")


def test_embedding_dimensions_and_normalization(embedder: EmbeddingService):
    assert embedder.dimension == 384

    docs = ["This is a test document about artificial intelligence.", "Fast vector search with FAISS."]
    embeddings = embedder.embed_documents(docs)

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32

    # Verify L2 normalization: Euclidean norm of each vector should equal 1.0
    for vec in embeddings:
        norm = np.linalg.norm(vec)
        assert pytest.approx(norm, abs=1e-5) == 1.0


def test_semantic_similarity(embedder: EmbeddingService):
    query_vec = embedder.embed_query("What is machine learning?")

    doc_similar = "Machine learning is a subfield of artificial intelligence focused on data and algorithms."
    doc_unrelated = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."

    docs_vec = embedder.embed_documents([doc_similar, doc_unrelated])
    scores = embedder.compute_similarity(query_vec, docs_vec)

    # Cosine similarity for the ML doc should be significantly higher than for the Eiffel Tower doc
    assert scores[0] > scores[1]
    assert scores[0] > 0.60
    assert scores[1] < 0.30


def test_empty_query_raises_value_error(embedder: EmbeddingService):
    with pytest.raises(ValueError):
        embedder.embed_query("   ")

